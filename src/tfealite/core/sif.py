import numpy as np

from ..core.dofs import BRANCH_DOFS, HEAVISIDE_DOFS
from ..elements.utils import fill_element_displacement
from ..elements.XQuad4n import XQuad4n
from ..elements.XTri3n import XTri3n
from ..elements.XTetr4n import XTetr4n
from .level_set import CutType, project_on_line, project_on_surface
from scipy.spatial import KDTree
import pyvista as pv


ELEM_FN_MAP = {"Tri3n": XTri3n, "Quad4n": XQuad4n, "Tetr4n": XTetr4n}


def _build_elem_2d(model, level_set, cut_type, element):
    _, elem_type, mat_id, real_id, elem_nodes = element
    elem_nodes = np.asarray(elem_nodes)

    elem_vertices = model.nodes[elem_nodes - 1, 1:3]
    elem_dofs = model.list_dof.get_elem_dofs(elem_nodes)
    real = model.reals[real_id - 1][1]
    local_dofs_per_node = np.bitwise_or.reduce(elem_dofs)
    h_enrich = bool(local_dofs_per_node & HEAVISIDE_DOFS)
    t_enrich = bool(local_dofs_per_node & BRANCH_DOFS)

    tip = model.tip[
        elem_nodes[np.argmax(np.bitwise_and(elem_dofs, BRANCH_DOFS) != 0)] - 1
    ]
    phi_n, phi_t = level_set.get(elem_nodes, tip)

    elem_fn = ELEM_FN_MAP[elem_type]

    elem = elem_fn(
        node_coords=elem_vertices,
        material=model.materials[mat_id - 1][1],
        real=real,
        phi_n=phi_n,
        phi_t=phi_t,
        h_enrich=h_enrich,
        t_enrich=t_enrich,
        partial_cut=(cut_type == CutType.PARTIAL),
        in_range=model.in_range[elem_nodes - 1],
    )
    return elem


def _build_elem_3d(model, level_set, cut_type, element):
    _, elem_type, mat_id, real_id, elem_nodes = element
    elem_nodes = np.asarray(elem_nodes)

    elem_vertices = model.nodes[elem_nodes - 1, 1:4]
    elem_dofs = model.list_dof.get_elem_dofs(elem_nodes)
    real = model.reals[real_id - 1][1]
    local_dofs_per_node = np.bitwise_or.reduce(elem_dofs)
    h_enrich = bool(local_dofs_per_node & HEAVISIDE_DOFS)
    t_enrich = bool(local_dofs_per_node & BRANCH_DOFS)

    tip = model.tip[
        elem_nodes[np.argmax(np.bitwise_and(elem_dofs, BRANCH_DOFS) != 0)] - 1
    ]
    phi_n, phi_t = level_set.get(elem_nodes, tip)

    elem_fn = ELEM_FN_MAP[elem_type]

    elem = elem_fn(
        node_coords=elem_vertices,
        material=model.materials[mat_id - 1][1],
        real=real,
        phi_n=phi_n,
        phi_t=phi_t,
        h_enrich=h_enrich,
        t_enrich=t_enrich,
        partial_cut=(cut_type == CutType.PARTIAL),
        in_range=model.in_range[elem_nodes - 1],
    )
    return elem


class DisplacementCorrelationMethodSIF:
    def __init__(self, kosolov, shear_mod, r, dr):
        self.kosolov = kosolov
        self.shear_mod = shear_mod
        self.r = np.asarray(r)
        self.dr = dr

    def cal_sif(self, level_set, model, cut_info: dict, u_tip: float):
        assert level_set.bspline is not None
        bspline = level_set.bspline

        tip = bspline(u_tip)
        t = bspline(np.array([u_tip]), nu=1)[0]

        if level_set.embedded and u_tip == 0.0:
            t *= -1

        t = t / np.linalg.norm(t)
        n = np.array([-t[1], t[0]])

        p1 = tip[None, :] - self.r[:, None] * t[None, :]

        u_range = np.linspace(0.0, 1.0, 1000)
        u_i = u_range[KDTree(bspline(u_range)).query(p1)[1]]

        project_on_line(
            lambda a=u_i: bspline(a),
            lambda a=u_i: bspline(a, nu=1),
            lambda a=u_i: bspline(a, nu=2),
            u_i,
            p1,
        )

        p1 = bspline(u_i)

        mesh = model.mesh
        cut_elem_ids = np.array(
            [
                elem_id - 1
                for elem_id, (cut_type, _, _) in cut_info.items()
                if cut_type != CutType.NONE
            ]
        )
        cut_mesh = mesh.extract_cells(cut_elem_ids)

        tip_3d = np.append(tip, 0.0)
        p1_3d = np.column_stack((p1, np.zeros(p1.shape[0])))

        tip_elem_idx = cut_mesh.find_containing_cell(tip_3d)
        # assert tip_elem_idx != -1, "Couldn't find element containing tip"
        if tip_elem_idx == -1:
            raise ValueError
        tip_elem_id = cut_elem_ids[tip_elem_idx]
        tip_elem = model.elements[tip_elem_id]

        elem = _build_elem_2d(model, level_set, CutType.PARTIAL, tip_elem)
        tip_nat_coords = elem._cal_tip_nat_coords()
        real_tip_coords = (
            elem._base_shape_functions(tip_nat_coords)[0] @ elem.node_coords
        )

        cell_indices = cut_mesh.find_containing_cell(p1_3d)
        orphans_mask = cell_indices == -1
        if np.any(orphans_mask):
            cell_indices[orphans_mask] = cut_mesh.find_closest_cell(p1_3d[orphans_mask])

        p1_elem_indices = cut_elem_ids[cell_indices]
        # print("p1_elem_indices", p1_elem_indices)

        sort_idx = np.argsort(p1_elem_indices)
        sorted_cells = p1_elem_indices[sort_idx]
        unique_cells, split_indices = np.unique(sorted_cells, return_index=True)

        grouped_point_indices = np.split(sort_idx, split_indices[1:])

        jump = np.full_like(p1, np.nan)
        r_1_star = np.full(p1.shape[0], np.nan)

        for elem_id, point_idx_batch in zip(unique_cells, grouped_point_indices):
            element = model.elements[elem_id]
            cut_type = cut_info[elem_id + 1][0]

            elem = _build_elem_2d(model, level_set, cut_type, element)
            _, nat_coords_batch = elem.nearest_point_on_crack(p1[point_idx_batch])
            Ue = fill_element_displacement(
                np.asarray(element[4]), model.list_dof, model.Ug
            ).reshape((-1, 2))
            jump_shape_fn_batch, r_1_batch = elem.jump_shape_functions(
                nat_coords_batch, real_tip_coords
            )

            jump[point_idx_batch, :] = jump_shape_fn_batch @ Ue
            r_1_star[point_idx_batch] = r_1_batch

        valid_mask = ~np.isnan(r_1_star)
        r_clean = r_1_star[valid_mask]
        jump_clean = jump[valid_mask]

        if len(r_clean) > 1:
            sort_idx = np.argsort(r_clean)
            r_clean = r_clean[sort_idx]
            jump_clean = jump_clean[sort_idx]

            dr_mask = np.insert(np.abs(np.diff(r_clean)) > 1e-10, 0, True)
            r_clean = r_clean[dr_mask]
            jump_clean = jump_clean[dr_mask]

        if len(r_clean) < len(r_1_star):
            print(f"Warning: lost {len(r_1_star) - len(r_clean)} DCM evaluation points")

        if len(r_clean) < 2:
            raise ValueError(
                "Richardson extrapolation requires at least 2 extraction points."
            )

        T_matrix = np.array([t, n])
        jump_local = jump_clean @ T_matrix.T

        coef = (self.shear_mod / (self.kosolov + 1.0)) * np.sqrt(2.0 * np.pi / r_clean)
        K_I_star = coef * jump_local[:, 1]
        K_II_star = coef * jump_local[:, 0]

        r_a, r_b = r_clean[:-1], r_clean[1:]

        extrap_multiplier = r_b / (r_b - r_a)
        r_ratio = r_a / r_b

        K_I_ext = extrap_multiplier * (K_I_star[:-1] - r_ratio * K_I_star[1:])
        K_II_ext = extrap_multiplier * (K_II_star[:-1] - r_ratio * K_II_star[1:])

        return float(np.mean(K_I_ext)), float(np.mean(K_II_ext))


class DisplacementCorrelationMethodSIF3D:
    def __init__(self, kosolov, shear_mod, r, dr):
        self.kosolov = kosolov
        self.shear_mod = shear_mod
        self.r = r
        self.dr = dr

    def cal_sif(self, level_set, model, cut_info: dict, tip_index, v_tip):
        uv_tip = np.stack([np.ones_like(v_tip), v_tip], axis=1)
        tip_surface = level_set.ndbsplines[tip_index][2]
        S = tip_surface(uv_tip)
        tip_points = S.copy()
        print("tip points", tip_points)
        Su = tip_surface(uv_tip, nu=(1, 0))
        Sv = tip_surface(uv_tip, nu=(0, 1))
        n = np.cross(Su, Sv, axis=1)
        n = n / np.linalg.norm(n, axis=1)[:, None]
        t = np.cross(Sv, n, axis=1)
        t = t / np.linalg.norm(t, axis=1)[:, None]
        b = np.cross(t, n, axis=1)
        b = b / np.linalg.norm(b, axis=1)[:, None]

        p1 = S[:, None, :] - self.r[None, :, None] * t[:, None, :]
        print("p1 points", p1)

        N_points = p1.shape[0] * p1.shape[1]
        p1_flat = p1.reshape((-1, 3))
        p1_b = np.repeat(b, p1.shape[1], axis=0)
        p1_tip = np.repeat(S, p1.shape[1], axis=0)

        p1_surface_index = np.full(N_points, -1, dtype=int)
        p1_best_uv = np.full((N_points, 2), np.nan, dtype=np.float64)
        p1_min_dist = np.full(N_points, np.inf, dtype=np.float64)
        p1_best_projection = np.full_like(p1_flat, np.nan, dtype=np.float64)

        for i, (_, _, ndbspline) in enumerate(level_set.ndbsplines):
            v_i_range = np.linspace(0, 1, 1000)
            uv_i_range = np.stack([np.ones_like(v_i_range), v_i_range], axis=1)
            S_i = ndbspline(uv_i_range)

            front_tree_i = KDTree(S_i)
            _, best_indices = front_tree_i.query(p1_flat)

            uv_front = uv_i_range[best_indices]
            S_front = ndbspline(uv_front)
            Su_front = ndbspline(uv_front, nu=(1, 0))
            Sv_front = ndbspline(uv_front, nu=(0, 1))

            n_front = np.cross(Su_front, Sv_front, axis=1)
            t_front = np.cross(Sv_front, n_front, axis=1)
            t_front = t_front / np.linalg.norm(t_front, axis=1)[:, None]

            distance_to_front = p1_flat - S_front
            phi_t = np.sum(distance_to_front * t_front, axis=1)
            behind_mask = phi_t <= 0

            behind_indices = np.where(behind_mask)[0]

            if len(behind_indices) == 0:
                continue

            p1_behind = p1_flat[behind_indices]
            p1_b_behind = p1_b[behind_indices]
            p1_tip_behind = p1_tip[behind_indices]

            # u_grid_vals = np.linspace(0.9, 1, 20)  # Coarse grid is fine
            # v_grid_vals = np.linspace(0, 1, 1000)
            # U_grid, V_grid = np.meshgrid(u_grid_vals, v_grid_vals)
            # uv_surface_grid = np.stack([U_grid.ravel(), V_grid.ravel()], axis=1)
            #
            # S_surface = ndbspline(uv_surface_grid)
            # surface_tree = KDTree(S_surface)
            # _, best_surf_indices = surface_tree.query(p1_behind)

            uv_proj = uv_front[behind_indices].copy()
            # uv_proj = uv_surface_grid[best_surf_indices].copy()

            project_on_surface(
                ndbspline,
                uv_proj,
                p1_behind,
                p1_tip_behind,
                p1_b_behind,
                penalty=1e6,
                independent=True,
                tol=1e-12,
            )

            S_proj = ndbspline(uv_proj)
            dist_to_surface = np.linalg.norm(p1_behind - S_proj, axis=1)

            current_min_dist = p1_min_dist[behind_indices]
            update_mask = np.isnan(current_min_dist) | (
                dist_to_surface < current_min_dist
            )

            global_update_indices = behind_indices[update_mask]

            if len(global_update_indices) > 0:
                p1_min_dist[global_update_indices] = dist_to_surface[update_mask]
                p1_surface_index[global_update_indices] = i
                p1_best_uv[global_update_indices] = uv_proj[update_mask]
                p1_best_projection[global_update_indices] = S_proj[update_mask]

        print("p1_best_projection", p1_best_projection)
        p1_mesh = pv.PolyData(p1_best_projection)
        enclosed_result = p1_mesh.select_enclosed_points(model.mesh_surface)
        # enclosed_result = model.mesh_surface.select_interior_points(p1_mesh)
        print(enclosed_result)
        print("p1_best_projection", p1_best_projection)

        inside_mask = enclosed_result["SelectedPoints"] == 1

        mesh = model.mesh
        cut_elem_ids = np.array(
            [
                elem_id - 1
                for elem_id, (cut_type, _, _) in cut_info.items()
                if cut_type != CutType.NONE
            ]
        )
        # cut_mesh = mesh.extract_cells(cut_elem_ids)
        # cell_indices = cut_mesh.find_containing_cell(p1_best_projection)

        cut_mesh = mesh.extract_cells(cut_elem_ids)
        cell_indices = cut_mesh.find_containing_cell(p1_best_projection)

        # ... your orphan logic here ...
        orphans_mask = cell_indices == -1
        # print("orphans_mask", orphans_mask)
        if np.any(orphans_mask):
            cell_indices[orphans_mask] = cut_mesh.find_closest_cell(
                p1_best_projection[orphans_mask]
            )

        # THE FIX: Use PyVista's built-in tracking array instead of your own index mapping
        original_cell_ids = cut_mesh.cell_data["vtkOriginalCellIds"]
        p1_elem_indices = original_cell_ids[cell_indices]

        # p1_elem_indices = cut_elem_ids[cell_indices]

        sort_idx = np.argsort(p1_elem_indices)
        sorted_cells = p1_elem_indices[sort_idx]
        unique_cells, split_indices = np.unique(sorted_cells, return_index=True)

        grouped_point_indices = np.split(sort_idx, split_indices[1:])

        jump = np.full_like(p1_flat, np.nan)
        r_1_star = np.full(p1_flat.shape[0], np.nan)

        for elem_id, point_idx_batch in zip(unique_cells, grouped_point_indices):
            element = model.elements[elem_id]
            cut_type = cut_info[elem_id + 1][0]

            elem = _build_elem_3d(model, level_set, cut_type, element)
            _, nat_coords_batch = elem.nearest_point_on_crack(
                p1_best_projection[point_idx_batch],
                p1_tip[point_idx_batch],
                p1_b[point_idx_batch],
            )
            Ue = fill_element_displacement(
                np.asarray(element[4]), model.list_dof, model.Ug
            ).reshape((-1, 3))
            jump_shape_fn_batch, r_1_batch = elem.jump_shape_functions(
                nat_coords_batch, tip_points[point_idx_batch // len(self.r)]
            )

            jump[point_idx_batch, :] = jump_shape_fn_batch @ Ue
            r_1_star[point_idx_batch] = r_1_batch

        valid_mask = ~np.isnan(r_1_star) & inside_mask
        valid_mask = valid_mask.reshape((p1.shape[0], p1.shape[1]))
        r_1_star = r_1_star.reshape((p1.shape[0], p1.shape[1]))
        jump = jump.reshape(p1.shape)

        sort_idx = np.argsort(r_1_star, axis=1)
        valid_mask = np.take_along_axis(valid_mask, sort_idx, axis=1)
        r_sorted = np.take_along_axis(r_1_star, sort_idx, axis=1)
        jump_sorted = np.take_along_axis(jump, sort_idx[..., np.newaxis], axis=1)

        T_matrix = np.stack([t, n, b], axis=1)

        jump_local = jump_sorted @ T_matrix.transpose(0, 2, 1)
        # jump_local = jump_sorted @ T_matrix

        coef_I_II = (self.shear_mod / (self.kosolov + 1.0)) * np.sqrt(
            2.0 * np.pi / r_sorted
        )

        K_I_star = coef_I_II * jump_local[..., 1]
        K_II_star = coef_I_II * jump_local[..., 0]

        coef_III = (self.shear_mod / 4.0) * np.sqrt(2.0 * np.pi / r_sorted)
        K_III_star = coef_III * jump_local[..., 2]

        r_a, r_b = r_sorted[:, :-1], r_sorted[:, 1:]
        print("r_a", r_a)
        K_I_a, K_I_b = K_I_star[:, :-1], K_I_star[:, 1:]
        K_II_a, K_II_b = K_II_star[:, :-1], K_II_star[:, 1:]
        K_III_a, K_III_b = K_III_star[:, :-1], K_III_star[:, 1:]

        dr = r_b - r_a
        dr_safe = np.where(dr < 1e-10, np.nan, dr)
        print("dr_safe", dr_safe)

        extrap_multiplier = r_b / dr_safe
        r_ratio = r_a / r_b

        K_I_ext = extrap_multiplier * (K_I_a - r_ratio * K_I_b)
        K_II_ext = extrap_multiplier * (K_II_a - r_ratio * K_II_b)
        K_III_ext = extrap_multiplier * (K_III_a - r_ratio * K_III_b)

        with np.errstate(invalid="ignore"):
            K_I_final = np.nanmean(K_I_ext, axis=1)
            K_II_final = np.nanmean(K_II_ext, axis=1)
            K_III_final = np.nanmean(K_III_ext, axis=1)

        failed_extrapolations = np.isnan(K_I_final)
        if np.any(failed_extrapolations):
            print(
                f"Warning: {np.sum(failed_extrapolations)} DCM evaluation points failed "
                "(likely < 2 valid radial extraction points after cleaning)."
            )

        return K_I_final, K_II_final, K_III_final
