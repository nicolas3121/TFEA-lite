import numpy as np

from ..core.dofs import BRANCH_DOFS, HEAVISIDE_DOFS
from ..elements.utils import fill_element_displacement
from ..elements.XQuad4n import XQuad4n
from ..elements.XTri3n import XTri3n
from .level_set import CutType, project_on_line
from scipy.spatial import KDTree


ELEM_FN_MAP = {
    "Tri3n": XTri3n,
    "Quad4n": XQuad4n,
}


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
        assert tip_elem_idx != -1, "Couldn't find element containing tip"
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
            # print(Ue.shape)
            # print(point_idx_batch)
            # print("nat_coords_batch", nat_coords_batch)
            jump_shape_fn_batch, r_1_batch = elem.jump_shape_functions(
                nat_coords_batch, real_tip_coords
            )
            print(jump_shape_fn_batch.shape)

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


# class DisplacementCorrelationMethodSIF3D:
#     def __init__(self, kosolov, shear_mod, r, dr):
#         self.kosolov = kosolov
#         self.shear_mod = shear_mod
#         self.r = r
#         self.dr = dr
#
#     def cal_sif(self, level_set, ls_id, model, cut_info: dict, tip_index, v_tip):
#         uv_tip = np.stack([np.ones_like(v_tip), v_tip], axis=1)
#         tip_surface = level_set.ndbsplines[tip_index]
#         S = tip_surface(uv_tip)
#         Su = tip_surface(uv_tip, nu=(1, 0))
#         Sv = tip_surface(uv_tip, nu=(0, 1))
#         n = np.cross(Su, Sv, axis=1)
#         t = np.cross(Sv, n, axis=1)
#         t = t / np.linalg.norm(t, axis=1)[:, None]
#
#         p1 = S[:, None, :] - self.r[None, :, None] * t[:, None, :]
#
#         p1_tip_index = np.full((p1.shape[0], p1.shape[1]), tip_index, dtype=int)
#         p1_best_uv = np.full((p1.shape[0], p1.shape[1], 2), np.nan, dtype=np.float64)
#
#         project_on_surface(
#             tip_surface, p1_best_uv.reshape((-1, 2)), p1.reshape((-1, 3))
#         )
#
#         S = np.full((p1.shape[0] * p1.shape[1], 3), np.nan, dtype=np.float64)
#         for i, ndbspline in enumerate(level_set.ndbsplines):
#             mask = p1_tip_index.ravel() == i
#             S[mask] = ndbspline(p1_best_uv.reshape(-1, 2)[mask])
#
#         mesh = model.mesh
#
#         front_cell_indices = mesh.find_containing_cell(S).reshape((-1, p1.shape[1]))
#         bad_point_mask = np.zeros_like(p1_tip_index, dtype=bool)
#         bad_point_mask[front_cell_indices == -1] = True
#
#         flat_cells = front_cell_indices.ravel()
#
#         valid_mask = flat_cells != -1
#         valid_cell_ids = flat_cells[valid_mask]
#
#         valid_point_indices = np.where(valid_mask)[0]
#
#         sort_idx = np.argsort(valid_cell_ids)
#         sorted_cells = valid_cell_ids[sort_idx]
#         sorted_point_indices = valid_point_indices[sort_idx]
#
#         unique_cells, split_indices = np.unique(sorted_cells, return_index=True)
#
#         # np.split creates a list of arrays. Each sub-array contains the
#         # original flat point indices that belong to that specific unique_cell.
#         # We skip the first split_index (0) because it creates an empty array.
#         grouped_point_indices = np.split(sorted_point_indices, split_indices[1:])
#
#         for elem_id, point_idx_batch in zip(unique_cells, grouped_point_indices):
#             # 1. Get the physical coordinates of the points in this element
#             # point_idx_batch might contain 1 index, or it might contain 20!
#             points_in_elem = S[point_idx_batch]
#
#             element = model.elements[elem_id]
#             cut_type, _, _ = level_set.is_cut(element)
#             if cut_type == CutType.NONE:
#                 valid_mask[point_idx_batch] = False
#                 continue
#
#             _, _, mat_id, real_id, elem_nodes = element
#             elem_nodes = np.asarray(elem_nodes)
#
#             elem_vertices = model.nodes[elem_nodes - 1, 1:4]
#             elem_dofs = model.list_dof.get_elem_dofs(elem_nodes)
#             local_dofs_per_node = np.bitwise_or.reduce(elem_dofs)
#             h_enrich = bool(local_dofs_per_node & HEAVISIDE_DOFS)
#             t_enrich = bool(local_dofs_per_node & BRANCH_DOFS)
#             phi_n, phi_t = model.level_sets[ls_id].get(elem_nodes, tip_index)
#
#             elem = XTetr4n(
#                 node_coords=elem_vertices,
#                 material=model.materials[mat_id - 1][1],
#                 real=None,
#                 phi_n=phi_n,
#                 phi_t=phi_t,
#                 h_enrich=h_enrich,
#                 t_enrich=t_enrich,
#                 partial_cut=(cut_type == CutType.PARTIAL),
#                 in_range=model.in_range[elem_nodes - 1],
#             )
#
#         # 2. Extract Element Data ONCE
#         # elem_nodes = mesh.extract_cells(elem_id).points
#         # elem_displacements = global_u[node_ids]
#
#         # 3. Vectorized Inverse Mapping
#         # natural_coords = inverse_map(elem_nodes, points_in_elem) # Shape: (N_batch, 3)
#
#         # 4. Vectorized Shape Function Evaluation
#         # N = evaluate_shape_functions(natural_coords)
#
#         # 5. Calculate Jump
#         # u_jump = N @ elem_displacements
#
#         # 6. Store the results back in a global array using point_idx_batch
#         # global_jumps[point_idx_batch] = u_jump
#
#
# # front_pts = tip_surface(uv_tip)
# # mesh = model.mesh
# # partial_cut_elem_list = level_set.partial_cut_elem_list[tip_index]
# # sub_mesh = mesh.extract_cells(np.asarray(partial_cut_elem_list, dtype=int) - 1)
# # front_cell_indices = sub_mesh.find_containing_cell(front_pts)
# # orphans_mask = front_cell_indices == -1
# # orphan_pts = front_pts[orphans_mask]
# # closest_cell_indices = sub_mesh.find_closest_cell(orphan_pts)
# # final_indices = np.where(orphans_mask, closest_cell_indices, front_cell_indices)
# # original_ids_map = sub_mesh.cell_data["eid"]
# # element_ids = original_ids_map[final_indices]
#
#
# # voor alle front cells de orientatie en locatie van crack front (dichste bij control point zoeken)
# # dan terug op basis van radius punten op crack surface zoeken om te evalueren, projecteren op oppervlak om punten in de buurt te vinden
# # dan elementen zoeken die die punten bevatten
# # binnen element mischien punt proberen zoeken dat in de praktijk loodrecht staat op crack front
# # dan displacement jump berekenen
# # op basis van orientatie crack front in elementen nieuwe control points berekenen
# # of moet allesinds rekening houden met verschil in orientatie tussen spline en echte versie in elementen
