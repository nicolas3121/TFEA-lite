import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

from ..core import quadratures as qd
from ..core.dofs import BRANCH_DOFS, HEAVISIDE_DOFS
from ..core.quadratures import DuffyDistance
from ..elements.utils import (
    cal_B_2d_vec,
    cut_embedding_tri_iter,
    fill_element_displacement,
    partial_cut_embedding_tri_iter,
)
from ..elements.XQuad4n import XQuad4n
from ..elements.XTetr4n import XTetr4n
from ..elements.XTri3n import XTri3n
from .level_set import CutType, project_on_line, project_on_surface

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


def compute_auxiliary_fields(r, theta, kosolov, shear_mod):
    """
    Computes Mode I & II auxiliary stresses, displacement gradients, and strains
    in crack-tip local coordinate system (Williams asymptotic fields).
    """
    sqr = np.sqrt(r)

    fac_stress = 1.0 / np.sqrt(2.0 * np.pi)
    fac_disp = 1.0 / (2.0 * shear_mod * np.sqrt(2.0 * np.pi))

    ct = np.cos(theta)
    st = np.sin(theta)
    ct2 = np.cos(theta / 2.0)
    st2 = np.sin(theta / 2.0)
    c3t2 = np.cos(1.5 * theta)
    s3t2 = np.sin(1.5 * theta)

    drdx, drdy = ct, st
    dtdx, dtdy = -st / r, ct / r

    aux_stress = np.zeros((2, 2, 2))  # [mode, i, j]
    aux_grad_u = np.zeros((2, 2, 2))  # [mode, i, j] (du_i / dx_j)
    aux_strain = np.zeros((2, 2, 2))

    # --- Mode I (aux_mode = 0) ---
    aux_stress[0, 0, 0] = (fac_stress / sqr) * ct2 * (1.0 - st2 * s3t2)
    aux_stress[0, 1, 1] = (fac_stress / sqr) * ct2 * (1.0 + st2 * s3t2)
    aux_stress[0, 0, 1] = (fac_stress / sqr) * st2 * ct2 * c3t2
    aux_stress[0, 1, 0] = aux_stress[0, 0, 1]

    du1_dr = fac_disp * 0.5 / sqr * ct2 * (kosolov - ct)
    du1_dt = fac_disp * sqr * (-0.5 * st2 * (kosolov - ct) + ct2 * st)
    du2_dr = fac_disp * 0.5 / sqr * st2 * (kosolov - ct)
    du2_dt = fac_disp * sqr * (0.5 * ct2 * (kosolov - ct) + st2 * st)

    aux_grad_u[0, 0, 0] = du1_dr * drdx + du1_dt * dtdx
    aux_grad_u[0, 0, 1] = du1_dr * drdy + du1_dt * dtdy
    aux_grad_u[0, 1, 0] = du2_dr * drdx + du2_dt * dtdx
    aux_grad_u[0, 1, 1] = du2_dr * drdy + du2_dt * dtdy

    # --- Mode II (aux_mode = 1) ---
    aux_stress[1, 0, 0] = -(fac_stress / sqr) * st2 * (2.0 + ct2 * c3t2)
    aux_stress[1, 1, 1] = (fac_stress / sqr) * st2 * ct2 * c3t2
    aux_stress[1, 0, 1] = (fac_stress / sqr) * ct2 * (1.0 - st2 * s3t2)
    aux_stress[1, 1, 0] = aux_stress[1, 0, 1]

    du1_dr = fac_disp * 0.5 / sqr * st2 * (kosolov + 2.0 + ct)
    du1_dt = fac_disp * sqr * (0.5 * ct2 * (kosolov + 2.0 + ct) - st2 * st)
    du2_dr = -fac_disp * 0.5 / sqr * ct2 * (kosolov - 2.0 + ct)
    du2_dt = -fac_disp * sqr * (-0.5 * st2 * (kosolov - 2.0 + ct) - ct2 * st)

    aux_grad_u[1, 0, 0] = du1_dr * drdx + du1_dt * dtdx
    aux_grad_u[1, 0, 1] = du1_dr * drdy + du1_dt * dtdy
    aux_grad_u[1, 1, 0] = du2_dr * drdx + du2_dt * dtdx
    aux_grad_u[1, 1, 1] = du2_dr * drdy + du2_dt * dtdy

    for m in range(2):
        aux_strain[m] = 0.5 * (aux_grad_u[m] + aux_grad_u[m].T)

    return aux_stress, aux_grad_u, aux_strain


def calculate_element_I_mode(element, Ue, T_matrix, q_nodes, real_tip, kosolov, shear_mod):
    I_mode = np.zeros(2)
    N_FN = 4
    DET_TOL = -1e-14


    def accumulate_I_mode(nat_coords, w_eff, sign=None):
        nonlocal I_mode

        if sign is not None:
            N, dN_dxi = element.shape_functions(nat_coords, enforce_sign=sign)
        else:
            N, dN_dxi = element.shape_functions(nat_coords)
            
        # N shape: (num_nodes, num_gp) or similar; adjust based on your shape function convention
        # x_gp shape: (num_gp, 2)
        x_gp = N[:, :N_FN].T @ element.node_coords

        J = dN_dxi[:, :, :N_FN] @ element.node_coords  # (num_gp, 2, 2)
        dN_dxy = np.linalg.solve(J, dN_dxi)             # (num_gp, 2, num_nodes)
        B = cal_B_2d_vec(dN_dxy)                        # (num_gp, 3, num_dof)

        eps_h_voigt = np.einsum("gij,j->gi", B, Ue)     # (num_gp, 3) -> [eps_xx, eps_yy, 2*eps_xy]
        sig_h_voigt = np.einsum("ij,gj->gi", element.C, eps_h_voigt) # (num_gp, 3)

        # Reshape Voigt vectors [s_xx, s_yy, s_xy] to 2x2 tensors (g, 2, 2)
        sig_h = np.zeros((len(w_eff), 2, 2))
        sig_h[:, 0, 0] = sig_h_voigt[:, 0]
        sig_h[:, 1, 1] = sig_h_voigt[:, 1]
        sig_h[:, 0, 1] = sig_h[:, 1, 0] = sig_h_voigt[:, 2]

        eps_h = np.zeros((len(w_eff), 2, 2))
        eps_h[:, 0, 0] = eps_h_voigt[:, 0]
        eps_h[:, 1, 1] = eps_h_voigt[:, 1]
        eps_h[:, 0, 1] = eps_h[:, 1, 0] = 0.5 * eps_h_voigt[:, 2]

        # q-gradient in global coordinates: sum(q_i * dN_i/dx_j) -> shape (num_gp, 2)
        grad_q_g = np.einsum('i,gdi->gd', q_nodes, dN_dxy[:, :, :element.N_FN])

        # Transform to crack-tip local coordinates
        x_loc = (T_matrix @ (x_gp - real_tip).T).T  # (num_gp, 2)
        r = np.linalg.norm(x_loc, axis=1)           # (num_gp,)
        
        # Filter out points too close to the tip to avoid division by zero
        valid_mask = r >= 1.0e-12
        if not np.any(valid_mask):
            return

        # Restrict arrays to valid Gauss points
        x_loc = x_loc[valid_mask]
        r_v = r[valid_mask]
        theta = np.arctan2(x_loc[:, 1], x_loc[:, 0])
        
        grad_q_loc = (T_matrix @ grad_q_g[valid_mask].T).T                        # (num_valid_gp, 2)
        eps_h_loc = T_matrix @ eps_h[valid_mask] @ T_matrix.T                    # (num_valid_gp, 2, 2)
        sig_h_loc = T_matrix @ sig_h[valid_mask] @ T_matrix.T                    # (num_valid_gp, 2, 2)
        w_eff_v = w_eff[valid_mask]

        aux_stress, aux_grad_u, aux_strain = compute_auxiliary_fields(r_v, theta, kosolov, shear_mod)

        # Fully vectorized interaction integral terms using einsum
        # i1: sig_h_loc[g, i, j] * aux_grad_u[m, g, i, 0] * grad_q_loc[g, j]
        i1 = np.einsum('gij, mgi, gj -> mg', sig_h_loc, aux_grad_u[:, :, :, 0], grad_q_loc)

        # i2: aux_stress[m, g, i, j] * eps_h_loc[g, i, 0] * grad_q_loc[g, j]
        i2 = np.einsum('mgij, gi, gj -> mg', aux_stress, eps_h_loc[:, :, 0], grad_q_loc)

        # Mutual strain energy W^(1,2): contraction of sig_h_loc and aux_strain
        W_12 = np.einsum('gij, mgij -> mg', sig_h_loc, aux_strain)

        # Domain integrand across all valid Gauss points: (2, num_valid_gp)
        integrand = (i1 + i2 - W_12 * grad_q_loc[:, 0][None, :]) * w_eff_v[None, :]

        # Accumulate results for both modes
        I_mode += np.sum(integrand, axis=1)


    if not getattr(element, "h_enrich", False) and not getattr(
        element, "partial_cut", False
    ):
        rule, correction = qd.QUAD_RULES[20]
        nat_coords = rule[:, :2].T

        _, dN_dxi = element.shape_functions(nat_coords)
        J = dN_dxi[:, :, :N_FN] @ element.node_coords
        detJ = np.linalg.det(J)
        w_eff = rule[:, 2] * correction * detJ

        accumulate_I_mode(nat_coords, w_eff)

    elif getattr(element, "h_enrich", False):
        Nc1, Nc2 = element._cal_intersections()
        rule, correction = qd.TRI_RULES[13] if element.t_enrich else qd.TRI_RULES[13]

        def integrate_sub_tri_I_mode(Nc, nat_x_e):
            for Ni, detJi in cut_embedding_tri_iter(Nc):
                xi, eta, w = rule[:, 0], rule[:, 1], rule[:, 2]
                nat_sub_x_e = nat_x_e.T @ Ni

                N, _ = element._base_shape_functions(nat_sub_x_e)
                sub_phi_n = N @ element.phi_n

                nat_coords_sub, detJi_mod, sign = element._get_mapped_coords(
                    xi,
                    eta,
                    sub_phi_n,
                    True,
                    nat_sub_x_e,
                    detJi,
                    False,
                )
                if np.any(detJi_mod < DET_TOL):
                    print("warning encounter negative detJi")

                n = np.array([1 - xi - eta, xi, eta])
                nat_coords_sub = nat_sub_x_e @ n

                N, dN_dxi_sub = element.shape_functions(
                    nat_coords_sub, enforce_sign=sign
                )

                J = dN_dxi_sub[:, :, :N_FN] @ element.node_coords
                detJ = np.linalg.det(J)

                w_eff = w * correction * detJ * detJi_mod
                accumulate_I_mode(nat_coords_sub, w_eff, sign=sign)

        integrate_sub_tri_I_mode(Nc1, element.NAT_1)
        integrate_sub_tri_I_mode(Nc2, element.NAT_2)

    elif getattr(element, "partial_cut", False):
        Nc1, Nc2 = element._cal_intersections()
        rule, correction = qd.QUAD_RULES[20]
        rule = rule.copy()
        rule[:, 0:2] = (1 + rule[:, 0:2]) / 2
        rule[:, 2] /= 4

        xi_tip, eta_tip = element._cal_tip_nat_coords()
        tri1_coords = np.vstack([element.NAT_1.T, np.ones(3)])
        tip1 = np.linalg.solve(tri1_coords, [xi_tip, eta_tip, 1.0])
        tri2_coords = np.vstack([element.NAT_2.T, np.ones(3)])
        tip2 = np.linalg.solve(tri2_coords, [xi_tip, eta_tip, 1.0])

        def integrate_partial_cut_I_mode(tip, Nc, rng, nat_x_e):
            for Ni, detJi in partial_cut_embedding_tri_iter(Nc, tip, rng):
                nat_sub_x_e = nat_x_e.T @ Ni
                N, _ = element._base_shape_functions(nat_sub_x_e)
                x_e_i = N @ element.node_coords
                sub_phi_n = N @ element.phi_n
                sub_phi_t = N @ element.phi_t
                behind_tip = sub_phi_t < 1e-10

                duffy = DuffyDistance(x_e_i)
                u, v = rule[:, 0], rule[:, 1]
                xi_d_2, eta_d_2, w_d_2 = duffy.transform(u, v, beta=2)

                nat_coords_sub, detJi_mod, sign = element._get_mapped_coords(
                    xi_d_2,
                    eta_d_2,
                    sub_phi_n,
                    behind_tip,
                    nat_sub_x_e,
                    detJi,
                    False,
                )
                if np.any(detJi_mod < DET_TOL):
                    print("warning encounter negative detJi")
                N, dN_dxi_sub = element.shape_functions(
                    nat_coords_sub, enforce_sign=sign
                )
                J = dN_dxi_sub[:, :, 0 : element.N_FN] @ element.node_coords
                detJ = np.linalg.det(J)

                w_eff = rule[:, 2] * correction * w_d_2 * detJ * detJi_mod
                accumulate_I_mode(nat_coords_sub, w_eff, sign=sign)

        integrate_partial_cut_I_mode(tip1, Nc1, range(4), element.NAT_1)
        integrate_partial_cut_I_mode(tip2, Nc2, range(2, 6), element.NAT_2)

    return I_mode


class InteractionIntegralMethodSIF:
    def __init__(self, r1, dr, E, nu, plane_stress):
        self.E_eff = E if plane_stress else E / (1.0 - nu**2)
        self.shear_mod = E / (2.0 * (1.0 + nu))
        self.kosolov = (3.0 - nu) / (1.0 + nu) if plane_stress else (3.0 - 4.0 * nu)
        self.r1 = r1
        self.dr = dr

    def _find_j_elements_idx(self, mesh, real_tip, r):
        real_tip_3d = (real_tip[0], real_tip[1], 0.0)
        sphere = pv.Sphere(radius=r, center=real_tip_3d)

        centers_poly = mesh.cell_centers()

        selected = centers_poly.select_interior_points(sphere)
        ball_elem_idx = np.where(selected["selected_points"])[0]
        return ball_elem_idx

    def cal_sif(self, level_set, model, cut_info: dict, u_tip: float):
        R_TOL = 0.3
        assert level_set.bspline is not None
        bspline = level_set.bspline

        tip = bspline(u_tip)

        mesh = model.mesh
        original_ids = mesh.cell_data["eid"]
        cut_elem_ids = np.array(
            [
                elem_id - 1
                for elem_id, (_, cut_type, _) in cut_info.items()
                if cut_type != CutType.NONE
            ]
        )
        assert np.all(original_ids[cut_elem_ids] == cut_elem_ids + 1)
        cut_mesh = mesh.extract_cells(cut_elem_ids)
        original_cell_ids = cut_mesh.cell_data["eid"]

        tip_3d = np.append(tip, 0.0)

        tip_elem_idx = cut_mesh.find_containing_cell(tip_3d)
        if tip_elem_idx == -1:
            for (_, cut_type, _) in cut_info.values():
                if cut_type == CutType.PARTIAL:
                    break
            else:
                print("couldn't find tip")
                raise ValueError
        tip_elem_id = original_cell_ids[tip_elem_idx]
        tip_elem = model.elements[tip_elem_id - 1]

        elem = _build_elem_2d(model, level_set, CutType.PARTIAL, tip_elem)
        tip_nat_coords = elem._cal_tip_nat_coords()
        N_tip, dN_dxi_tip = elem._base_shape_functions(tip_nat_coords)
        J = dN_dxi_tip @ elem.node_coords
        dN_dxy_tip = np.linalg.solve(J, dN_dxi_tip)
        real_tip = N_tip[0] @ elem.node_coords
        real_tip_n = dN_dxy_tip[0] @ elem.phi_n
        real_tip_n = real_tip_n / np.linalg.norm(real_tip_n)
        real_tip_t = dN_dxy_tip[0] @ elem.phi_t
        real_tip_t = real_tip_t - np.dot(real_tip_n, real_tip_t) * real_tip_n
        real_tip_t = real_tip_t / np.linalg.norm(real_tip_t)

        T_matrix = np.array([real_tip_t, real_tip_n])

        r_max = self.r1 + np.maximum(self.dr)
        j_element_idx = self._find_j_elements_idx(mesh, real_tip, (1 + R_TOL) * r_max)
        j_element_ids = original_ids[j_element_idx]

        I_mode = np.zeros(2)
        for elem_id in j_element_ids:
            element = model.elements[elem_id - 1]
            cut_type = cut_info[elem_id][1]

            elem = _build_elem_2d(model, level_set, cut_type, element)
            Ue = fill_element_displacement(
                np.asarray(element[4]), model.list_dof, model.Ug
            ).reshape((-1, 2))

            q_nodes = np.linalg.norm(elem.node_coords - real_tip[None, :], axis=1) <= self.r1
            # TODO fixen

            I_mode += calculate_element_I_mode(element, Ue, T_matrix, q_nodes, real_tip, self.kosolov, self.shear_mod)

        K_calc = I_mode * self.E_eff / 2.0
        KI, KII = K_calc[0], K_calc[1]
        return KI, KII






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
        # n = np.array([-t[1], t[0]])

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
        # cut_elem_ids = np.array(
        #     [
        #         elem_id - 1
        #         for elem_id, (cut_type, _, _) in cut_info.items()
        #         if cut_type != CutType.NONE
        #     ]
        # )
        original_ids = mesh.cell_data["eid"]
        cut_elem_ids = np.array(
            [
                elem_id - 1
                for elem_id, (_, cut_type, _) in cut_info.items()
                if cut_type != CutType.NONE
            ]
        )
        assert np.all(original_ids[cut_elem_ids] == cut_elem_ids + 1)
        cut_mesh = mesh.extract_cells(cut_elem_ids)
        original_cell_ids = cut_mesh.cell_data["eid"]

        tip_3d = np.append(tip, 0.0)
        p1_3d = np.column_stack((p1, np.zeros(p1.shape[0])))

        tip_elem_idx = cut_mesh.find_containing_cell(tip_3d)
        # assert tip_elem_idx != -1, "Couldn't find element containing tip"
        if tip_elem_idx == -1:
            for (_, cut_type, _) in cut_info.values():
                if cut_type == CutType.PARTIAL:
                    break
            else:
                print("couldn't find tip")
                raise ValueError
        tip_elem_id = original_cell_ids[tip_elem_idx]
        tip_elem = model.elements[tip_elem_id - 1]

        elem = _build_elem_2d(model, level_set, CutType.PARTIAL, tip_elem)
        tip_nat_coords = elem._cal_tip_nat_coords()
        N_tip, dN_dxi_tip = elem._base_shape_functions(tip_nat_coords)
        J = dN_dxi_tip @ elem.node_coords
        dN_dxy_tip = np.linalg.solve(J, dN_dxi_tip)
        real_tip = N_tip[0] @ elem.node_coords
        real_tip_n = dN_dxy_tip[0] @ elem.phi_n
        real_tip_n = real_tip_n / np.linalg.norm(real_tip_n)
        real_tip_t = dN_dxy_tip[0] @ elem.phi_t
        real_tip_t = real_tip_t - np.dot(real_tip_n, real_tip_t) * real_tip_n
        real_tip_t = real_tip_t / np.linalg.norm(real_tip_t)

        cell_indices = cut_mesh.find_containing_cell(p1_3d)
        orphans_mask = cell_indices == -1
        if np.any(orphans_mask):
            cell_indices[orphans_mask] = cut_mesh.find_closest_cell(p1_3d[orphans_mask])

        p1_elem_indices = original_cell_ids[cell_indices]

        sort_idx = np.argsort(p1_elem_indices)
        sorted_cells = p1_elem_indices[sort_idx]
        unique_cells, split_indices = np.unique(sorted_cells, return_index=True)

        grouped_point_indices = np.split(sort_idx, split_indices[1:])

        jump = np.full_like(p1, np.nan)
        r_1_star = np.full(p1.shape[0], np.nan)

        for elem_id, point_idx_batch in zip(unique_cells, grouped_point_indices):
            element = model.elements[elem_id - 1]
            cut_type = cut_info[elem_id][1]
            print("cut_type", cut_type)

            elem = _build_elem_2d(model, level_set, cut_type, element)
            _, nat_coords_batch = elem.nearest_point_on_crack(p1[point_idx_batch])
            Ue = fill_element_displacement(
                np.asarray(element[4]), model.list_dof, model.Ug
            ).reshape((-1, 2))
            jump_shape_fn_batch, r_1_batch = elem.jump_shape_functions(
                nat_coords_batch, real_tip
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

        T_matrix = np.array([real_tip_t, real_tip_n])
        jump_local = jump_clean @ T_matrix.T

        coef = (self.shear_mod / (self.kosolov + 1.0)) * np.sqrt(2.0 * np.pi / r_clean)
        K_I_star = coef * jump_local[:, 1]
        K_II_star = coef * jump_local[:, 0]

        if len(r_clean) < 2:
            print("Warning: < 2 valid radial extraction points. Extrapolation failed.")
            return float("nan"), float("nan")

        K_I_final = np.polyfit(r_clean, K_I_star, 1)[1]
        K_II_final = np.polyfit(r_clean, K_II_star, 1)[1]

        return float(K_I_final), float(K_II_final), T_matrix


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
        # cut_elem_ids = np.array(
        #     [
        #         elem_id - 1
        #         for elem_id, (cut_type, _, _) in cut_info.items()
        #         if cut_type != CutType.NONE
        #     ]
        # )
        cut_elem_ids = np.array(
            [
                elem_id - 1
                for elem_id, (_, cut_type, _) in cut_info.items()
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
        in_element_mask = np.zeros(p1_flat.shape[0], dtype=bool)

        for elem_id, point_idx_batch in zip(unique_cells, grouped_point_indices):
            element = model.elements[elem_id]
            cut_type = cut_info[elem_id + 1][1]
            print("cut_type", cut_type)

            elem = _build_elem_3d(model, level_set, cut_type, element)
            print("tip", p1_tip[point_idx_batch])
            _, nat_coords_batch, is_in_element = elem.nearest_point_on_crack(
                p1_best_projection[point_idx_batch],
                p1_tip[point_idx_batch],
                p1_b[point_idx_batch],
            )
            Ue = fill_element_displacement(
                np.asarray(element[4]), model.list_dof, model.Ug
            ).reshape((-1, 3))
            jump_shape_fn_batch, r_1_batch = elem.jump_shape_functions(
                nat_coords_batch.T, p1_tip[point_idx_batch]
            )

            jump[point_idx_batch, :] = jump_shape_fn_batch @ Ue
            r_1_star[point_idx_batch] = r_1_batch
            in_element_mask[point_idx_batch] = is_in_element

        valid_mask = ~np.isnan(r_1_star) & inside_mask & in_element_mask
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

        valid_counts = np.sum(~np.isnan(r_sorted), axis=1)
        sufficient_pts = valid_counts >= 2

        with np.errstate(invalid="ignore", divide="ignore"):
            # 2. Calculate the mean of r and K* (ignoring NaNs)
            r_mean = np.nanmean(r_sorted, axis=1, keepdims=True)
            K_I_mean = np.nanmean(K_I_star, axis=1, keepdims=True)
            K_II_mean = np.nanmean(K_II_star, axis=1, keepdims=True)
            K_III_mean = np.nanmean(K_III_star, axis=1, keepdims=True)

            # 3. Calculate deviations from the mean
            dr = r_sorted - r_mean
            dK_I = K_I_star - K_I_mean
            dK_II = K_II_star - K_II_mean
            dK_III = K_III_star - K_III_mean

            # 4. Calculate the Sum of Squares (denominator)
            SS_xx = np.nansum(dr**2, axis=1)

            # 5. Calculate the slope (c_1) for each mode
            slope_I = np.nansum(dr * dK_I, axis=1) / SS_xx
            slope_II = np.nansum(dr * dK_II, axis=1) / SS_xx
            slope_III = np.nansum(dr * dK_III, axis=1) / SS_xx

            # 6. Calculate the intercept (c_0), which is the final extrapolated SIF
            # Formula: c_0 = mean(y) - c_1 * mean(x)
            K_I_final = np.squeeze(K_I_mean) - slope_I * np.squeeze(r_mean)
            K_II_final = np.squeeze(K_II_mean) - slope_II * np.squeeze(r_mean)
            K_III_final = np.squeeze(K_III_mean) - slope_III * np.squeeze(r_mean)

        # 7. Safety check: mask out any nodes that didn't have enough valid points
        K_I_final = np.where(sufficient_pts, K_I_final, np.nan)
        K_II_final = np.where(sufficient_pts, K_II_final, np.nan)
        K_III_final = np.where(sufficient_pts, K_III_final, np.nan)

        failed_extrapolations = ~sufficient_pts
        if np.any(failed_extrapolations):
            print(
                f"Warning: {np.sum(failed_extrapolations)} DCM evaluation points failed "
                "(< 2 valid radial extraction points)."
            )

        return K_I_final, K_II_final, K_III_final
