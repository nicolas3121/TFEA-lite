import numpy as np
from ..core.dofs import BASE_DOFS, HEAVISIDE_DOFS, BRANCH_DOFS
from typing import Final


def branch_functions(sqrt_r, theta):
    return sqrt_r * np.array(
        [
            np.sin(theta / 2),
            np.cos(theta / 2),
            np.sin(theta / 2) * np.sin(theta),
            np.cos(theta / 2) * np.sin(theta),
        ]
    )


def cal_B_2d(dN_dxy):
    DOFS: Final = 2
    B = np.zeros((3, DOFS * dN_dxy.shape[1]))
    B[0, ::DOFS] = dN_dxy[0, :]
    B[1, 1::DOFS] = dN_dxy[1, :]
    B[2, ::DOFS] = dN_dxy[1, :]
    B[2, 1::DOFS] = dN_dxy[0, :]
    return B


def cal_B_2d_vec(dN_dxy):
    DOFS: Final = 2
    B = np.zeros((dN_dxy.shape[0], 3, DOFS * dN_dxy.shape[2]))
    B[:, 0, ::DOFS] = dN_dxy[:, 0, :]
    B[:, 1, 1::DOFS] = dN_dxy[:, 1, :]
    B[:, 2, ::DOFS] = dN_dxy[:, 1, :]
    B[:, 2, 1::DOFS] = dN_dxy[:, 0, :]
    return B


def cut_embedding_tri_iter(Nc, range=range(4)):
    for i in range:
        if i != 3:
            Ni = np.eye(3)
            Ni[:, (i + 1) % 3] = Nc[:, i]
            Ni[:, (i + 2) % 3] = Nc[:, (i + 2) % 3]
        else:
            Ni = Nc.copy()
        detJi = np.linalg.det(Ni)
        if not np.isclose(detJi, 0.0):
            yield Ni, detJi


def partial_cut_embedding_tri_iter(Nc, tip, range):
    Ni_template = np.zeros((3, 3))
    Ni_template[:, 0] = tip
    for i in range:
        Ni = Ni_template.copy()
        Ni[int((i % 5 + 1) / 2), 1 + i % 2] = 1
        Ni[:, 2 - i % 2] = Nc[:, int(i / 2)]
        detJi = np.linalg.det(Ni)
        if not np.isclose(detJi, 0):
            yield Ni, detJi


def jump_shape_functions(elem, shape_fn, pu_fn, xi, eta, tip_coords):
    n_points = xi.shape[0]
    N, _ = shape_fn(xi, eta)
    Q, _ = pu_fn(xi, eta)
    N_jump = np.empty(
        (
            n_points,
            elem.N_FN
            + int(elem.h_enrich) * elem.H_FN
            + int(elem.t_enrich) * elem.TIP_FN,
        )
    )
    N_jump[:, : elem.N_FN] = 0.0
    r = np.linalg.norm(N @ elem.node_coords - tip_coords, axis=1)
    if elem.h_enrich:
        begin_h, end_h = elem.N_FN, elem.N_FN + elem.H_FN
        N_jump[:, begin_h:end_h] = N[:, : elem.N_FN]
    if elem.t_enrich:
        ramp = np.sum(N[:, np.where(elem.in_range)[0]], axis=1)
        sqrt_r = np.sqrt(r)
        begin_tip = elem.N_FN + int(elem.h_enrich) * elem.H_FN
        end_tip = begin_tip + elem.TIP_FN
        N_jump[:, begin_tip:end_tip] = 0.0
        N_jump[:, begin_tip::4] = 2 * sqrt_r[:, None] * ramp * Q
    return N_jump, r


def contains_points(node_coords, points):
    points = np.atleast_2d(points)
    x_e_1 = node_coords
    x_e_2 = np.empty_like(x_e_1)
    x_e_2[:-1, :] = x_e_1[1:, :]
    x_e_2[-1, :] = x_e_1[0, :]

    edges = x_e_2 - x_e_1
    vec = points[:, None, :] - x_e_1[None, :, :]
    cross = edges[None, :, 0] * vec[:, :, 1] - edges[None, :, 1] * vec[:, :, 0]

    is_inside = np.all(cross >= -1e-12, axis=1)
    return is_inside


def fill_element_displacement(elem_nodes, list_dof, Ug):
    DOF_TYPES = np.array(
        [
            BASE_DOFS,
            HEAVISIDE_DOFS,
            BRANCH_DOFS,
        ]
    )
    elem_dofs = list_dof.get_elem_dofs(elem_nodes)
    local_dofs_per_node = np.bitwise_or.reduce(elem_dofs)
    n_nodes = len(elem_nodes)
    Ue = np.zeros((n_nodes, np.bitwise_count(local_dofs_per_node))).flatten()
    DOFs = np.concatenate(
        (
            list_dof.get_elem_dof_numbers_flat(elem_nodes, BASE_DOFS).flatten(),
            list_dof.get_elem_dof_numbers_flat(elem_nodes, HEAVISIDE_DOFS).flatten(),
            list_dof.get_elem_dof_numbers_flat(elem_nodes, BRANCH_DOFS).flatten(),
        )
    )
    Ueg = Ug[DOFs]
    if len(DOFs) < len(Ue):
        is_present = np.bitwise_count(
            np.bitwise_and(DOF_TYPES[:, None], elem_dofs)
        ).flatten()
        absent = np.bitwise_count(
            np.bitwise_and(
                local_dofs_per_node,
                np.bitwise_and(DOF_TYPES[:, None], np.bitwise_not(elem_dofs)),
            )
        ).flatten()
        counts = np.empty(len(is_present) * 2, dtype=int)
        counts[0::2] = absent
        counts[1::2] = is_present
        values = np.empty(len(is_present) * 2, dtype=bool)
        values[0::2] = False
        values[1::2] = True
        mask = np.repeat(values, counts)
        Ue[mask] = Ueg
    else:
        Ue[:] = Ueg
    return Ue


def enriched_shape_functions(elem, shape_fn, pu_fn, xi, eta):
    n_points = xi.shape[0]
    N = np.empty(
        (
            n_points,
            elem.N_FN
            + int(elem.h_enrich) * elem.H_FN
            + int(elem.t_enrich) * elem.TIP_FN,
        )
    )
    dN_dxi = np.empty(
        (
            n_points,
            elem.DOFS,
            elem.N_FN
            + int(elem.h_enrich) * elem.H_FN
            + int(elem.t_enrich) * elem.TIP_FN,
        )
    )
    (N[:, : elem.N_FN], dN_dxi[:, :, : elem.N_FN]) = shape_fn(xi, eta)
    Q, dQ_dxi = pu_fn(xi, eta)
    phi_n = np.sum(elem.phi_n * N[:, : elem.N_FN], axis=1)
    phi_t = np.sum(elem.phi_t * N[:, : elem.N_FN], axis=1)
    if elem.h_enrich:
        h_shifted = (np.sign(phi_n)[:, None] - np.sign(elem.phi_n)) / 2
        begin_h, end_h = elem.N_FN, elem.N_FN + elem.H_FN
        N[:, begin_h:end_h] = h_shifted * N[:, : elem.N_FN]
        dN_dxi[:, :, begin_h:end_h] = h_shifted[:, None, :] * dN_dxi[:, :, : elem.N_FN]
    if elem.t_enrich:
        r = np.sqrt(phi_n**2 + phi_t**2)
        r = np.maximum(r, 1e-14)  # avoid divide by zero
        sqrt_r = np.sqrt(r)
        sqrt_r_i = (elem.phi_n**2 + elem.phi_t**2) ** (1 / 4)
        theta = np.atan2(phi_n, phi_t)
        theta_i = np.atan2(elem.phi_n, elem.phi_t)
        dphi_n_dxi = np.sum(elem.phi_n * dN_dxi[:, :, : elem.N_FN], axis=2)
        dphi_t_dxi = np.sum(elem.phi_t * dN_dxi[:, :, : elem.N_FN], axis=2)
        # sin(theta) = phi_n / r, cos(theta) = phi_t / r
        dr_dxi = (
            1 / r[:, None] * (phi_n[:, None] * dphi_n_dxi + phi_t[:, None] * dphi_t_dxi)
        )  # = np.sin(theta) * dphi_n_dxi - np.cos(theta) * dphi_t_dxi
        dtheta_dxi = (dphi_n_dxi * phi_t[:, None] - phi_n[:, None] * dphi_t_dxi) / (
            phi_t**2 + phi_n**2
        )[:, None]  # = (dphi_n_dxi * np.cos(theta) + np.sin(theta) dphi_t_dxi) / r
        bf = branch_functions(sqrt_r, theta).T
        bf_i = branch_functions(sqrt_r_i, theta_i).T

        dbf_dxi = 1 / (2 * sqrt_r[:, None, None]) * dr_dxi.reshape((-1, 2, 1)) * (
            bf / sqrt_r[:, None]
        )[:, None, :] + sqrt_r[:, None, None] * np.array(
            [
                np.cos(theta[:, None] / 2) * dtheta_dxi / 2,
                -np.sin(theta[:, None] / 2) * dtheta_dxi / 2,
                np.cos(theta[:, None] / 2) * dtheta_dxi / 2 * np.sin(theta[:, None])
                + np.sin(theta[:, None] / 2) * np.cos(theta[:, None]) * dtheta_dxi,
                -np.sin(theta[:, None] / 2) * dtheta_dxi / 2 * np.sin(theta[:, None])
                + np.cos(theta[:, None] / 2) * np.cos(theta[:, None]) * dtheta_dxi,
            ]
        ).transpose(1, 2, 0)
        shifter = bf_i[None, :, :]
        interpolant = np.sum(shifter * N[:, : elem.N_FN, None], axis=1)
        begin_tip = elem.N_FN + int(elem.h_enrich) * elem.H_FN
        end_tip = begin_tip + elem.TIP_FN
        bf_shifted = bf - interpolant
        ramp = np.sum(N[:, np.where(elem.in_range)[0]], axis=1)
        dramp_dxi = np.sum(dN_dxi[:, :, np.where(elem.in_range)[0]], axis=2)

        N[:, begin_tip:end_tip] = (
            bf_shifted[:, None, :] * ramp[:, None, None] * Q[:, :, None]
        ).reshape(-1, elem.TIP_FN)

        term1 = (
            (
                dbf_dxi
                - np.sum(
                    shifter[:, None, :, :] * dN_dxi[:, :, : elem.N_FN, None], axis=2
                )
            )[:, None, :, :]  # (n, 1, 2, 4)
            * ramp[:, None, None, None]  # (n)
            * Q[:, :, None, None]  # (n, NODES, 1, 1)
        )  # (n, NODES, 2, 4)

        term2 = (
            bf_shifted[:, None, None, :]
            * dramp_dxi[:, :, None, None]
            * Q[:, None, :, None]
        )  # (n, 2, NODES, 4)

        term3 = (
            bf_shifted[:, None, None, :]  # (n, 1, 1, 4)
            * ramp[:, None, None, None]
            * dQ_dxi[:, :, :, None]  # (n, 2, NODES, 1)
        )  # (n, 2, NODES, 4)
        dN_dxi[:, 0, begin_tip:end_tip] = (
            term1[:, :, 0, :] + term2[:, 0, :, :] + term3[:, 0, :, :]
        ).reshape(-1, elem.TIP_FN)
        dN_dxi[:, 1, begin_tip:end_tip] = (
            term1[:, :, 1, :] + term2[:, 1, :, :] + term3[:, 1, :, :]
        ).reshape(-1, elem.TIP_FN)
    return N, dN_dxi
