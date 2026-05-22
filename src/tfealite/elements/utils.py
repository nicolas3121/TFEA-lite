import numpy as np
from ..core.dofs import BASE_DOFS, HEAVISIDE_DOFS, BRANCH_DOFS
from typing import Final

ELEM_EDGES = {
    "Tri3n": ([1, 2, 0], [0, 1, 2]),
    "Quad4n": ([1, 2, 3, 0], [0, 1, 2, 3]),
    "Tetr4n": ([1, 2, 3, 2, 3, 3], [0, 1, 2, 0, 0, 1]),
    "Hex8n": (
        [1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3],
    ),
}


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


def cal_N_2d_vec(N):
    DOFS: Final = 2
    N_2d = np.zeros((N.shape[0], 2, DOFS * N.shape[1]))
    N_2d[:, 0, ::DOFS] = N[:, :]
    N_2d[:, 0, 1::2] = N[:, :]
    return N_2d


def cal_B_3d_vec(dN_dxy):
    DOFS: Final = 3
    B = np.zeros((dN_dxy.shape[0], 6, DOFS * dN_dxy.shape[2]))
    B[:, 0, ::DOFS] = dN_dxy[:, 0, :]
    B[:, 1, 1::DOFS] = dN_dxy[:, 1, :]
    B[:, 2, 2::DOFS] = dN_dxy[:, 2, :]
    B[:, 3, ::DOFS] = dN_dxy[:, 1, :]
    B[:, 3, 1::DOFS] = dN_dxy[:, 0, :]
    B[:, 4, ::DOFS] = dN_dxy[:, 2, :]
    B[:, 4, 2::DOFS] = dN_dxy[:, 0, :]
    B[:, 5, 1::DOFS] = dN_dxy[:, 2, :]
    B[:, 5, 2::DOFS] = dN_dxy[:, 1, :]
    return B


def cut_embedding_tetr_iter(Nc, kappa, range_iter=range(12)):
    eye = np.eye(4)
    # Nc colums:
    #  0  1  2  3  4  5
    # [0, 1, 2, 0, 0, 1]
    # [1, 2, 3, 2, 3, 3]
    # -1 --> use kappa
    TET_SPLIT_MAP = [
        # Group 1: 4 Corner Tets
        (0, 0, 3, 4),
        (1, 1, 5, 0),
        (2, 2, 3, 1),
        (3, 4, 5, 2),
        # Group 2: 4 Inner Tets (Kappa connecting to the corner faces)
        (-1, 0, 4, 3),
        (-1, 1, 0, 5),
        (-1, 2, 1, 3),
        (-1, 4, 2, 5),
        # Group 3: 4 Face Tets (Kappa connecting to the outer faces)
        (-1, 0, 3, 1),
        (-1, 1, 5, 2),
        (-1, 2, 3, 4),
        (-1, 4, 5, 0),
    ]

    if kappa is None:
        yield eye, 1.0
    else:
        for i in range_iter:
            node_idx, e1, e2, e3 = TET_SPLIT_MAP[i]
            Ni = np.empty((4, 4))

            col = i % 4

            if node_idx != -1:
                Ni[:, col] = eye[:, node_idx]
            else:
                Ni[:, col] = kappa

            Ni[:, (col + 1) % 4] = Nc[:, e1]
            Ni[:, (col + 2) % 4] = Nc[:, e2]
            Ni[:, (col + 3) % 4] = Nc[:, e3]

            detJi = np.linalg.det(Ni)

            if not np.isclose(detJi, 0.0, atol=1e-12):
                if detJi < 0:
                    print(i, detJi, "|", node_idx, e1, e2, e3)
                yield Ni, detJi


def partial_cut_embedding_tetr_iter(Nc, tip, tip_on_interface, range_iter=range(17)):
    eye = np.eye(4)

    # Mapping structure: (Source, Index)
    # Source 0: Node (from 'eye' array)
    # Source 1: Tip (from 'tip' array)
    # Source 2: Edge (from 'Nc' array)
    #
    # Nc colums:
    #  0  1  2  3  4  5
    # [0, 1, 2, 0, 0, 1]
    # [1, 2, 3, 2, 3, 3]
    PARTIAL_TET_MAP = [
        # Corner 0
        ((1, 2), (1, 3), (1, 1), (0, 0)),
        ((1, 2), (1, 1), (2, 4), (0, 0)),
        ((1, 1), (1, 3), (2, 3), (0, 0)),
        ((1, 2), (1, 3), (2, 0), (0, 0)),
        # Corner 1
        ((1, 3), (1, 0), (1, 2), (0, 1)),
        ((1, 3), (1, 2), (2, 0), (0, 1)),
        ((1, 2), (1, 0), (2, 5), (0, 1)),
        ((1, 3), (1, 0), (2, 1), (0, 1)),
        # Corner 2
        ((1, 0), (1, 1), (1, 3), (0, 2)),
        ((1, 0), (1, 3), (2, 1), (0, 2)),
        ((1, 3), (1, 1), (2, 3), (0, 2)),
        ((1, 0), (1, 1), (2, 2), (0, 2)),
        # Corner 3
        ((1, 1), (1, 2), (1, 0), (0, 3)),
        ((1, 1), (1, 0), (2, 2), (0, 3)),
        ((1, 0), (1, 2), (2, 5), (0, 3)),
        ((1, 1), (1, 2), (2, 4), (0, 3)),
        # Central
        ((1, 0), (1, 1), (1, 2), (1, 3)),
    ]

    for k in range_iter:
        mapping = PARTIAL_TET_MAP[k]

        is_singular = []
        for src, idx in mapping:
            if src == 1 and tip_on_interface[idx]:
                is_singular.append(True)
            else:
                is_singular.append(False)

        singular_cols = [i for i, s in enumerate(is_singular) if s]
        regular_cols = [i for i, s in enumerate(is_singular) if not s]
        new_order = singular_cols + regular_cols

        Nik = np.empty((4, 4))
        for target_col, original_map_idx in enumerate(new_order):
            src, idx = mapping[original_map_idx]
            if src == 0:
                Nik[:, target_col] = eye[:, idx]
            elif src == 1:
                Nik[:, target_col] = tip[:, idx]
            else:
                Nik[:, target_col] = Nc[:, idx]

        detJik = np.linalg.det(Nik)

        if np.isclose(detJik, 0.0, atol=1e-12):
            continue

        # print(len(singular_cols))
        if detJik < 0:
            if len(regular_cols) >= 2:
                Nik[:, [2, 3]] = Nik[:, [3, 2]]
            else:
                Nik[:, [0, 1]] = Nik[:, [1, 0]]
            detJik = -detJik

        yield Nik, detJik, len(singular_cols)


def cut_embedding_tri_iter(Nc, range=range(4)):
    ZERO_TOL = 1e-10
    sum = 0
    for i in range:
        if i != 3:
            Ni = np.eye(3)
            Ni[:, (i + 1) % 3] = Nc[:, i]
            Ni[:, (i + 2) % 3] = Nc[:, (i + 2) % 3]
        else:
            Ni = Nc.copy()
        detJi = np.linalg.det(Ni)
        if detJi < 0:
            print("detJi negative", detJi)
        if not np.abs(detJi) < ZERO_TOL:
            sum += detJi
            # print(detJi)
            yield Ni, detJi
    print("full cut", sum)


def partial_cut_embedding_tri_iter(Nc, tip, range):
    ZERO_TOL = 1e-10
    Ni_template = np.zeros((3, 3))
    Ni_template[:, 0] = tip
    sum = 0
    for i in range:
        Ni = Ni_template.copy()
        Ni[int((i % 5 + 1) / 2), 1 + i % 2] = 1
        Ni[:, 2 - i % 2] = Nc[:, int(i / 2)]
        detJi = np.linalg.det(Ni)
        if detJi < 0:
            print("detJi negative", detJi)
        if not np.abs(detJi) < ZERO_TOL:
            sum += detJi
            # print(detJi)
            yield Ni, detJi
    print("partial cut", sum)


def jump_shape_functions(elem, shape_fn, pu_fn, nat_coords, tip_coords):
    n_points = nat_coords.shape[1]
    N, _ = shape_fn(nat_coords)
    Q, _ = pu_fn(nat_coords)
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
        N_jump[:, begin_tip::4] = 2 * sqrt_r[:, None] * ramp[:, None] * Q
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

    is_inside = np.all(cross >= 0.0, axis=1)
    return is_inside


def contains_points_3d(node_coords, points, tol=1e-10):
    points = np.atleast_2d(points)
    num_nodes = node_coords.shape[0]

    # 1. Define the triangular boundary faces based on element type.
    # Winding order is crucial: vertices must be ordered counter-clockwise
    # from the outside so the calculated normal points OUTWARD.
    if num_nodes == 4:
        # Tetrahedron (Standard FEM numbering: 0,1,2 on base, 3 is top)
        faces = np.array(
            [
                [0, 2, 1],  # Base (normal points down)
                [0, 1, 3],  # Front face
                [1, 2, 3],  # Right face
                [2, 0, 3],  # Left face
            ]
        )
    elif num_nodes == 8:
        # Hexahedron (Standard FEM: bottom 0,1,2,3 CCW; top 4,5,6,7 CCW)
        # We split the 6 quad faces into 12 triangles to handle warped faces.
        faces = np.array(
            [
                [0, 3, 2],
                [0, 2, 1],  # Bottom (normal points -z)
                [4, 5, 6],
                [4, 6, 7],  # Top    (normal points +z)
                [0, 1, 5],
                [0, 5, 4],  # Front  (normal points -y)
                [1, 2, 6],
                [1, 6, 5],  # Right  (normal points +x)
                [2, 3, 7],
                [2, 7, 6],  # Back   (normal points +y)
                [3, 0, 4],
                [3, 4, 7],  # Left   (normal points -x)
            ]
        )
    else:
        raise ValueError("Element must have exactly 4 (Tet) or 8 (Hex8) nodes.")

    # 2. Extract triangle vertices A, B, C for all faces
    A = node_coords[faces[:, 0]]  # Shape: (num_faces, 3)
    B = node_coords[faces[:, 1]]
    C = node_coords[faces[:, 2]]

    # 3. Compute outward normals using cross product: (B - A) x (C - A)
    # No need to normalize the vectors for a simple sign check
    normals = np.cross(B - A, C - A)  # Shape: (num_faces, 3)

    # 4. Calculate vectors from vertex A on each face to every test point
    # points: (num_points, 1, 3)  |  A: (1, num_faces, 3)
    vecs = points[:, None, :] - A[None, :, :]  # Shape: (num_points, num_faces, 3)

    # 5. Dot product of outward normals and point vectors
    # A positive dot product means the point is "outside" that face.
    # A negative dot product means the point is "inside".
    dots = np.sum(normals[None, :, :] * vecs, axis=2)  # Shape: (num_points, num_faces)

    # 6. Check if points are inside ALL faces
    is_inside = np.all(dots <= tol, axis=1)

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
        ).ravel()
        absent = np.bitwise_count(
            np.bitwise_and(
                local_dofs_per_node,
                np.bitwise_and(DOF_TYPES[:, None], np.bitwise_not(elem_dofs)),
            )
        ).ravel()
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


# def enriched_shape_functions(
#     elem,
#     shape_fn,
#     pu_fn,
#     nat_coords,
#     phi_n=None,
#     phi_t=None,
#     dphi_n_dxi=None,
#     dphi_t_dxi=None,
#     enforce_sign=None,
# ):
#     n_points = nat_coords.shape[1]
#     N = np.empty(
#         (
#             n_points,
#             elem.N_FN
#             + int(elem.h_enrich) * elem.H_FN
#             + int(elem.t_enrich) * elem.TIP_FN,
#         )
#     )
#     dN_dxi = np.empty(
#         (
#             n_points,
#             elem.DOFS,
#             elem.N_FN
#             + int(elem.h_enrich) * elem.H_FN
#             + int(elem.t_enrich) * elem.TIP_FN,
#         )
#     )
#     (N[:, : elem.N_FN], dN_dxi[:, :, : elem.N_FN]) = shape_fn(nat_coords)
#     if pu_fn is not None:
#         Q, dQ_dxi = pu_fn(nat_coords)
#     else:
#         Q, dQ_dxi = N[:, : elem.N_FN], dN_dxi[:, :, : elem.N_FN]
#
#     if phi_n is None or phi_t is None:
#         phi_n = np.sum(elem.phi_n * N[:, : elem.N_FN], axis=1)
#         phi_t = np.sum(elem.phi_t * N[:, : elem.N_FN], axis=1)
#     to_flip = None
#     if enforce_sign is not None:
#         to_flip = (phi_n != 0.0) & (np.sign(phi_n) != enforce_sign)
#         if np.any(to_flip):
#             print("warning had to flip", np.sum(to_flip))
#             print("phi_n:", elem.phi_n)
#         phi_n[to_flip] *= -1
#
#     if elem.h_enrich:
#         assert phi_n is not None and phi_t is not None
#         h_shifted = (np.sign(phi_n)[:, None] - np.sign(elem.phi_n)) / 2
#         begin_h, end_h = elem.N_FN, elem.N_FN + elem.H_FN
#         N[:, begin_h:end_h] = h_shifted * N[:, : elem.N_FN]
#         dN_dxi[:, :, begin_h:end_h] = h_shifted[:, None, :] * dN_dxi[:, :, : elem.N_FN]
#     if elem.t_enrich:
#         if dphi_n_dxi is None or dphi_t_dxi is None:
#             dphi_n_dxi = np.sum(elem.phi_n * dN_dxi[:, :, : elem.N_FN], axis=2)
#             dphi_t_dxi = np.sum(elem.phi_t * dN_dxi[:, :, : elem.N_FN], axis=2)
#         if enforce_sign is not None:
#             dphi_n_dxi[to_flip] *= -1
#         r = np.sqrt(phi_n**2 + phi_t**2)
#         r = np.maximum(r, 1e-14)  # avoid divide by zero
#         sqrt_r = np.sqrt(r)
#         sqrt_r_i = (elem.phi_n**2 + elem.phi_t**2) ** (1 / 4)
#         theta = np.atan2(phi_n, phi_t)
#         theta_i = np.atan2(elem.phi_n, elem.phi_t)
#         # sin(theta) = phi_n / r, cos(theta) = phi_t / r
#         dr_dxi = (
#             1 / r[:, None] * (phi_n[:, None] * dphi_n_dxi + phi_t[:, None] * dphi_t_dxi)
#         )  # = np.sin(theta) * dphi_n_dxi - np.cos(theta) * dphi_t_dxi
#         dtheta_dxi = (dphi_n_dxi * phi_t[:, None] - phi_n[:, None] * dphi_t_dxi) / (
#             phi_t**2 + phi_n**2
#         )[:, None]  # = (dphi_n_dxi * np.cos(theta) + np.sin(theta) dphi_t_dxi) / r
#         bf = branch_functions(sqrt_r, theta).T
#         bf_i = branch_functions(sqrt_r_i, theta_i).T
#
#         half_cos = np.cos(theta[:, None] / 2)
#         half_sin = np.sin(theta[:, None] / 2)
#         cos = np.cos(theta[:, None])
#         sin = np.sin(theta[:, None])
#
#         dbf_dxi = 1 / (2 * sqrt_r[:, None, None]) * dr_dxi.reshape(
#             (-1, elem.DOFS, 1)
#         ) * (bf / sqrt_r[:, None])[:, None, :] + sqrt_r[:, None, None] * np.array(
#             [
#                 half_cos * dtheta_dxi / 2,
#                 -half_sin * dtheta_dxi / 2,
#                 half_cos * dtheta_dxi / 2 * sin + half_sin * cos * dtheta_dxi,
#                 -half_sin * dtheta_dxi / 2 * sin + half_cos * cos * dtheta_dxi,
#             ]
#         ).transpose(1, 2, 0)
#
#         # dbf_dxi = 1 / (2 * sqrt_r[:, None, None]) * dr_dxi.reshape(
#         #     (-1, elem.DOFS, 1)
#         # ) * (bf / sqrt_r[:, None])[:, None, :] + sqrt_r[:, None, None] * np.array(
#         #     [
#         #         np.cos(theta[:, None] / 2) * dtheta_dxi / 2,
#         #         -np.sin(theta[:, None] / 2) * dtheta_dxi / 2,
#         #         np.cos(theta[:, None] / 2) * dtheta_dxi / 2 * np.sin(theta[:, None])
#         #         + np.sin(theta[:, None] / 2) * np.cos(theta[:, None]) * dtheta_dxi,
#         #         -np.sin(theta[:, None] / 2) * dtheta_dxi / 2 * np.sin(theta[:, None])
#         #         + np.cos(theta[:, None] / 2) * np.cos(theta[:, None]) * dtheta_dxi,
#         #     ]
#         # ).transpose(1, 2, 0)
#
#         ramp = np.sum(N[:, np.where(elem.in_range)[0]], axis=1)
#         dramp_dxi = np.sum(dN_dxi[:, :, np.where(elem.in_range)[0]], axis=2)
#
#         ramped_bf = bf * ramp[:, None]
#
#         ramp_i = elem.in_range.astype(float)
#         ramped_shifter = bf_i * ramp_i[:, None]
#
#         interpolant = np.sum(
#             ramped_shifter[None, :, :] * N[:, : elem.N_FN, None], axis=1
#         )
#
#         # interpolant = np.sum(bf_i[None, :, :] * N[:, : elem.N_FN, None], axis=1)
#         # bf_shifted = bf - interpolant
#
#         bf_shifted = ramped_bf - interpolant
#
#         begin_tip = elem.N_FN + int(elem.h_enrich) * elem.H_FN
#         end_tip = begin_tip + elem.TIP_FN
#
#         N[:, begin_tip:end_tip] = (bf_shifted[:, None, :] * Q[:, :, None]).reshape(
#             -1, elem.TIP_FN
#         )
#
#         # N[:, begin_tip:end_tip] = (
#         #     ramp[:, None, None] * bf_shifted[:, None, :] * Q[:, :, None]
#         # ).reshape(-1, elem.TIP_FN)
#
#         dramped_bf_dxi = (
#             dramp_dxi[:, :, None] * bf[:, None, :] + ramp[:, None, None] * dbf_dxi
#         )
#
#         dinterpolant_dxi = np.sum(
#             ramped_shifter[None, None, :, :] * dN_dxi[:, :, : elem.N_FN, None], axis=2
#         )
#
#         dbf_shifted_dxi = dramped_bf_dxi - dinterpolant_dxi
#
#         term_A = dbf_shifted_dxi[:, None, :, :] * Q[:, :, None, None]
#         term_B = bf_shifted[:, None, None, :] * dQ_dxi[:, :, :, None]
#
#         # dbf_shifted_dxi = dbf_dxi - np.sum(
#         #     bf_i[None, None, :, :] * dN_dxi[:, :, : elem.N_FN, None], axis=2
#         # )
#         #
#         # term_A = (
#         #     dbf_shifted_dxi[:, None, :, :]
#         #     * ramp[:, None, None, None]
#         #     * Q[:, :, None, None]
#         # )
#         #
#         # term_B = (
#         #     bf_shifted[:, None, None, :]
#         #     * dramp_dxi[:, :, None, None]
#         #     * Q[:, None, :, None]
#         # )
#         #
#         # term_C = (
#         #     bf_shifted[:, None, None, :]
#         #     * ramp[:, None, None, None]
#         #     * dQ_dxi[:, :, :, None]
#         # )
#         #
#         # combined_terms = term_A.swapaxes(1, 2) + term_B + term_C
#
#         combined_terms = term_A.swapaxes(1, 2) + term_B
#
#         N_points, N_dims = combined_terms.shape[0], combined_terms.shape[1]
#
#         dN_dxi[:, :, begin_tip:end_tip] = combined_terms.reshape(
#             N_points, N_dims, elem.TIP_FN
#         )
#     return N, dN_dxi
#


#
#
def enriched_shape_functions(
    elem,
    shape_fn,
    pu_fn,
    nat_coords,
    phi_n=None,
    phi_t=None,
    dphi_n_dxi=None,
    dphi_t_dxi=None,
    enforce_sign=None,
):
    n_points = nat_coords.shape[1]
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
    (N[:, : elem.N_FN], dN_dxi[:, :, : elem.N_FN]) = shape_fn(nat_coords)
    if pu_fn is not None:
        Q, dQ_dxi = pu_fn(nat_coords)
    else:
        Q, dQ_dxi = N[:, : elem.N_FN], dN_dxi[:, :, : elem.N_FN]

    if phi_n is None or phi_t is None:
        phi_n = np.sum(elem.phi_n * N[:, : elem.N_FN], axis=1)
        phi_t = np.sum(elem.phi_t * N[:, : elem.N_FN], axis=1)
    to_flip = None
    if enforce_sign is not None:
        # eps = 1e-16
        # is_zero = phi_n == 0.0
        # phi_n[is_zero] = enforce_sign * eps

        to_flip = np.sign(phi_n) != enforce_sign
        phi_n[to_flip] *= -1
        if np.any(to_flip):
            print("warning had to flip", np.sum(to_flip))

    if elem.h_enrich:
        assert phi_n is not None and phi_t is not None
        sign = np.where(phi_n >= 0.0, 1.0, -1.0)
        sign_i = np.where(elem.phi_n >= 0.0, 1.0, -1.0)
        # h_shifted = (np.sign(phi_n)[:, None] - np.sign(elem.phi_n)) / 2
        h_shifted = (sign[:, None] - sign_i) / 2
        begin_h, end_h = elem.N_FN, elem.N_FN + elem.H_FN
        N[:, begin_h:end_h] = h_shifted * N[:, : elem.N_FN]
        dN_dxi[:, :, begin_h:end_h] = h_shifted[:, None, :] * dN_dxi[:, :, : elem.N_FN]
    if elem.t_enrich:
        if dphi_n_dxi is None or dphi_t_dxi is None:
            dphi_n_dxi = np.sum(elem.phi_n * dN_dxi[:, :, : elem.N_FN], axis=2)
            dphi_t_dxi = np.sum(elem.phi_t * dN_dxi[:, :, : elem.N_FN], axis=2)
        if enforce_sign is not None:
            dphi_n_dxi[to_flip] *= -1
        r = np.sqrt(phi_n**2 + phi_t**2)
        r = np.maximum(r, 1e-14)  # avoid divide by zero
        sqrt_r = np.sqrt(r)
        phi_n_i_safe = np.where(elem.phi_n == 0.0, 1e-16, elem.phi_n)
        sqrt_r_i = (phi_n_i_safe**2 + elem.phi_t**2) ** (1 / 4)
        theta = np.atan2(phi_n, phi_t)
        theta_i = np.atan2(phi_n_i_safe, elem.phi_t)
        # sin(theta) = phi_n / r, cos(theta) = phi_t / r
        dr_dxi = (
            1 / r[:, None] * (phi_n[:, None] * dphi_n_dxi + phi_t[:, None] * dphi_t_dxi)
        )  # = np.sin(theta) * dphi_n_dxi - np.cos(theta) * dphi_t_dxi
        dtheta_dxi = (dphi_n_dxi * phi_t[:, None] - phi_n[:, None] * dphi_t_dxi) / (
            phi_t**2 + phi_n**2
        )[:, None]  # = (dphi_n_dxi * np.cos(theta) + np.sin(theta) dphi_t_dxi) / r
        bf = branch_functions(sqrt_r, theta).T
        bf_i = branch_functions(sqrt_r_i, theta_i).T

        half_cos = np.cos(theta[:, None] / 2)
        half_sin = np.sin(theta[:, None] / 2)
        cos = np.cos(theta[:, None])
        sin = np.sin(theta[:, None])

        dbf_dxi = 1 / (2 * sqrt_r[:, None, None]) * dr_dxi.reshape(
            (-1, elem.DOFS, 1)
        ) * (bf / sqrt_r[:, None])[:, None, :] + sqrt_r[:, None, None] * np.array(
            [
                half_cos * dtheta_dxi / 2,
                -half_sin * dtheta_dxi / 2,
                half_cos * dtheta_dxi / 2 * sin + half_sin * cos * dtheta_dxi,
                -half_sin * dtheta_dxi / 2 * sin + half_cos * cos * dtheta_dxi,
            ]
        ).transpose(1, 2, 0)

        ramp = np.sum(N[:, np.where(elem.in_range)[0]], axis=1)
        dramp_dxi = np.sum(dN_dxi[:, :, np.where(elem.in_range)[0]], axis=2)

        interpolant = np.sum(bf_i[None, :, :] * N[:, : elem.N_FN, None], axis=1)
        bf_shifted = bf - interpolant

        begin_tip = elem.N_FN + int(elem.h_enrich) * elem.H_FN
        end_tip = begin_tip + elem.TIP_FN

        N[:, begin_tip:end_tip] = (
            ramp[:, None, None] * bf_shifted[:, None, :] * Q[:, :, None]
        ).reshape(-1, elem.TIP_FN)

        dbf_shifted_dxi = dbf_dxi - np.sum(
            bf_i[None, None, :, :] * dN_dxi[:, :, : elem.N_FN, None], axis=2
        )

        term_A = (
            dbf_shifted_dxi[:, None, :, :]
            * ramp[:, None, None, None]
            * Q[:, :, None, None]
        )

        term_B = (
            bf_shifted[:, None, None, :]
            * dramp_dxi[:, :, None, None]
            * Q[:, None, :, None]
        )

        term_C = (
            bf_shifted[:, None, None, :]
            * ramp[:, None, None, None]
            * dQ_dxi[:, :, :, None]
        )

        combined_terms = term_A.swapaxes(1, 2) + term_B + term_C

        N_points, N_dims = combined_terms.shape[0], combined_terms.shape[1]

        dN_dxi[:, :, begin_tip:end_tip] = combined_terms.reshape(
            N_points, N_dims, elem.TIP_FN
        )
    return N, dN_dxi
