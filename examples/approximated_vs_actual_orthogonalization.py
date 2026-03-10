import tfealite as tf
import scipy as sp
import numpy as np
from tfealite.core.dofs import HEAVISIDE_DOFS, BRANCH_DOFS, BASE_DOFS
import matplotlib.pyplot as plt


def cal_approx_coeff(model):
    coef = []
    T_global = sp.sparse.eye(model.Kg.shape[0], format="csr")
    Kg_work = model.Kg.copy().tocsr()

    for id, dofs in enumerate(model.list_dof.list_dof):
        l_node = []
        if dofs & (HEAVISIDE_DOFS | BRANCH_DOFS) == 0:
            continue

        n_dof_per_node = model.dof_per_node.bit_count()

        # ADDED BASE_DOFS BACK IN so the matrix actually gets conditioned!
        dof_numbers = np.concatenate(
            (
                model.list_dof.get_elem_dof_numbers_flat(
                    id + 1, HEAVISIDE_DOFS
                ).flatten(),
                model.list_dof.get_elem_dof_numbers_flat(id + 1, BRANCH_DOFS).flatten(),
            )
        )

        for j in range(n_dof_per_node, len(dof_numbers), n_dof_per_node):
            j_dofs = dof_numbers[j : j + n_dof_per_node]

            for i in range(0, j, n_dof_per_node):
                i_dofs = dof_numbers[i : i + n_dof_per_node]

                denom = Kg_work[i_dofs[0], i_dofs[0]] + Kg_work[i_dofs[1], i_dofs[1]]
                num = Kg_work[i_dofs[0], j_dofs[0]] + Kg_work[i_dofs[1], j_dofs[1]]

                T_step = sp.sparse.eye(model.Kg.shape[0], format="lil")
                l_val = num / denom

                for _, (i_idx, j_idx) in enumerate(zip(i_dofs, j_dofs)):
                    T_step[i_idx, j_idx] = -l_val
                l_node.append(l_val)

                T_step = T_step.tocsr()

                Kg_work = T_step.T @ Kg_work @ T_step
                T_global = T_global @ T_step
        coef.append(l_node)
    return coef


# def calculate_angle(model, ortho_T):
#     thetag = []
#     Kg = ortho_T.T @ model.Kg.copy().tocsr() @ ortho_T
#
#     for id, dofs in enumerate(model.list_dof.list_dof):
#         if dofs & (HEAVISIDE_DOFS | BRANCH_DOFS) == 0:
#             continue
#
#         dof_numbers = np.concatenate(
#             (
#                 model.list_dof.get_elem_dof_numbers_flat(id + 1, BASE_DOFS).flatten(),
#                 model.list_dof.get_elem_dof_numbers_flat(
#                     id + 1, HEAVISIDE_DOFS
#                 ).flatten(),
#                 model.list_dof.get_elem_dof_numbers_flat(id + 1, BRANCH_DOFS).flatten(),
#             )
#         )
#         theta = np.zeros((len(dof_numbers), len(dof_numbers)))
#
#         for j, j_dof in enumerate(dof_numbers):
#             djj = Kg[j_dof, j_dof]
#             for i, i_dof in enumerate(dof_numbers):
#                 dii = Kg[i_dof, i_dof]
#                 dij = Kg[j_dof, i_dof]
#                 theta[i, j] = np.acos(dij / np.sqrt(dii * djj)) / np.pi * 180
#
#         thetag.append(theta)
#
#     return thetag
def calculate_angle(model, ortho_T):
    thetag = []
    Kg = ortho_T.T @ model.Kg.copy().tocsr() @ ortho_T

    for id, dofs in enumerate(model.list_dof.list_dof):
        if dofs & (HEAVISIDE_DOFS | BRANCH_DOFS) == 0:
            continue

        dof_numbers = np.concatenate(
            (
                model.list_dof.get_elem_dof_numbers_flat(id + 1, BASE_DOFS).flatten(),
                model.list_dof.get_elem_dof_numbers_flat(
                    id + 1, HEAVISIDE_DOFS
                ).flatten(),
                model.list_dof.get_elem_dof_numbers_flat(id + 1, BRANCH_DOFS).flatten(),
            )
        )

        K_sub = Kg[np.ix_(dof_numbers, dof_numbers)].toarray()

        diag = np.diag(K_sub)

        denom = np.sqrt(np.outer(diag, diag))

        cos_theta = K_sub / (denom + 1e-14)

        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        theta = np.arccos(cos_theta) / np.pi * 180.0

        thetag.append(theta)

    return thetag


def estimate_condition_number_ilu(A_sparse, drop_tol=1e-4):
    """
    Estimates the condition number of a sparse matrix using the
    diagonal pivots of its Incomplete LU (ILU) factorization.
    """
    import scipy.sparse.linalg as spla

    try:
        # 1. Perform Incomplete LU factorization
        # We use a drop tolerance to speed it up. If you want a more
        # accurate (but slower) estimate, decrease drop_tol (e.g., 1e-6)
        ilu = spla.spilu(A_sparse, drop_tol=drop_tol)

        # 2. Extract the diagonal of the U factor
        U_diag = ilu.U.diagonal()

        # 3. Take the absolute values of the pivots
        abs_pivots = np.abs(U_diag)

        # 4. Find the max and min pivots
        max_pivot = np.max(abs_pivots)
        min_pivot = np.min(abs_pivots)

        # Prevent division by zero if the matrix is perfectly singular
        if min_pivot == 0:
            print("Warning: Minimum pivot is exactly zero. Matrix is singular.")
            return np.inf

        # 5. Calculate the ratio
        cond_estimate = max_pivot / min_pivot

        return cond_estimate

    except RuntimeError as e:
        # spilu will throw a RuntimeError if it encounters a zero pivot
        # during factorization that it cannot work around.
        print(f"ILU Factorization failed: {e}")
        print(
            "This usually means the condition number is effectively infinite (> 10^16)."
        )
        return np.inf


def scaled_condition_number(K_sparse):
    """
    Calculates the diagonally scaled condition number of a sparse matrix K.
    Formula: k_d = D * K * D, where D_ii = K_ii^(-1/2)
    """
    import scipy.sparse.linalg as spla

    # 1. Extract the main diagonal of K
    diag_K = K_sparse.diagonal()

    # Safety Check: Stiffness matrices should have positive diagonals.
    # If there are zeros or negative numbers, it indicates a rigid body
    # motion (unconstrained boundary) or a collapsed element.
    if np.any(diag_K <= 0):
        print("Warning: Non-positive values found on the diagonal!")
        # Temporarily replace <= 0 values with a tiny number to prevent
        # divide-by-zero or NaN errors during the square root.
        diag_K = np.where(diag_K > 0, diag_K, 1e-12)

    # 2. Calculate the diagonal elements of D: D_ii = 1 / sqrt(K_ii)
    diag_D = 1.0 / np.sqrt(diag_K)

    # 3. Construct D as a sparse diagonal matrix
    D = sp.sparse.diags(diag_D)

    # 4. Compute the scaled matrix Kd = D * K * D
    # The '@' operator in Python natively handles sparse matrix multiplication
    Kd = D @ K_sparse @ D

    # 5. Attempt to find the exact eigenvalues of Kd
    try:
        # Find the largest magnitude eigenvalue
        # (Using a looser tolerance 1e-3 to speed up convergence)
        lambda_max, _ = spla.eigsh(Kd, k=1, which="LM", tol=1e-3)

        # Find the smallest magnitude eigenvalue
        lambda_min, _ = spla.eigsh(Kd, k=1, which="SM", tol=1e-3)

        scaled_cond = np.abs(lambda_max[0] / lambda_min[0])
        print("Successfully calculated exact scaled condition number.")
        return scaled_cond

    except spla.ArpackNoConvergence:
        print("eigsh failed to converge on Kd. Falling back to ILU estimation...")
        # 6. Fallback: Use the ILU diagonal ratio trick on Kd
        try:
            ilu = spla.spilu(Kd, drop_tol=1e-4)
            U_diag = np.abs(ilu.U.diagonal())
            return np.max(U_diag) / np.min(U_diag)
        except RuntimeError as e:
            print(f"ILU also failed: {e}. Matrix is likely singular.")
            return np.inf


def cal_exact_coeff(model):
    coef = []
    T_global = sp.sparse.eye(model.Kg.shape[0], format="csr")
    Kg_work = model.Kg.copy().tocsr()

    for id, dofs in enumerate(model.list_dof.list_dof):
        l_node = []
        if dofs & (HEAVISIDE_DOFS | BRANCH_DOFS) == 0:
            continue

        n_dof_per_node = model.dof_per_node.bit_count()

        # ADDED BASE_DOFS BACK IN so the matrix actually gets conditioned!
        dof_numbers = np.concatenate(
            (
                model.list_dof.get_elem_dof_numbers_flat(
                    id + 1, HEAVISIDE_DOFS
                ).flatten(),
                model.list_dof.get_elem_dof_numbers_flat(id + 1, BRANCH_DOFS).flatten(),
            )
        )

        for j in range(n_dof_per_node, len(dof_numbers), n_dof_per_node):
            j_dofs = dof_numbers[j : j + n_dof_per_node]

            for i in range(0, j, n_dof_per_node):
                i_dofs = dof_numbers[i : i + n_dof_per_node]

                # Extract 1D arrays of the diagonals [K_xx, K_yy, ...]
                # denom = Kg_work[i_dofs, i_dofs].diagonal()
                # denom = np.array([])
                # num = Kg_work[i_dofs, j_dofs].diagonal()

                T_step = sp.sparse.eye(model.Kg.shape[0], format="lil")

                # We iterate through the x, y (and z) components simultaneously
                li = []
                for idx, (i_idx, j_idx) in enumerate(zip(i_dofs, j_dofs)):
                    denom = Kg_work[i_idx, i_idx]
                    num = Kg_work[i_idx, j_idx]
                    # print(denom, num)
                    # print(num, denom)
                    l_val = num / denom
                    T_step[i_idx, j_idx] = -l_val
                    li.append(l_val)
                l_node.append(li)

                T_step = T_step.tocsr()

                Kg_work = T_step.T @ Kg_work @ T_step
                T_global = T_global @ T_step
        coef.append(l_node)
    return coef


y_crack = np.linspace(0.1, 0.9, 30)
l_approx = []
l_exact = []
orth_angles = []
base_angles = []


def elem_func(
    node_coords,
    material,
    real,
    phi_n=None,
    phi_t=None,
    h_enrich: bool = False,
    t_enrich: bool = False,
    partial_cut: bool = False,
    h_enrich_per_node=None,
):
    elem = tf.XQuad4n(
        node_coords,
        material,
        real,
        phi_n,
        phi_t,
        h_enrich,
        t_enrich,
        partial_cut,
        h_enrich_per_node,
    )
    elem.C = np.eye(3)
    elem.C[2, 2] = 0
    return elem


n = np.arange(11, 50, 2)
conditioning_no_orth = []
conditioning = []
conditioning_fem = []

for i in n:
    nodes, elements = tf.gen_rect_Quad4n(L=1.0, H=1.0, nx=i, ny=i)
    materials = [[1, {"E": 1, "nu": 0.33, "rho": 7850}]]
    reals = [[1, {"t": 1}]]
    model = tf.XFEModel(
        nodes,
        elements,
        materials,
        reals,
        tip_enrichment=True,
        geometrical_range=0.1,
        corrected=True,
    )
    p1 = np.array([-0.1, 0.5])
    p2 = np.array([0.5, 0.65])
    model.insert_crack_segment(p1, p2, embedded=False)
    model.gen_list_dof(dof_per_node=tf.IS_2D)
    # model.list_dof.remove_dofs(
    #     1 + np.arange(model.n_nodes), tf.DofType.HX | tf.DofType.HY
    # )
    model.cal_global_matrices(tf.XQuad4n)
    c0 = scaled_condition_number(model.Kg)
    c1 = scaled_condition_number(model.ortho_T.T @ model.Kg @ model.ortho_T)
    conditioning_no_orth.append(c0)
    conditioning.append(c1)
    model = tf.FEModel(nodes, elements, materials, reals)
    model.gen_list_dof(dof_per_node=tf.IS_2D)
    model.cal_global_matrices(tf.XQuad4n)
    c_fem = scaled_condition_number(model.Kg)
    conditioning_fem.append(c_fem)


plt.figure()
plt.plot(n, conditioning_no_orth, label="no orth")
plt.plot(n, conditioning, label="orth")
plt.plot(n, conditioning_fem, label="fem")
plt.yscale("log", base=10)
plt.xscale("log", base=10)
plt.title("conditioning")
plt.legend()
plt.show()
