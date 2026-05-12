import tfealite as tf
import scipy as sp
import numpy as np
from tfealite.core.dofs import HEAVISIDE_DOFS, BRANCH_DOFS, BASE_DOFS, BRANCH_4_DOFS
from tfealite.core import bc
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
        # lambda_min, _ = spla.eigsh(Kd, k=1, sigma=1e-10, which="LM")

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


# n = np.array([11, 21, 41, 81])
n = np.array([11, 15, 21, 31, 41, 49, 55, 61, 71, 81, 91, 121, 161])
conditioning_no_orth = []
conditioning = []
conditioning_fem = []


for i in n:
    nodes, elements = tf.gen_rect_Quad4n(L=10.0, H=10.0, nx=i, ny=i)
    materials = [[1, {"E": 1, "nu": 0.33, "rho": 7850}]]
    reals = [[1, {"t": 1}]]
    model = tf.XFEModel(
        nodes,
        elements,
        materials,
        reals,
        tip_enrichment=True,
        geometrical_range=1.2,
        corrected=True,
    )
    p1 = np.array([-0.1, 0.5])
    p2 = np.array([0.5, 0.5])
    model.insert_crack_segment(p1, p2, embedded=False)
    model.gen_list_dof(dof_per_node=tf.IS_2D)
    # model.list_dof.remove_dofs(
    #     1 + np.arange(model.n_nodes), tf.DofType.HX | tf.DofType.HY
    # )
    model.cal_global_matrices({"Quad4n": tf.XQuad4n}, eval_mass=False)

    def sel_condition(x, y, z):
        return y - 0.0

    blending_nodes = (model.in_range == 0) & (
        model.list_dof.list_dof & BRANCH_DOFS != 0
    )

    extra_fix_dofs = model.list_dof.get_elem_dof_numbers_flat(
        np.where(blending_nodes)[0][:2] + 1, BRANCH_4_DOFS
    )

    bc.my_gen_dirichlet_bc(model, sel_condition, extra_fix_dofs)
    # model.gen_dirichlet_bc(sel_condition)

    # 3. Scale both the matrix and the load vector
    # Kg_bc = model.ortho_T.T @ model.Kg @ model.ortho_T
    Kg_bc = model.P.T @ model.ortho_T.T @ model.Kg @ model.ortho_T @ model.P

    # if np.any(dead_mask):
    #     print(f"   - Eliminating {np.sum(dead_mask)} dead/redundant DOFs...")
    #
    #     # Create sparse diagonal filtering matrices
    #     S_alive = sp.sparse.diags((~dead_mask).astype(float))
    #     S_dead = sp.sparse.diags(dead_mask.astype(float))
    #
    #     # Algebraically zero out rows/cols of dead DOFs, then put 1.0 on their diagonals
    #     Kg_bc = S_alive @ Kg_bc @ S_alive + S_dead
    #
    #     # Zero out the corresponding entries in the load vector
    #     # Fg_bc = S_alive @ Fg_bc

    c1 = scaled_condition_number(Kg_bc)
    conditioning.append(c1)
    model = tf.FEModel(nodes, elements, materials, reals)
    model.gen_list_dof(dof_per_node=tf.IS_2D)
    model.cal_global_matrices({"Quad4n": tf.XQuad4n})
    c_fem = scaled_condition_number(model.Kg)
    conditioning_fem.append(c_fem)
    if i < 10:
        nodes, elements = tf.gen_rect_Quad4n(L=1.0, H=1.0, nx=i, ny=i)
        materials = [[1, {"E": 1, "nu": 0.33, "rho": 7850}]]
        reals = [[1, {"t": 1}]]
        model = tf.XFEModel(
            nodes,
            elements,
            materials,
            reals,
            tip_enrichment=True,
            geometrical_range=0.12,
            corrected=True,
        )
        p1 = np.array([-0.1, 0.5])
        p2 = np.array([0.5, 0.5])
        model.insert_crack_segment(p1, p2, embedded=False)
        model.gen_list_dof(dof_per_node=tf.IS_2D)
        # model.list_dof.remove_dofs(
        #     1 + np.arange(model.n_nodes), tf.DofType.HX | tf.DofType.HY
        # )
        model.cal_global_matrices({"Quad4n": tf.XQuad4n})

        def sel_condition(x, y, z):
            return y - 0.0

        # bc.my_gen_dirichlet_bc(model, sel_condition, to_delete)
        model.gen_dirichlet_bc(sel_condition)

        # 3. Scale both the matrix and the load vector
        # Kg_bc = model.ortho_T.T @ model.Kg @ model.ortho_T
        Kg_bc = model.P.T @ model.Kg @ model.P
        c0 = scaled_condition_number(Kg_bc)
        conditioning_no_orth.append(c0)
    else:
        conditioning_no_orth.append(np.nan)


def annotate_local_slopes(x_vals, y_vals, ax, text_offset, x_is_h=False):
    """
    Calculates the local log-log slope between consecutive points and adds
    an arrow annotation to the midpoint of the line segment.
    """
    for i in range(len(x_vals) - 1):
        x1, x2 = x_vals[i], x_vals[i + 1]
        y1, y2 = y_vals[i], y_vals[i + 1]

        # Calculate slope. If x is 'h' (which decreases), we invert the x-ratio
        # so the rate correctly reflects growth with respect to 'n'.
        if x_is_h:
            rate = np.log(y2 / y1) / np.log(x1 / x2)
        else:
            rate = np.log(y2 / y1) / np.log(x2 / x1)

        # Calculate midpoint in log-space for accurate arrow placement
        x_mid = np.exp((np.log(x1) + np.log(x2)) / 2)
        y_mid = np.exp((np.log(y1) + np.log(y2)) / 2)

        # Add the annotation with an arrow
        ax.annotate(
            f"{rate:.3f}",
            xy=(x_mid, y_mid),
            xytext=text_offset,
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=11,
            arrowprops=dict(arrowstyle="->", color="black", shrinkA=0, shrinkB=6),
        )


# --- Assuming you have an array `n_vals` (e.g., n_vals = np.array([11, 21, 41, 81, 161])) ---
# If you prefer plotting against `h_vals`, just swap `n_vals` with `h_vals` in the plt.loglog
# calls and set x_is_h=True in the annotate_local_slopes function.
# --- Assuming you have an array `n` (e.g., n = np.array([11, 21, 41, 81])) ---
h_vals = 1.0 / n

plt.figure(figsize=(8, 6))
ax = plt.gca()

# 1. Standard FEM
plt.loglog(
    h_vals,
    conditioning_fem,
    linestyle="--",
    linewidth=1.5,
    color="#0072BD",  # Matplotlib standard blue
    marker="o",
    markersize=7,
    markerfacecolor="none",  # Open marker
    markeredgewidth=1.5,
    label="FEM",
)
# Annotate FEM (set x_is_h=True)
annotate_local_slopes(h_vals, conditioning_fem, ax, text_offset=(20, -20), x_is_h=True)

# 2. Stable-Corrected XFEM
plt.loglog(
    h_vals,
    conditioning,
    linestyle="-",
    linewidth=1.5,
    color="#D95319",  # Matplotlib standard orange/red
    marker="s",
    markersize=7,
    markerfacecolor="none",
    markeredgewidth=1.5,
    label="Stable-Corrected XFEM",
)
# Annotate SC-XFEM (set x_is_h=True)
annotate_local_slopes(h_vals, conditioning, ax, text_offset=(20, -20), x_is_h=True)

# 3. Standard XFEM (No Orthogonalization)
plt.loglog(
    h_vals,
    conditioning_no_orth,
    linestyle="--",
    linewidth=1.5,
    color="#EDB120",  # Matplotlib standard yellow/orange
    marker="v",
    markersize=7,
    markerfacecolor="none",
    markeredgewidth=1.5,
    label="Standard XFEM",
)
# Annotate Standard XFEM (set x_is_h=True)
annotate_local_slopes(
    h_vals, conditioning_no_orth, ax, text_offset=(-25, 25), x_is_h=True
)

# --- Formatting for Publication ---
plt.xlabel("Element Size $h$", fontsize=13)
plt.ylabel("Scaled Condition Number $\\kappa_d$", fontsize=13)

# Invert X axis so the mesh gets "finer" (smaller h) as you read left to right
plt.gca().invert_xaxis()

# Add major and minor grids for log-scale readability
plt.grid(True, which="major", ls="-", color="lightgray", alpha=0.8)
plt.grid(True, which="minor", ls="--", color="lightgray", alpha=0.4)

# Move the legend to the bottom right!
plt.legend(
    fontsize=11, loc="lower right", framealpha=1.0, edgecolor="black", fancybox=False
)

# Set x-axis ticks to show the exact h_vals, formatted to 3 decimal places
plt.xticks(h_vals, labels=[f"{h:.3f}" for h in h_vals])

plt.tight_layout()
plt.savefig("scn_corrected_annotated.pdf", dpi=300, bbox_inches="tight")
plt.show()

# plt.figure(figsize=(8, 6))
# ax = plt.gca()
#
# # 1. Standard FEM
# plt.loglog(
#     n,
#     conditioning_fem,
#     linestyle="--",
#     linewidth=1.5,
#     color="#0072BD",  # Matplotlib standard blue
#     marker="o",
#     markersize=7,
#     markerfacecolor="none",  # Open marker
#     markeredgewidth=1.5,
#     label="FEM",
# )
# # Annotate FEM (arrows pointing slightly down and right)
# annotate_local_slopes(n, conditioning_fem, ax, text_offset=(20, -20))
#
# # 2. Stable-Corrected XFEM
# plt.loglog(
#     n,
#     conditioning,
#     linestyle="-",
#     linewidth=1.5,
#     color="#D95319",  # Matplotlib standard orange/red
#     marker="s",
#     markersize=7,
#     markerfacecolor="none",
#     markeredgewidth=1.5,
#     label="Stable-Corrected XFEM",
# )
# # Annotate SC-XFEM (arrows pointing slightly down and right)
# annotate_local_slopes(n, conditioning, ax, text_offset=(20, -20))
#
# # 3. Standard XFEM (No Orthogonalization)
# plt.loglog(
#     n,
#     conditioning_no_orth,
#     linestyle="--",
#     linewidth=1.5,
#     color="#EDB120",  # Matplotlib standard yellow/orange
#     marker="v",
#     markersize=7,
#     markerfacecolor="none",
#     markeredgewidth=1.5,
#     label="Standard XFEM (No Orth)",
# )
# # Annotate Standard XFEM (arrows pointing up and left to avoid crossing lines)
# annotate_local_slopes(n, conditioning_no_orth, ax, text_offset=(-25, 25))
#
# # --- Formatting for Publication ---
# plt.xlabel("n", fontsize=13)
# plt.ylabel("scaled condition number", fontsize=13)
#
# # Add major and minor grids for log-scale readability
# plt.grid(True, which="major", ls="-", color="lightgray", alpha=0.8)
# plt.grid(True, which="minor", ls="--", color="lightgray", alpha=0.4)
#
# # Configure the legend to match the reference image
# plt.legend(
#     fontsize=11, loc="upper left", framealpha=1.0, edgecolor="black", fancybox=False
# )
#
# # Optional: Ensure x-axis ticks show exactly the n_vals provided
# plt.xticks(n, labels=[str(int(n)) for n in n])
#
# plt.tight_layout()
# plt.savefig("scn_corrected_annotated.pdf", dpi=300, bbox_inches="tight")
# plt.show()
