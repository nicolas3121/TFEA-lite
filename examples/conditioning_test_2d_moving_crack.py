import tfealite as tf
from geomdl import knotvector
from scipy.interpolate import BSpline
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
# n = np.array([11, 21, 31, 41, 49, 61, 81, 121, 161])

n = 55
n_steps = 10
h = 10 / n
step = h / 2 / n_steps
# n = np.array([11, 41, 56, 80])
conditioning = []

distance_vals = [h / 2 - step * i for i in range(n_steps + 1)]


for i in range(n_steps + 1):
    nodes, elements = tf.gen_rect_Quad4n(L=10.0, H=10.0, nx=n, ny=n)
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
    L_crack = 10
    crack_angle = 0
    # p1 = np.array([-0.1, 0.5 - i * step])
    # p2 = np.array([0.5, 0.5 - i * step])

    tip_x, tip_y = 5.0, 5.0 - i * step
    p1 = np.array(
        [tip_x - L_crack * np.cos(crack_angle), tip_y - L_crack * np.sin(crack_angle)]
    )
    p2 = np.array([tip_x, tip_y])

    control_points = np.linspace(p1, p2, 12).tolist()
    n_spline = len(control_points)
    k = 2
    knots = knotvector.generate(k, n_spline)
    bspline = BSpline(knots, np.array(control_points), k)

    model.insert_crack_spline(bspline, embedded=False, h=h, snapping_tolerance=0.05)
    model.gen_list_dof(dof_per_node=tf.IS_2D)
    model.cal_global_matrices({"Quad4n": tf.XQuad4n}, eval_mass=False)

    def sel_condition(x, y, z):
        return y - 0.0

    model.gen_dirichlet_bc(sel_condition)

    Kg_bc = model.P.T @ model.ortho_T.T @ model.Kg @ model.ortho_T @ model.P

    c1 = scaled_condition_number(Kg_bc)
    conditioning.append(c1)


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


plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",  # Matches standard LaTeX/Word fonts
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,  # Slightly smaller for ticks to prevent overlap
        "ytick.labelsize": 10,
        "legend.fontsize": 10,  # Scale legend down slightly to fit the small box
    }
)

# 2. Set the figure size to exactly 3.15 inches wide
# (Height is set to 2.5 inches for a nice aspect ratio, adjust as needed)
fig, ax = plt.subplots(figsize=(3.15, 2.5))

plt.plot(
    distance_vals,
    conditioning,
    linestyle="-",
    linewidth=1.0,  # Scale down linewidth slightly for the smaller canvas
    color="#D95319",
    marker="s",
    markersize=5,  # Scale down markers slightly
    markerfacecolor="none",
    markeredgewidth=1.0,
    label="Stable-Corrected XFEM",
)

plt.yscale("log")
plt.ylim(1e4, 1e7)

plt.xlabel("Distance to Element Edge")
plt.ylabel("Condition Number $\\kappa_d$")

# Use thinner grid lines for the smaller canvas
plt.grid(True, which="major", ls="-", color="lightgray", alpha=0.8, linewidth=0.5)
plt.grid(True, which="minor", ls="--", color="lightgray", alpha=0.4, linewidth=0.5)

# plt.legend(loc="lower right", framealpha=1.0, edgecolor="black", fancybox=False)

# 3. Use tight_layout with zero padding so it doesn't waste space
plt.tight_layout(pad=0.1)
plt.savefig("scn_moving_crack.pdf", dpi=300, bbox_inches="tight")
plt.show()
