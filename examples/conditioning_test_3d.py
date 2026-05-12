import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

import tfealite as tf
from tfealite.core.dofs import DofType
from tfealite.elements.XTetr4n import XTetr4n


def scaled_condition_number(K_sparse):
    """
    Calculates the diagonally scaled condition number of a sparse matrix K.
    Formula: k_d = D * K * D, where D_ii = K_ii^(-1/2)
    """
    diag_K = K_sparse.diagonal()

    if np.any(diag_K <= 0):
        print("Warning: Non-positive values found on the diagonal!")
        diag_K = np.where(diag_K > 0, diag_K, 1e-12)

    diag_D = 1.0 / np.sqrt(diag_K)
    D = sps.diags(diag_D)
    Kd = D @ K_sparse @ D

    try:
        lambda_max, _ = spla.eigsh(Kd, k=1, which="LM", tol=1e-3)
        lambda_min, _ = spla.eigsh(Kd, k=1, which="SM", tol=1e-3)

        scaled_cond = np.abs(lambda_max[0] / lambda_min[0])
        print(f"Calculated scaled condition number: {scaled_cond:.2e}")
        return scaled_cond

    except spla.ArpackNoConvergence:
        print("eigsh failed to converge on Kd. Falling back to ILU estimation...")
        try:
            ilu = spla.spilu(Kd, drop_tol=1e-4)
            U_diag = np.abs(ilu.U.diagonal())
            return np.max(U_diag) / np.min(U_diag)
        except RuntimeError as e:
            print(f"ILU also failed: {e}. Matrix is likely singular.")
            return np.inf


def annotate_local_slopes(x_vals, y_vals, ax, text_offset, x_is_h=False):
    """
    Calculates the local log-log slope between consecutive points and adds
    an arrow annotation to the midpoint of the line segment.
    """
    # Filter out NaNs for slope calculation
    valid_idx = ~np.isnan(y_vals)
    x_v = x_vals[valid_idx]
    y_v = np.array(y_vals)[valid_idx]

    for i in range(len(x_v) - 1):
        x1, x2 = x_v[i], x_v[i + 1]
        y1, y2 = y_v[i], y_v[i + 1]

        if x_is_h:
            rate = np.log(y2 / y1) / np.log(x1 / x2)
        else:
            rate = np.log(y2 / y1) / np.log(x2 / x1)

        x_mid = np.exp((np.log(x1) + np.log(x2)) / 2)
        y_mid = np.exp((np.log(y1) + np.log(y2)) / 2)

        ax.annotate(
            f"{rate:.2f}",
            xy=(x_mid, y_mid),
            xytext=text_offset,
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="black", shrinkA=0, shrinkB=6),
        )


def run_condition_number_study_3d():
    # Keep n smaller for 3D because matrix size scales cubically!
    n_vals = np.array([5, 9, 13, 17, 21, 31, 41, 51])
    h_vals = 10.0 / n_vals

    cond_fem = []
    cond_sc = []
    cond_std = []

    L, H, D = 10.0, 10.0, 2.0
    tol = 1e-8

    for i in n_vals:
        print(f"\n--- Processing 3D Mesh {i}x{i}x2 ---")
        nodes, elements = tf.core.model.gen_rect_Tetr4n(L, H, D, i, i, 2)
        materials = [[1, {"E": 1.0, "nu": 0.3, "rho": 1.0}]]
        reals = [[1, {}]]

        # Crack spans entire thickness for plane strain
        p1 = np.array([0.0, 5.0, -1.0])
        p2 = np.array([5.0, 5.0, -1.0])
        p3 = np.array([5.0, 5.0, 3.0])

        all_nodes = np.arange(1, len(nodes) + 1)
        bottom_nodes = np.where(np.abs(nodes[:, 2]) < tol)[0] + 1  # y=0 plane

        # =========================================================
        # 1. STABLE-CORRECTED XFEM
        # =========================================================
        model_sc = tf.XFEModel(
            nodes,
            elements,
            materials,
            reals,
            tip_enrichment=True,
            geometrical_range=2,
            corrected=True,
        )
        model_sc.insert_planar_crack_segment(p1, p2, p3, embedded=False)
        model_sc.gen_list_dof(dof_per_node=tf.IS_3D)
        model_sc.cal_global_matrices({"Tetr4n": XTetr4n}, eval_mass=False)

        fix_dofs = []

        fix_dofs.extend(
            model_sc.list_dof.get_elem_dof_numbers_flat(bottom_nodes, tf.IS_3D)
        )
        # B. Enforce Plane Strain (Lock all Z DOFs everywhere)
        uz_dofs = np.concatenate(
            [
                model_sc.list_dof.get_elem_dof_numbers_flat(all_nodes, dof)
                for dof in [
                    DofType.UZ,
                    DofType.HZ,
                    DofType.B1Z,
                    DofType.B2Z,
                    DofType.B3Z,
                    DofType.B4Z,
                ]
            ]
        )
        fix_dofs.extend(list(uz_dofs))

        model_sc.gen_P(np.unique(sorted(fix_dofs)))

        Kg_sc_bc = (
            model_sc.P.T
            @ model_sc.ortho_T.T
            @ model_sc.Kg
            @ model_sc.ortho_T
            @ model_sc.P
        )
        cond_sc.append(scaled_condition_number(Kg_sc_bc))

        # =========================================================
        # 2. STANDARD FEM
        # =========================================================
        model_fem = tf.FEModel(nodes, elements, materials, reals)
        model_fem.gen_list_dof(dof_per_node=tf.IS_3D)

        # NOTE: Assuming you have tf.Tetr4n as the standard element class
        try:
            model_fem.cal_global_matrices({"Tetr4n": tf.elements.Tetr4n.Tetr4n})
        except AttributeError:
            model_fem.cal_global_matrices(
                {"Tetr4n": XTetr4n}
            )  # Fallback if Tetr4n isn't imported

        fix_dofs_fem = []
        fix_dofs_fem.extend(
            model_fem.list_dof.get_elem_dof_numbers_flat(bottom_nodes, tf.IS_3D)
        )
        fix_dofs_fem.extend(
            model_fem.list_dof.get_elem_dof_numbers_flat(all_nodes, DofType.UZ)
        )

        model_fem.gen_P(np.unique(sorted(fix_dofs_fem)))
        Kg_fem_bc = model_fem.P.T @ model_fem.Kg @ model_fem.P
        cond_fem.append(scaled_condition_number(Kg_fem_bc))

        # =========================================================
        # 3. STANDARD XFEM (No Orthogonalization)
        # =========================================================
        # We only calculate this for smaller meshes to prevent Arpack explosions
        if i <= 31:
            model_std = tf.XFEModel(
                nodes,
                elements,
                materials,
                reals,
                tip_enrichment=True,
                geometrical_range=1.5,
                corrected=False,
            )
            model_std.insert_planar_crack_segment(p1, p2, p3, embedded=False)
            model_std.gen_list_dof(dof_per_node=tf.IS_3D)
            model_std.cal_global_matrices({"Tetr4n": XTetr4n}, eval_mass=False)

            fix_dofs_std = []

            fix_dofs.extend(
                model_std.list_dof.get_elem_dof_numbers_flat(bottom_nodes, tf.IS_3D)
            )

            uz_dofs_std = np.concatenate(
                [
                    model_std.list_dof.get_elem_dof_numbers_flat(all_nodes, dof)
                    for dof in [
                        DofType.UZ,
                        DofType.HZ,
                        DofType.B1Z,
                        DofType.B2Z,
                        DofType.B3Z,
                        DofType.B4Z,
                    ]
                ]
            )
            fix_dofs_std.extend(list(uz_dofs_std))

            model_std.gen_P(np.unique(sorted(fix_dofs_std)))
            Kg_std_bc = model_std.P.T @ model_std.Kg @ model_std.P
            cond_std.append(scaled_condition_number(Kg_std_bc))
        else:
            cond_std.append(np.nan)

    # =========================================================
    # PLOTTING
    # =========================================================
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # 1. Standard FEM
    plt.loglog(
        h_vals,
        cond_fem,
        linestyle="--",
        linewidth=1.5,
        color="#0072BD",
        marker="o",
        markersize=7,
        markerfacecolor="none",
        markeredgewidth=1.5,
        label="FEM",
    )
    annotate_local_slopes(h_vals, cond_fem, ax, text_offset=(20, -20), x_is_h=True)

    # 2. Stable-Corrected XFEM
    plt.loglog(
        h_vals,
        cond_sc,
        linestyle="-",
        linewidth=1.5,
        color="#D95319",
        marker="s",
        markersize=7,
        markerfacecolor="none",
        markeredgewidth=1.5,
        label="Stable-Corrected XFEM",
    )
    annotate_local_slopes(h_vals, cond_sc, ax, text_offset=(20, -20), x_is_h=True)

    # 3. Standard XFEM
    plt.loglog(
        h_vals,
        cond_std,
        linestyle="--",
        linewidth=1.5,
        color="#EDB120",
        marker="v",
        markersize=7,
        markerfacecolor="none",
        markeredgewidth=1.5,
        label="Standard XFEM",
    )
    annotate_local_slopes(h_vals, cond_std, ax, text_offset=(-25, 25), x_is_h=True)

    # --- Formatting for Publication ---
    plt.xlabel("Element Size $h$", fontsize=13)
    plt.ylabel("Scaled Condition Number $\\kappa_d$", fontsize=13)

    plt.gca().invert_xaxis()
    plt.grid(True, which="major", ls="-", color="lightgray", alpha=0.8)
    plt.grid(True, which="minor", ls="--", color="lightgray", alpha=0.4)
    plt.legend(
        fontsize=11,
        loc="lower right",
        framealpha=1.0,
        edgecolor="black",
        fancybox=False,
    )
    plt.xticks(h_vals, labels=[f"{h:.3f}" for h in h_vals])

    plt.tight_layout()
    plt.savefig("scn_corrected_annotated_3d.pdf", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    run_condition_number_study_3d()
