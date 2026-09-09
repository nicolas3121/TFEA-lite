import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from geomdl import knotvector
from scipy.interpolate import BSpline

import tfealite as tf
from tfealite.core.dofs import DofType
from tfealite.core.sif import DisplacementCorrelationMethodSIF as DCMSIF
from tfealite.visualization.build_mesh import (
    build_XQuad4n,
    my_build_Quad4n,
)


def calculate_growth_direction(K_I, K_II):
    # Added **2 to K_II in both the numerator and denominator
    numerator = 3 * K_II**2 + np.sqrt(K_I**4 + 8 * K_I**2 * K_II**2)
    denominator = K_I**2 + 9 * K_II**2

    # Clip ensures floating point errors don't push the ratio to 1.00000001
    argument = np.clip(numerator / denominator, -1.0, 1.0)

    theta = np.sign(-K_II) * np.acos(argument)
    return theta


def solve_model_and_sweep_ra(
    x_elem,
    y_elem,
    corrected,
    rho_bar_list,
    gen=tf.gen_rect_Quad4n,
    geometrical_range=1.0,
    crack_angle=0.0,
    plot=False,
):
    """
    Solves the FEA model ONCE, then loops over an array of rho_bar values
    to extract the SIF using the DCM method at varying distances.
    """
    E_mod = 1
    nu = 0.3
    kappa = (3 - nu) / (1 + nu)
    G_mod = (E_mod) / (2 * (1 + nu))
    a = 0.115 * 10
    W = 0.5 * 10
    H = W
    angle = crack_angle

    # Re-calculate h exactly as done in mesh generation
    h = 10 / x_elem

    nodes, elements = tf.gen_rect_Quad4n(2 * W, 2 * H, x_elem, y_elem)
    materials = [[1, {"E": E_mod, "nu": nu, "rho": 7850}]]
    reals = [[1, {"t": 1}]]
    model = tf.XFEModel(
        nodes,
        elements,
        materials,
        reals,
        tip_enrichment=True,
        geometrical_range=geometrical_range,
        corrected=corrected,
    )
    p1 = np.array([W - a * np.cos(angle), H - a * np.sin(angle)])
    p2 = np.array([W + a * np.cos(angle), H + a * np.sin(angle)])

    control_points = np.linspace(p1, p2, 12).tolist()
    n = len(control_points)
    k = 2
    knots = knotvector.generate(k, n)
    bspline = BSpline(knots, np.array(control_points), k)

    model.insert_crack_spline(bspline, embedded=True, h=h, snapping_tolerance=0.08)

    model.gen_list_dof(dof_per_node=tf.IS_2D)
    elem_dict = {"Quad4n": tf.XQuad4n, "Tri3n": tf.XTri3n}
    model.cal_global_matrices(elem_dict, eval_mass=False)

    is_bottom = model.nodes[:, 2] < 1e-8
    is_left = model.nodes[:, 1] < 1e-8
    is_right = np.abs(model.nodes[:, 1] - 2 * W) < 1e-8

    # Get specific node IDs
    bl_node = 1 + np.where(is_bottom & is_left)[0][0]
    br_node = 1 + np.where(is_bottom & is_right)[0][0]

    fix_dofs = []
    # Pin bottom-left in X and Y
    fix_dofs.append(model.list_dof[(bl_node, DofType.UX)])
    fix_dofs.append(model.list_dof[(bl_node, DofType.UY)])
    # Roller bottom-right in Y (stops rotation but allows Poisson contraction)
    fix_dofs.append(model.list_dof[(br_node, DofType.UY)])

    model.gen_P(np.array(sorted(set(fix_dofs)), dtype=int))

    def sel_condition1(x, y, z):
        return y - 2 * H

    def force_expression1(x, y, z):
        return 0.0, 1, 0.0

    model.gen_surface_tractions(sel_condition1, force_expression1, tf.Quad4n, 2)

    def sel_condition2(x, y, z):
        return y - 0

    def force_expression2(x, y, z):
        return 0.0, -1, 0.0

    model.gen_surface_tractions(
        sel_condition2, force_expression2, tf.Quad4n, 2, reset=False
    )

    # ==========================================
    # SOLVE FEA ONCE
    # ==========================================
    model.solve_static()

    if plot:
        displacement_mult = 1e-2
        mesh1 = my_build_Quad4n(
            model, mult=displacement_mult
        ).cast_to_unstructured_grid()
        ghosts = np.argwhere(mesh1["is_enriched"] > 0)
        mesh1.remove_cells(ghosts, inplace=True)

        mesh2 = build_XQuad4n(model, mult=displacement_mult)

        blocks = pv.MultiBlock([mesh1, mesh2])
        pl = pv.Plotter()

        vm_1 = mesh1.point_data.get("von_mises", np.zeros(mesh1.n_points))
        vm_2 = mesh2.point_data.get("von_mises", np.zeros(mesh2.n_points))
        all_vm = np.concatenate([vm_1, vm_2])
        v_max = np.percentile(all_vm, 99.7) if len(all_vm) > 0 else 1.0

        pl.add_mesh(
            blocks,
            scalars="von_mises",
            cmap="turbo",
            show_edges=True,
            clim=[0, v_max],
            scalar_bar_args={"title": "Von Mises Stress"},
        )

        pl.view_xy()
        pl.enable_anti_aliasing()
        pl.show()

    # ==========================================
    # SWEEP OVER R_A VALUES
    # ==========================================
    # K_I_analytical = 1 * np.sqrt(np.pi * a) * np.sqrt(1 / np.cos((np.pi * a) / (2 * W)))
    # K_I_analytical = 1 * np.sqrt(np.pi * a) * np.sin(angle) ** 2
    # K_II_analytical = 1 * np.sqrt(np.pi * a) * np.sin(angle) * np.cos(angle)
    K_I_analytical = 1 * np.sqrt(np.pi * a) * (np.cos(angle) ** 2)
    K_II_analytical = 1 * np.sqrt(np.pi * a) * np.sin(angle) * np.cos(angle)
    theta_analytical = calculate_growth_direction(K_I_analytical, K_II_analytical)

    errors_K1 = []
    errors_K2 = []
    errors_theta = []

    for rho_bar in rho_bar_list:
        r_a = rho_bar * h

        # Defining N_sampl=10 points starting at r_a with spacing Delta = 0.1*h
        distances = r_a + 0.095 * h * np.arange(10)

        dcm = DCMSIF(kappa, G_mod, distances, None)
        K_I, K_II, T_matrix = dcm.cal_sif(
            model.level_sets[0], model, model.cut_info, 1.0
        )
        theta = calculate_growth_direction(K_I, K_II)

        err_K1 = np.abs(K_I - K_I_analytical) / np.abs(K_I_analytical)
        err_K2 = np.abs(K_II - K_II_analytical) / np.abs(K_II_analytical)
        err_theta = np.abs(theta - theta_analytical) / np.abs(theta_analytical)

        errors_K1.append(err_K1)
        errors_K2.append(err_K2)
        errors_theta.append(err_theta)

    return np.array(errors_K1), np.array(errors_K2), np.array(errors_theta)


def run_ra_sweep_study(geometrical_range, crack_angle):
    # Fix the mesh size for the parameter sweep
    x_elem = 121
    y_elem = x_elem
    h = 10 / x_elem

    # Define a range for the extraction parameter rho_bar = r_a / h
    # Sweeping from 0.5 to 7.0 (typical path-independence investigation range)
    rho_bar_vals = np.linspace(0.5, 7.0, 30)

    print(f"Fixed Mesh: {x_elem}x{y_elem} elements")
    print("-" * 65)

    # Sweep Uncorrected (SO-XFEM)
    print("Evaluating Uncorrected SO-XFEM model...")
    errors_uncorr_K1, errors_uncorr_K2, errors_uncorr_theta = solve_model_and_sweep_ra(
        x_elem,
        y_elem,
        False,
        rho_bar_vals,
        geometrical_range=geometrical_range * h,
        crack_angle=crack_angle,
        plot=False,
    )

    # Sweep Corrected (SCO-XFEM)
    print("Evaluating Corrected SCO-XFEM model...")
    errors_corr_K1, errors_corr_K2, errors_corr_theta = solve_model_and_sweep_ra(
        x_elem,
        y_elem,
        True,
        rho_bar_vals,
        geometrical_range=geometrical_range * h,
        crack_angle=crack_angle,
        plot=False,
    )

    # Convert errors to percentages for plotting
    errors_uncorr_K1 *= 100.0
    errors_corr_K1 *= 100.0
    errors_uncorr_K2 *= 100.0
    errors_corr_K2 *= 100.0
    errors_corr_theta *= 100.0
    errors_uncorr_theta *= 100.0

    # ==========================================
    # Plot 1: K_I Relative Error vs rho_bar
    # ==========================================
    plt.figure(figsize=(4, 3))
    ax1 = plt.gca()

    ax1.plot(
        rho_bar_vals,
        errors_uncorr_K1,
        marker="s",
        markersize=6,
        linestyle="-",
        linewidth=2,
        color="red",
        markerfacecolor="none",
        label="SO-XFEM",
    )

    ax1.plot(
        rho_bar_vals,
        errors_corr_K1,
        marker="o",
        markersize=6,
        linestyle="-",
        linewidth=2,
        color="blue",
        markerfacecolor="none",
        label="SCO-XFEM",
    )

    ax1.set_xlabel(r"Extraction parameter $\bar{\rho}$", fontsize=13)
    ax1.set_ylabel(r"Relative $K_I$ Error (%)", fontsize=13)
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # Standard linear limits usually 0-20% for robustness plots
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=11, framealpha=1.0, edgecolor="black")

    plt.tight_layout()
    plt.savefig(
        f"ra_sweep_angled_K1_error_{geometrical_range}_{int(np.degrees(crack_angle))}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()

    # ==========================================
    # Plot 2: K_II Absolute Error vs rho_bar
    # ==========================================
    plt.figure(figsize=(4, 3))
    ax3 = plt.gca()

    ax3.plot(
        rho_bar_vals,
        errors_uncorr_theta,
        marker="s",
        markersize=6,
        linestyle="-",
        linewidth=2,
        color="red",
        markerfacecolor="none",
        label="SO-XFEM",
    )

    ax3.plot(
        rho_bar_vals,
        errors_corr_theta,
        marker="o",
        markersize=6,
        linestyle="-",
        linewidth=2,
        color="blue",
        markerfacecolor="none",
        label="SCO-XFEM",
    )

    ax3.set_xlabel(r"Extraction parameter $\bar{\rho}$", fontsize=13)
    ax3.set_ylabel(r"Relative $\theta$ Error (%)", fontsize=13)
    ax3.grid(True, which="both", ls="--", alpha=0.5)
    ax3.set_ylim(bottom=0)
    ax3.legend(fontsize=11, framealpha=1.0, edgecolor="black")

    plt.tight_layout()
    plt.savefig(
        f"ra_sweep_angled_theta_error_{geometrical_range}_{int(np.degrees(crack_angle))}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()

    plt.figure(figsize=(4, 3))
    ax2 = plt.gca()

    ax2.plot(
        rho_bar_vals,
        errors_uncorr_K2,
        marker="s",
        markersize=6,
        linestyle="-",
        linewidth=2,
        color="red",
        markerfacecolor="none",
        label="SO-XFEM",
    )

    ax2.plot(
        rho_bar_vals,
        errors_corr_K2,
        marker="o",
        markersize=6,
        linestyle="-",
        linewidth=2,
        color="blue",
        markerfacecolor="none",
        label="SCO-XFEM",
    )

    ax2.set_xlabel(r"Extraction parameter $\bar{\rho} = r_a / h_{min}$", fontsize=13)
    ax2.set_ylabel(r"Relative $K_{II}$ Error (%)", fontsize=13)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=11, framealpha=1.0, edgecolor="black")

    plt.tight_layout()
    plt.savefig(
        f"ra_sweep_angled_K2_error_{geometrical_range}_{int(np.degrees(crack_angle))}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()


if __name__ == "__main__":
    # Run the full r_a parameter sweep study
    for i in range(1, 7):
        run_ra_sweep_study(4.5, np.radians(5) * i)
