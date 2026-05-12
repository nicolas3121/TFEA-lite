import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.sparse.linalg import spsolve
import scipy.sparse as sps

import tfealite as tf
import tfealite.core.quadratures as qd
from tfealite.core.dofs import BRANCH_DOFS, HEAVISIDE_DOFS, DofType, BRANCH_4_DOFS
from tfealite.core.level_set import CutType
from tfealite.core.quadratures import DuffyDistance
from tfealite.elements.utils import (
    cal_B_2d_vec,
    cut_embedding_tri_iter,
    fill_element_displacement,
    partial_cut_embedding_tri_iter,
)
from tfealite.elements.XQuad4n import XQuad4n


def get_analytical_displacements(nu=0.30, plane_strain=True, xc=5.0, yc=5.0):
    kappa = 3.0 - 4.0 * nu if plane_strain else (3.0 - nu) / (1.0 + nu)

    def eval_displacements(x, y):
        x_rel, y_rel = x - xc, y - yc
        r = np.sqrt(x_rel**2 + y_rel**2)
        theta = np.arctan2(y_rel, x_rel)
        u_x = np.sqrt(r) * (
            (kappa - 0.5) * np.cos(theta / 2) - 0.5 * np.cos(3 * theta / 2)
        )
        u_y = np.sqrt(r) * (
            (kappa + 0.5) * np.sin(theta / 2) - 0.5 * np.sin(3 * theta / 2)
        )
        return u_x, u_y

    return eval_displacements


def derive_analytical_fields(xc_val=5.0, yc_val=5.0):
    """Generates exact strain and stress evaluators for the energy norm."""
    x, y = sp.symbols("x y", real=True)

    E = 1.0
    nu = 0.30
    kappa = 3.0 - 4.0 * nu

    mu = E / (2.0 * (1.0 + nu))
    lmbda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # Replaced hardcoded 0.5 with parameters
    xc, yc = x - xc_val, y - yc_val
    r = sp.sqrt(xc**2 + yc**2)
    theta = sp.atan2(yc, xc)

    u_x = sp.sqrt(r) * ((kappa - 0.5) * sp.cos(theta / 2) - 0.5 * sp.cos(3 * theta / 2))
    u_y = sp.sqrt(r) * ((kappa + 0.5) * sp.sin(theta / 2) - 0.5 * sp.sin(3 * theta / 2))

    eps_xx = sp.diff(u_x, x)
    eps_yy = sp.diff(u_y, y)
    gamma_xy = sp.diff(u_x, y) + sp.diff(u_y, x)

    sig_xx = 2 * mu * eps_xx + lmbda * (eps_xx + eps_yy)
    sig_yy = 2 * mu * eps_yy + lmbda * (eps_xx + eps_yy)
    sig_xy = mu * gamma_xy

    get_strain = sp.lambdify((x, y), (eps_xx, eps_yy, gamma_xy), modules="numpy")
    get_stress = sp.lambdify((x, y), (sig_xx, sig_yy, sig_xy), modules="numpy")

    return get_strain, get_stress


# def get_analytical_displacements(nu=0.30, plane_strain=True, xc=0.5, yc=0.5):
#     kappa = 3.0 - 4.0 * nu if plane_strain else (3.0 - nu) / (1.0 + nu)
#
#     def eval_displacements(x, y):
#         x_rel, y_rel = x - xc, y - yc
#         r = np.sqrt(x_rel**2 + y_rel**2)
#         theta = np.arctan2(y_rel, x_rel)
#         u_x = np.sqrt(r) * (
#             (kappa - 0.5) * np.cos(theta / 2) - 0.5 * np.cos(3 * theta / 2)
#         )
#         u_y = np.sqrt(r) * (
#             (kappa + 0.5) * np.sin(theta / 2) - 0.5 * np.sin(3 * theta / 2)
#         )
#         return u_x, u_y
#
#     return eval_displacements
#
#
# def derive_analytical_fields():
#     """Generates exact strain and stress evaluators for the energy norm."""
#     x, y = sp.symbols("x y", real=True)
#
#     E = 1.0
#     nu = 0.30
#     kappa = 3.0 - 4.0 * nu
#
#     mu = E / (2.0 * (1.0 + nu))
#     lmbda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
#
#     xc, yc = x - 0.5, y - 0.5
#     r = sp.sqrt(xc**2 + yc**2)
#     theta = sp.atan2(yc, xc)
#
#     u_x = sp.sqrt(r) * ((kappa - 0.5) * sp.cos(theta / 2) - 0.5 * sp.cos(3 * theta / 2))
#     u_y = sp.sqrt(r) * ((kappa + 0.5) * sp.sin(theta / 2) - 0.5 * sp.sin(3 * theta / 2))
#
#     eps_xx = sp.diff(u_x, x)
#     eps_yy = sp.diff(u_y, y)
#     gamma_xy = sp.diff(u_x, y) + sp.diff(u_y, x)
#
#     sig_xx = 2 * mu * eps_xx + lmbda * (eps_xx + eps_yy)
#     sig_yy = 2 * mu * eps_yy + lmbda * (eps_xx + eps_yy)
#     sig_xy = mu * gamma_xy
#
#     get_strain = sp.lambdify((x, y), (eps_xx, eps_yy, gamma_xy), modules="numpy")
#     get_stress = sp.lambdify((x, y), (sig_xx, sig_yy, sig_xy), modules="numpy")
#
#     return get_strain, get_stress


def calculate_element_energy_norm(element, Ue, get_exact_strain, get_exact_stress):
    exact_energy_sq = 0.0
    error_energy_sq = 0.0
    N_FN = 4

    def accumulate_error(nat_coords, w_eff, sign=None):
        nonlocal exact_energy_sq, error_energy_sq

        if sign is not None:
            N, dN_dxi = element.shape_functions(nat_coords, enforce_sign=sign)
        else:
            N, dN_dxi = element.shape_functions(nat_coords)
        x_gp = N[:, :N_FN] @ element.node_coords

        J = dN_dxi[:, :, :N_FN] @ element.node_coords
        dN_dxy = np.linalg.solve(J, dN_dxi)
        B = cal_B_2d_vec(dN_dxy)

        # Numerical Strains and Stresses
        # eps_h = np.einsum("gij,j->gi", B, Ue)

        eps_h = np.einsum("gij,j->gi", B, Ue)

        sig_h = np.einsum("ij,gj->gi", element.C, eps_h)

        # eps_h = B @ Ue
        # sig_h = element.C @ eps_h[:, :, None]

        # Exact Strains and Stresses
        eps_ex = np.column_stack(get_exact_strain(x_gp[:, 0], x_gp[:, 1]))
        sig_ex = np.column_stack(get_exact_stress(x_gp[:, 0], x_gp[:, 1]))

        # Differences
        diff_eps = eps_ex - eps_h
        diff_sig = sig_ex - sig_h

        # Energy Inner Products
        ex_energy_density = np.sum(sig_ex * eps_ex, axis=1)
        err_energy_density = np.sum(diff_sig * diff_eps, axis=1)

        # Integrate
        exact_energy_sq += np.sum(ex_energy_density * w_eff)
        error_energy_sq += np.sum(err_energy_density * w_eff)

    if not getattr(element, "h_enrich", False) and not getattr(
        element, "partial_cut", False
    ):
        rule, correction = qd.QUAD_RULES[20]
        nat_coords = rule[:, :2].T

        _, dN_dxi = element.shape_functions(nat_coords)
        J = dN_dxi[:, :, :N_FN] @ element.node_coords
        detJ = np.linalg.det(J)
        w_eff = rule[:, 2] * correction * detJ

        accumulate_error(nat_coords, w_eff)

    elif getattr(element, "h_enrich", False):
        Nc1, Nc2 = element._cal_intersections()
        rule, correction = qd.TRI_RULES[17] if element.t_enrich else qd.TRI_RULES[5]

        def integrate_sub_tri_error(Nc, nat_x_e):
            for Ni, detJi in cut_embedding_tri_iter(Nc):
                xi, eta, w = rule[:, 0], rule[:, 1], rule[:, 2]
                nat_sub_x_e = nat_x_e.T @ Ni

                sign = None

                n = np.array([1 - xi - eta, xi, eta])
                nat_coords_sub = nat_sub_x_e @ n

                _, dN_dxi_sub = element.shape_functions(
                    nat_coords_sub, enforce_sign=sign
                )
                J = dN_dxi_sub[:, :, :N_FN] @ element.node_coords
                detJ = np.linalg.det(J)

                w_eff = w * correction * detJ * detJi * 4
                accumulate_error(nat_coords_sub, w_eff, sign=sign)

        integrate_sub_tri_error(Nc1, element.NAT_1)
        integrate_sub_tri_error(Nc2, element.NAT_2)

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

        def integrate_partial_cut_error(tip, Nc, rng, nat_x_e):
            for Ni, detJi in partial_cut_embedding_tri_iter(Nc, tip, rng):
                nat_sub_x_e = nat_x_e.T @ Ni
                x_e_i = (
                    element._base_shape_functions(nat_sub_x_e)[0] @ element.node_coords
                )

                duffy = DuffyDistance(x_e_i)
                u, v = rule[:, 0], rule[:, 1]
                xi_d_2, eta_d_2, w_d_2 = duffy.transform(u, v, beta=1)

                N_map = np.array([1.0 - xi_d_2 - eta_d_2, xi_d_2, eta_d_2])
                nat_coords_sub = nat_sub_x_e @ N_map

                _, dN_dxi_sub = element.shape_functions(
                    nat_coords_sub, enforce_sign=None
                )
                J = dN_dxi_sub[:, :, :N_FN] @ element.node_coords
                detJ = np.linalg.det(J)

                w_eff = rule[:, 2] * correction * w_d_2 * detJ * detJi * 4
                accumulate_error(nat_coords_sub, w_eff, sign=None)

        integrate_partial_cut_error(tip1, Nc1, range(4), element.NAT_1)
        integrate_partial_cut_error(tip2, Nc2, range(2, 6), element.NAT_2)

    return exact_energy_sq, error_energy_sq


def test_pure_mode_1_analytical_benchmark(x_elem, y_elem, corrected):
    E_mod = 1.0
    nu = 0.3

    # --- CONVERT TO EFFECTIVE PLANE STRAIN PROPERTIES ---
    E_eff = E_mod / (1.0 - nu**2)
    nu_eff = nu / (1.0 - nu)

    # Pass the effective properties into the FEModel
    materials = [[1, {"E": E_eff, "nu": nu_eff, "rho": 7850}]]
    reals = [[1, {"t": 1}]]

    nodes, elements = tf.gen_rect_Quad4n(10.0, 10.0, x_elem, y_elem)

    model = tf.XFEModel(
        nodes,
        elements,
        materials,
        reals,
        tip_enrichment=True,
        geometrical_range=1.5,
        corrected=corrected,
    )

    p1 = np.array([0.0, 5])
    p2 = np.array([5, 5])
    model.insert_crack_segment(p1, p2, embedded=False)

    model.gen_list_dof(dof_per_node=tf.IS_2D)
    model.cal_global_matrices({"Quad4n": tf.XQuad4n}, eval_mass=False)

    calc_disp = get_analytical_displacements(nu=nu, plane_strain=True)

    tol = 1e-8
    x_coords = model.nodes[:, 1]
    y_coords = model.nodes[:, 2]

    is_boundary = (
        (np.abs(x_coords - 0.0) < tol)
        | (np.abs(x_coords - 10.0) < tol)
        | (np.abs(y_coords - 0.0) < tol)
        | (np.abs(y_coords - 10.0) < tol)
    )
    boundary_node_ids = np.where(is_boundary)[0] + 1  # 1-based indexing

    # 3. Build the exact prescribed displacement vector (U_dir)
    U_dir = np.zeros(model.Kg.shape[0])
    fix_dofs = []

    boundary_x_val = model.nodes[boundary_node_ids - 1, 1]
    boundary_y_val = model.nodes[boundary_node_ids - 1, 2]

    u_x_exact, u_y_exact = calc_disp(boundary_x_val, boundary_y_val)

    dof_x = model.list_dof.get_elem_dof_numbers(boundary_node_ids, DofType.UX).ravel()
    dof_y = model.list_dof.get_elem_dof_numbers(boundary_node_ids, DofType.UY).ravel()

    U_dir[dof_x] = u_x_exact
    U_dir[dof_y] = u_y_exact
    fix_dofs.extend(dof_x)
    fix_dofs.extend(dof_y)

    blending_nodes = (model.in_range == 0) & (
        model.list_dof.list_dof & BRANCH_DOFS != 0
    )
    extra_fix_dofs = model.list_dof.get_elem_dof_numbers_flat(
        np.where(blending_nodes)[0][:2] + 1, BRANCH_4_DOFS
    )

    fix_dofs.extend(list(extra_fix_dofs))

    # bc.my_gen_dirichlet_bc(model, sel_condition, extra_fix_dofs)

    fix_dofs = np.array(sorted(set(fix_dofs)), dtype=int)
    model.gen_P(fix_dofs)

    F_ext = np.zeros(model.Kg.shape[0])

    # 1. Lift (Algebraic Elimination) in Physical Space FIRST
    # Calculate physical reaction forces: (F_ext - K @ U_dir)
    F_physical_reduced = F_ext - model.Kg @ U_dir

    # 2. Orthogonalize the global system (Babuška Shift)
    K_ortho = model.ortho_T.T @ model.Kg @ model.ortho_T
    K_ortho = (K_ortho + K_ortho.T) / 2
    F_ortho_reduced = model.ortho_T.T @ F_physical_reduced
    print("   - K_ortho evaluated.")

    # 3. Apply the Boolean projection (P) to strip constrained DOFs
    K_reduced = model.P.T @ K_ortho @ model.P
    F_final = model.P.T @ F_ortho_reduced
    print("   - K_reduced (Lifting) evaluated.")

    # 4. Diagonal Scaling (Jacobi Preconditioning)
    D = K_reduced.diagonal()
    D_inv_sqrt = sps.diags(1.0 / np.sqrt(D))

    Kg_scaled = D_inv_sqrt @ K_reduced @ D_inv_sqrt
    Fg_scaled = D_inv_sqrt @ F_final
    print("   - Diagonal scaling applied.")

    # 5. Solve the optimally conditioned system
    print("   - Start solving for U = inv(K)F ...")
    Ug_scaled = spsolve(Kg_scaled, Fg_scaled)

    # 6. Reverse the transformations
    Ug_reduced = D_inv_sqrt @ Ug_scaled  # Unscale
    Ug_tilde = model.P @ Ug_reduced + U_dir  # Add Dirichlet boundaries back
    model.Ug = model.ortho_T @ Ug_tilde

    # mult = 0.1
    # # model.Ug = np.zeros(len(model.list_dof))
    # mesh1 = my_build_Quad4n(model, mult=mult).cast_to_unstructured_grid()
    # ghosts = np.argwhere(mesh1["is_enriched"] > 0)
    # mesh1.remove_cells(ghosts, inplace=True)
    # mesh2 = build_XQuad4n(model, mult=mult)
    # blocks = pv.MultiBlock([mesh1, mesh2])
    # pl = pv.Plotter()
    # pl.add_mesh(
    #     blocks,
    #     color="lightblue",
    #     # scalars="von_mises",  # The exact string key in your point_data dict
    #     # cmap="turbo",  # A great colormap for stress fields (or use "jet", "viridis")
    #     show_edges=True,  # Shows the mesh grid (including your sub-triangulations!)
    # )
    # pl.view_xy()
    # # pl.enable_parallel_projection()
    # pl.show()

    # --- Global Energy Norm Calculation ---
    get_exact_strain, get_exact_stress = derive_analytical_fields()

    total_exact_energy_sq = 0.0
    total_error_energy_sq = 0.0

    # Iterate through all elements to compute the global error

    for i, element_data in enumerate(model.elements):
        elem_id, _, mat_id, real_id, elem_nodes = element_data
        elem_nodes = np.asarray(elem_nodes)

        _, cut_type, _ = model.cut_info.get(elem_id, (None, CutType.NONE, None))

        elem_dofs = model.list_dof.get_elem_dofs(elem_nodes)
        local_dofs_per_node = np.bitwise_or.reduce(elem_dofs)
        h_enrich = bool(local_dofs_per_node & HEAVISIDE_DOFS)
        t_enrich = bool(local_dofs_per_node & BRANCH_DOFS)
        partial_cut = cut_type == CutType.PARTIAL

        # 3. Fetch Level Set Parameters (Only if the element is actually enriched)
        if h_enrich or t_enrich:
            most_enriched_node = elem_nodes[
                np.argmax(np.bitwise_and(elem_dofs, BRANCH_DOFS | HEAVISIDE_DOFS))
            ]
            ls_idx = model.ls[most_enriched_node - 1]

            tip = 0
            if t_enrich:
                tip = model.tip[
                    elem_nodes[np.argmax(np.bitwise_and(elem_dofs, BRANCH_DOFS))] - 1
                ]
            phi_n, phi_t = model.level_sets[ls_idx].get(elem_nodes, tip)
        else:
            # Standard element far from the crack
            phi_n, phi_t = None, None

        # 4. Formulate the Element
        elem_vertices = model.nodes[elem_nodes - 1, 1:3]
        material = model.materials[mat_id - 1][1]
        real = model.reals[real_id - 1][1]
        in_range = (
            model.in_range[elem_nodes - 1]
            if model.corrected
            else np.ones(len(elem_nodes))
        )

        elem = XQuad4n(
            elem_vertices,
            material,
            real,
            phi_n,
            phi_t,
            h_enrich,
            t_enrich,
            partial_cut,
            in_range,
        )

        Ue = fill_element_displacement(elem_nodes, model.list_dof, model.Ug)

        ex_sq, err_sq = calculate_element_energy_norm(
            elem, Ue, get_exact_strain, get_exact_stress
        )

        total_exact_energy_sq += ex_sq
        total_error_energy_sq += err_sq

    relative_error = np.sqrt(total_error_energy_sq) / np.sqrt(total_exact_energy_sq)
    return relative_error


def run_convergence_study():
    mesh_sizes = [9, 19, 31, 41, 55, 77, 101]
    h_vals = [1.0 / n for n in mesh_sizes]

    errors_uncorrected = []
    errors_corrected = []

    print("Running Convergence Study...")
    print(
        f"{'Elements (NxN)':<15} | {'h':<10} | {'Uncorrected Error':<20} | {'Corrected Error'}"
    )
    print("-" * 72)

    for n, h in zip(mesh_sizes, h_vals):
        # Run solver with Corrected XFEM turned OFF (Standard/SGFEM)
        err_uncorr = test_pure_mode_1_analytical_benchmark(n, n, False)

        # Run solver with Corrected XFEM turned ON (Stable-Corrected XFEM)
        err_corr = test_pure_mode_1_analytical_benchmark(n, n, True)

        errors_uncorrected.append(err_uncorr)
        errors_corrected.append(err_corr)

        print(
            f"{n:<15} | {h:<10.4f} | {err_uncorr * 100.0:>16.4f}% | {err_corr * 100.0:>14.4f}%"
        )

    # --- Calculate Slopes ---
    log_h = np.log(h_vals)
    slope_uncorr, _ = np.polyfit(log_h, np.log(errors_uncorrected), 1)
    slope_corr, _ = np.polyfit(log_h, np.log(errors_corrected), 1)

    print("-" * 72)
    print(f"Uncorrected Rate (Slope): {slope_uncorr:.4f} (Expected: ~0.5)")
    print(f"Corrected Rate (Slope):   {slope_corr:.4f} (Expected: ~1.0)")

    # --- Plotting ---
    plt.figure(figsize=(9, 7))

    # 1. Plot Measured Data
    plt.loglog(
        h_vals,
        errors_uncorrected,
        marker="s",
        markersize=8,
        linestyle="-",
        linewidth=2,
        color="red",
        label=f"Standard XFEM (Slope: {slope_uncorr:.2f})",
    )

    plt.loglog(
        h_vals,
        errors_corrected,
        marker="o",
        markersize=8,
        linestyle="-",
        linewidth=2,
        color="blue",
        label=f"Stable-Corrected XFEM (Slope: {slope_corr:.2f})",
    )

    # 2. Plot Theoretical Reference Lines
    # O(h^1.0) anchored to the first point of the corrected curve
    C_1 = errors_corrected[0] / (h_vals[0] ** 1.0)
    theo_1 = [C_1 * (h**1.0) for h in h_vals]
    plt.loglog(
        h_vals,
        theo_1,
        linestyle="--",
        color="black",
        label=r"Optimal $\mathcal{O}(h^{1.0})$",
    )

    # O(h^0.5) anchored to the first point of the uncorrected curve
    C_05 = errors_uncorrected[0] / (h_vals[0] ** 0.5)
    theo_05 = [C_05 * (h**0.5) for h in h_vals]
    plt.loglog(
        h_vals,
        theo_05,
        linestyle=":",
        color="black",
        label=r"Sub-optimal $\mathcal{O}(h^{0.5})$",
    )

    # 3. Formatting for Publication
    plt.xlabel(r"Element Size $h$", fontsize=14)
    plt.ylabel(r"Relative Energy Norm Error", fontsize=14)
    # plt.title("Convergence Rate: Standard vs Stable-Corrected XFEM", fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)

    # Invert X axis so the mesh gets "finer" as you read left to right
    plt.gca().invert_xaxis()

    plt.tight_layout()
    plt.savefig("convergence_test.pdf")
    plt.show()


if __name__ == "__main__":
    run_convergence_study()
