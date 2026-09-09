import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
import sympy as sp
from scipy.sparse.linalg import spsolve

import tfealite as tf
import tfealite.core.quadratures as qd
from tfealite.core.dofs import BRANCH_DOFS, HEAVISIDE_DOFS, DofType
from tfealite.core.level_set import CutType
from tfealite.core.quadratures import DuffyDistance
from tfealite.elements.utils import (
    cal_B_2d_vec,
    cut_embedding_tri_iter,
    fill_element_displacement,
    partial_cut_embedding_tri_iter,
)


def annotate_local_slopes(x_vals, y_vals, ax, text_offset, x_is_h=False, color="black"):
    """
    Calculates the local log-log slope between consecutive points and adds
    an arrow annotation to the midpoint of the line segment.
    """
    for i in range(len(x_vals) - 1):
        x1, x2 = x_vals[i], x_vals[i + 1]
        y1, y2 = y_vals[i], y_vals[i + 1]

        # For convergence: rate p where Error ~ h^p
        # If x_is_h=True and x decreases (x1 > x2), we want log(y1/y2) / log(x1/x2)
        if x_is_h:
            rate = np.log(y1 / y2) / np.log(x1 / x2)
        else:
            rate = np.log(y2 / y1) / np.log(x2 / x1)

        # Calculate midpoint in log-space for accurate placement
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
            color=color,
            arrowprops={
                "arrowstyle": "->", "color": color, "shrinkA": 0, "shrinkB": 5, "alpha": 0.7
            },
        )


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


def calculate_element_energy_norm_tri(element, Ue, get_exact_strain, get_exact_stress):
    exact_energy_sq = 0.0
    error_energy_sq = 0.0
    N_FN = 3

    def accumulate_error(nat_coords, w_eff, sign=None):
        nonlocal exact_energy_sq, error_energy_sq

        N, dN_dxi = element.shape_functions(nat_coords)
        x_gp = N[:, :N_FN] @ element.node_coords

        J = dN_dxi[:, :, :N_FN] @ element.node_coords
        dN_dxy = np.linalg.solve(J, dN_dxi)
        B = cal_B_2d_vec(dN_dxy)

        eps_h = np.einsum("gij,j->gi", B, Ue)

        sig_h = np.einsum("ij,gj->gi", element.C, eps_h)

        eps_ex = np.column_stack(get_exact_strain(x_gp[:, 0], x_gp[:, 1]))
        sig_ex = np.column_stack(get_exact_stress(x_gp[:, 0], x_gp[:, 1]))

        diff_eps = eps_ex - eps_h
        diff_sig = sig_ex - sig_h

        ex_energy_density = np.sum(sig_ex * eps_ex, axis=1)
        err_energy_density = np.sum(diff_sig * diff_eps, axis=1)

        exact_energy_sq += np.sum(ex_energy_density * w_eff)
        error_energy_sq += np.sum(err_energy_density * w_eff)

    if not getattr(element, "h_enrich", False) and not getattr(
        element, "partial_cut", False
    ):
        rule, correction = qd.TRI_RULES[19]
        nat_coords = rule[:, :2].T

        _, dN_dxi = element.shape_functions(nat_coords)
        J = dN_dxi[:, :, :N_FN] @ element.node_coords
        detJ = np.linalg.det(J)
        w_eff = rule[:, 2] * correction * detJ

        accumulate_error(nat_coords, w_eff)

    elif getattr(element, "h_enrich", False):
        Nc1 = element._cal_intersections()
        rule, correction = qd.TRI_RULES[19] if element.t_enrich else qd.TRI_RULES[19]

        def integrate_sub_tri_error(Nc, nat_x_e):
            for Ni, detJi in cut_embedding_tri_iter(Nc):
                xi, eta, w = rule[:, 0], rule[:, 1], rule[:, 2]
                nat_sub_x_e = nat_x_e.T @ Ni

                sign = None

                n = np.array([1 - xi - eta, xi, eta])
                nat_coords_sub = nat_sub_x_e @ n

                _, dN_dxi_sub = element.shape_functions(nat_coords_sub)
                J = dN_dxi_sub[:, :, :N_FN] @ element.node_coords
                detJ = np.linalg.det(J)

                w_eff = w * correction * detJ * detJi
                accumulate_error(nat_coords_sub, w_eff, sign=sign)

        integrate_sub_tri_error(Nc1, element.NAT_COORDS)

    elif getattr(element, "partial_cut", False):
        Nc1 = element._cal_intersections()
        rule, correction = qd.QUAD_RULES[20]
        rule = rule.copy()
        rule[:, 0:2] = (1 + rule[:, 0:2]) / 2
        rule[:, 2] /= 4

        tip = np.linalg.solve(
            np.array([element.phi_t, element.phi_n, [1, 1, 1]]), np.array([0, 0, 1])
        )

        def integrate_partial_cut_error(tip, Nc, rng, nat_x_e):
            for Ni, detJi in partial_cut_embedding_tri_iter(Nc, tip, rng):
                nat_sub_x_e = nat_x_e.T @ Ni
                x_e_i = (
                    element._base_shape_functions(nat_sub_x_e)[0] @ element.node_coords
                )

                duffy = DuffyDistance(x_e_i)
                u, v = rule[:, 0], rule[:, 1]
                xi_d_2, eta_d_2, w_d_2 = duffy.transform(u, v, beta=2)

                N_map = np.array([1.0 - xi_d_2 - eta_d_2, xi_d_2, eta_d_2])
                nat_coords_sub = nat_sub_x_e @ N_map

                _, dN_dxi_sub = element.shape_functions(nat_coords_sub)
                J = dN_dxi_sub[:, :, :N_FN] @ element.node_coords
                detJ = np.linalg.det(J)

                w_eff = rule[:, 2] * correction * w_d_2 * detJ * detJi
                accumulate_error(nat_coords_sub, w_eff, sign=None)

        integrate_partial_cut_error(tip, Nc1, range(6), element.NAT_COORDS)

    return exact_energy_sq, error_energy_sq


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
        rule, correction = qd.TRI_RULES[19] if element.t_enrich else qd.TRI_RULES[19]

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
                xi_d_2, eta_d_2, w_d_2 = duffy.transform(u, v, beta=2)

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


def test_pure_mode_1_analytical_benchmark(
    x_elem,
    y_elem,
    corrected,
    gen=tf.gen_rect_Quad4n,
    error_fn=calculate_element_energy_norm,
    geometrical_range=1.2,
):
    E_mod = 1.0
    nu = 0.3

    # --- CONVERT TO EFFECTIVE PLANE STRAIN PROPERTIES ---
    E_eff = E_mod / (1.0 - nu**2)
    nu_eff = nu / (1.0 - nu)

    # Pass the effective properties into the FEModel
    materials = [[1, {"E": E_eff, "nu": nu_eff, "rho": 7850}]]
    reals = [[1, {"t": 1}]]

    nodes, elements = gen(10.0, 10.0, x_elem, y_elem)

    model = tf.XFEModel(
        nodes,
        elements,
        materials,
        reals,
        tip_enrichment=True,
        geometrical_range=geometrical_range,
        corrected=corrected,
    )

    p1 = np.array([0.0, 5])
    p2 = np.array([5, 5])
    model.insert_crack_segment(p1, p2, embedded=False)

    model.gen_list_dof(dof_per_node=tf.IS_2D)
    elem_dict = {"Quad4n": tf.XQuad4n, "Tri3n": tf.XTri3n}
    model.cal_global_matrices(elem_dict, eval_mass=False)

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

    get_exact_strain, get_exact_stress = derive_analytical_fields()

    total_exact_energy_sq = 0.0
    total_error_energy_sq = 0.0

    # Iterate through all elements to compute the global error

    for i, element_data in enumerate(model.elements):
        elem_id, elem_name, mat_id, real_id, elem_nodes = element_data
        elem_fn = elem_dict[elem_name]
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

        elem = elem_fn(
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

        ex_sq, err_sq = error_fn(elem, Ue, get_exact_strain, get_exact_stress)

        total_exact_energy_sq += ex_sq
        total_error_energy_sq += err_sq

    relative_error = np.sqrt(total_error_energy_sq) / np.sqrt(total_exact_energy_sq)
    return relative_error


def run_convergence_study():
    mesh_sizes = [9, 19, 31, 41, 55, 77, 101, 161]
    # mesh_sizes = [9, 19, 31, 41, 55]
    h_vals = [1.0 / n for n in mesh_sizes]

    errors_uncorrected = []
    errors_corrected = []
    errors_corrected_tri = []

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

        # Run solver with tri elements
        err_corr_tri = test_pure_mode_1_analytical_benchmark(
            n,
            n,
            False,
            calculate_element_energy_norm_tri,
            0.6,
        )

        errors_uncorrected.append(err_uncorr)
        errors_corrected.append(err_corr)
        errors_corrected_tri.append(err_corr_tri)

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

    plt.figure(figsize=(9, 7))
    ax = plt.gca()

    # 1. Standard XFEM (Red)
    plt.loglog(
        h_vals,
        errors_uncorrected,
        marker="s",
        markersize=8,
        linestyle="-",
        linewidth=2,
        color="red",
        markerfacecolor="none",
        label=f"Stable-Orthogonalized XFEM (Avg Slope: {slope_uncorr:.2f})",
    )
    # Annotate local slopes for Red line (Offset upwards to stay clear of the blue line)
    annotate_local_slopes(
        h_vals, errors_uncorrected, ax, text_offset=(-20, 30), x_is_h=True, color="red"
    )

    # 2. Stable-Corrected XFEM (Blue)
    plt.loglog(
        h_vals,
        errors_corrected,
        marker="o",
        markersize=8,
        linestyle="-",
        linewidth=2,
        color="blue",
        markerfacecolor="none",
        label=f"Stable-Corrected-Orthogonalized XFEM (Avg Slope: {slope_corr:.2f})",
    )
    annotate_local_slopes(
        h_vals, errors_corrected, ax, text_offset=(20, -30), x_is_h=True, color="black"
    )

    plt.loglog(
        h_vals,
        errors_corrected_tri,
        marker="o",
        markersize=8,
        linestyle="-",
        linewidth=2,
        color="blue",
        markerfacecolor="none",
        label=f"Tri Stable-Corrected-Orthogonalized XFEM (Avg Slope: {slope_corr:.2f})",
    )
    annotate_local_slopes(
        h_vals,
        errors_corrected_tri,
        ax,
        text_offset=(20, -30),
        x_is_h=True,
        color="black",
    )

    # 3. Theoretical Reference Lines
    # O(h^1.0)
    C_1 = errors_corrected[0] / (h_vals[0] ** 1.0)
    theo_1 = [C_1 * (h**1.0) for h in h_vals]
    plt.loglog(
        h_vals,
        theo_1,
        linestyle="--",
        color="black",
        alpha=0.6,
        label=r"Optimal $\mathcal{O}(h^{1.0})$",
    )

    # O(h^0.5)
    C_05 = errors_uncorrected[0] / (h_vals[0] ** 0.5)
    theo_05 = [C_05 * (h**0.5) for h in h_vals]
    plt.loglog(
        h_vals,
        theo_05,
        linestyle=":",
        color="black",
        alpha=0.6,
        label=r"Sub-optimal $\mathcal{O}(h^{0.5})$",
    )

    # --- Formatting for Publication ---
    plt.xlabel(r"Element Size $h$", fontsize=14)
    plt.ylabel(r"Relative Energy Norm Error", fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # Invert X axis so the mesh gets "finer" (smaller h) as you read left to right
    plt.gca().invert_xaxis()

    # Legend at lower right to avoid overlapping data
    plt.legend(
        fontsize=11,
        loc="upper right",
        framealpha=1.0,
        edgecolor="black",
        fancybox=False,
    )

    # Optional: Set x-ticks to actual h values for clarity
    plt.xticks(h_vals, labels=[f"{h:.3f}" for h in h_vals])

    plt.tight_layout()
    plt.savefig("convergence_annotated.pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    run_convergence_study()
