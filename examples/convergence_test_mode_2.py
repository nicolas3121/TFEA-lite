import numpy as np
import pyvista as pv
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
from tfealite.visualization.build_mesh import (
    build_XQuad4n,
    my_build_Quad4n,
)


def annotate_local_slopes(x_vals, y_vals, ax, text_offset, x_is_h=False, color="black"):
    for i in range(len(x_vals) - 1):
        x1, x2 = x_vals[i], x_vals[i + 1]
        y1, y2 = y_vals[i], y_vals[i + 1]

        if x_is_h:
            rate = np.log(y1 / y2) / np.log(x1 / x2)
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
            color=color,
            arrowprops={
                "arrowstyle": "->", "color": color, "shrinkA": 0, "shrinkB": 5, "alpha": 0.7
            },
        )


def get_analytical_displacements(nu=0.30, plane_strain=True, xc=5.0, yc=5.0):
    """Generates exact displacement field for pure Mode II (Shear)."""
    kappa = 3.0 - 4.0 * nu if plane_strain else (3.0 - nu) / (1.0 + nu)

    def eval_displacements(x, y):
        x_rel, y_rel = x - xc, y - yc
        r = np.sqrt(x_rel**2 + y_rel**2)
        theta = np.arctan2(y_rel, x_rel)

        # Exact Mode II analytical displacement components
        u_x = np.sqrt(r) * (
            (kappa + 1.5) * np.sin(theta / 2) + 0.5 * np.sin(3 * theta / 2)
        )
        u_y = np.sqrt(r) * (
            -(kappa - 1.5) * np.cos(theta / 2) - 0.5 * np.cos(3 * theta / 2)
        )
        return u_x, u_y

    return eval_displacements


def derive_analytical_fields(xc_val=5.0, yc_val=5.0):
    """Generates exact strain and stress fields for pure Mode II via SymPy."""
    x, y = sp.symbols("x y", real=True)

    E = 1.0
    nu = 0.30
    kappa = 3.0 - 4.0 * nu

    mu = E / (2.0 * (1.0 + nu))
    lmbda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

    xc, yc = x - xc_val, y - yc_val
    r = sp.sqrt(xc**2 + yc**2)
    theta = sp.atan2(yc, xc)

    # Pure Mode II Williams Asymptotic Equations
    u_x = sp.sqrt(r) * ((kappa + 1.5) * sp.sin(theta / 2) + 0.5 * sp.sin(3 * theta / 2))
    u_y = sp.sqrt(r) * (
        -(kappa - 1.5) * sp.cos(theta / 2) - 0.5 * sp.cos(3 * theta / 2)
    )

    eps_xx = sp.diff(u_x, x)
    eps_yy = sp.diff(u_y, y)
    gamma_xy = sp.diff(u_x, y) + sp.diff(u_y, x)

    sig_xx = 2 * mu * eps_xx + lmbda * (eps_xx + eps_yy)
    sig_yy = 2 * mu * eps_yy + lmbda * (eps_xx + eps_yy)
    sig_xy = mu * gamma_xy

    get_strain = sp.lambdify((x, y), (eps_xx, eps_yy, gamma_xy), modules="numpy")
    get_stress = sp.lambdify((x, y), (sig_xx, sig_yy, sig_xy), modules="numpy")

    return get_strain, get_stress


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


def test_pure_mode_2_analytical_benchmark(
    x_elem,
    y_elem,
    corrected,
    gen=tf.gen_rect_Quad4n,
    error_fn=calculate_element_energy_norm,
    geometrical_range=1.2,
    show_plot=False,
):
    E_mod = 1.0
    nu = 0.3

    E_eff = E_mod / (1.0 - nu**2)
    nu_eff = nu / (1.0 - nu)

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

    # Apply Mode II exact surface tractions
    _, get_exact_stress = derive_analytical_fields()

    def sel_right(x, y, z=0):
        return x - 10.0

    def sel_top(x, y, z=0):
        return y - 10.0

    def sel_left(x, y, z=0):
        return x - 0.0

    def sel_bot(x, y, z=0):
        return y - 0.0

    def trac_right(x, y, z=0):
        sig_xx, sig_yy, sig_xy = get_exact_stress(x, y)
        return sig_xx, sig_xy, np.zeros_like(x)

    def trac_top(x, y, z=0):
        sig_xx, sig_yy, sig_xy = get_exact_stress(x, y)
        return sig_xy, sig_yy, np.zeros_like(x)

    def trac_left(x, y, z=0):
        sig_xx, sig_yy, sig_xy = get_exact_stress(x, y)
        return -sig_xx, -sig_xy, np.zeros_like(x)

    def trac_bot(x, y, z=0):
        sig_xx, sig_yy, sig_xy = get_exact_stress(x, y)
        return -sig_xy, -sig_yy, np.zeros_like(x)

    model.gen_surface_tractions(sel_right, trac_right, tf.Quad4n, 2, reset=True)
    model.gen_surface_tractions(sel_top, trac_top, tf.Quad4n, 2, reset=False)
    model.gen_surface_tractions(sel_left, trac_left, tf.Quad4n, 2, reset=False)
    model.gen_surface_tractions(sel_bot, trac_bot, tf.Quad4n, 2, reset=False)

    F_ext = model.Fg.copy()

    # Prevent rigid body motion using analytical Mode II fields
    calc_disp = get_analytical_displacements(nu=nu, plane_strain=True)
    U_dir = np.zeros(model.Kg.shape[0])
    fix_dofs = []

    x_coords = model.nodes[:, 1]
    y_coords = model.nodes[:, 2]

    dist_to_bl = x_coords**2 + y_coords**2
    dist_to_br = (x_coords - 10.0) ** 2 + y_coords**2

    bl_idx = np.argmin(dist_to_bl) + 1
    br_idx = np.argmin(dist_to_br) + 1

    bx, by = model.nodes[bl_idx - 1, 1], model.nodes[bl_idx - 1, 2]
    ux, uy = calc_disp(bx, by)
    dof_x_bl = model.list_dof.get_elem_dof_numbers([bl_idx], DofType.UX).ravel()
    dof_y_bl = model.list_dof.get_elem_dof_numbers([bl_idx], DofType.UY).ravel()

    U_dir[dof_x_bl] = ux
    U_dir[dof_y_bl] = uy
    fix_dofs.extend(dof_x_bl)
    fix_dofs.extend(dof_y_bl)

    bx, by = model.nodes[br_idx - 1, 1], model.nodes[br_idx - 1, 2]
    _, uy = calc_disp(bx, by)
    dof_y_br = model.list_dof.get_elem_dof_numbers([br_idx], DofType.UY).ravel()

    U_dir[dof_y_br] = uy
    fix_dofs.extend(dof_y_br)

    fix_dofs = np.array(sorted(set(fix_dofs)), dtype=int)
    model.gen_P(fix_dofs)

    # Solve System
    F_physical_reduced = F_ext - model.Kg @ U_dir
    K_ortho = model.ortho_T.T @ model.Kg @ model.ortho_T
    K_ortho = (K_ortho + K_ortho.T) / 2
    F_ortho_reduced = model.ortho_T.T @ F_physical_reduced

    K_reduced = model.P.T @ K_ortho @ model.P
    F_final = model.P.T @ F_ortho_reduced

    D = K_reduced.diagonal()
    D_inv_sqrt = sps.diags(1.0 / np.sqrt(D))

    Kg_scaled = D_inv_sqrt @ K_reduced @ D_inv_sqrt
    Fg_scaled = D_inv_sqrt @ F_final

    Ug_scaled = spsolve(Kg_scaled, Fg_scaled)

    Ug_reduced = D_inv_sqrt @ Ug_scaled
    Ug_tilde = model.P @ Ug_reduced + U_dir
    model.Ug = model.ortho_T @ Ug_tilde

    # PyVista rendering if requested (run once on a fine mesh to verify shear stress visually)
    if show_plot:
        mesh1 = my_build_Quad4n(model, mult=0.0).cast_to_unstructured_grid()
        ghosts = np.argwhere(mesh1["is_enriched"] > 0)
        mesh1.remove_cells(ghosts, inplace=True)
        mesh2 = build_XQuad4n(model, mult=0.0)
        blocks = pv.MultiBlock([mesh1, mesh2])

        pl = pv.Plotter()
        all_vm = np.concatenate(
            [mesh1.point_data["von_mises"], mesh2.point_data["von_mises"]]
        )
        v_max = np.percentile(all_vm, 99.5)
        pl.add_mesh(
            blocks, scalars="von_mises", cmap="turbo", show_edges=True, clim=[0, v_max]
        )
        pl.view_xy()
        pl.show()

    # Error Evaluation Loop
    get_exact_strain, get_exact_stress = derive_analytical_fields()
    total_exact_energy_sq = 0.0
    total_error_energy_sq = 0.0

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

        if h_enrich or t_enrich:
            most_enriched_node = elem_nodes[
                np.argmax(np.bitwise_and(elem_dofs, BRANCH_DOFS | HEAVISIDE_DOFS))
            ]
            ls_idx = model.ls[most_enriched_node - 1]
            tip = (
                model.tip[
                    elem_nodes[np.argmax(np.bitwise_and(elem_dofs, BRANCH_DOFS))] - 1
                ]
                if t_enrich
                else 0
            )
            phi_n, phi_t = model.level_sets[ls_idx].get(elem_nodes, tip)
        else:
            phi_n, phi_t = None, None

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

    return np.sqrt(total_error_energy_sq) / np.sqrt(total_exact_energy_sq)


def run_convergence_study():
    # mesh_sizes = [42, 56, 80, 100, 120]
    mesh_sizes = [9, 19, 31, 41, 55, 77, 101, 161]
    h_vals = [1.0 / n for n in mesh_sizes]

    errors_uncorrected = []
    errors_corrected = []

    print("Running Mode II Convergence Study...")
    print(
        f"{'Elements (NxN)':<15} | {'h':<10} | {'Uncorrected Error':<20} | {'Corrected Error'}"
    )
    print("-" * 72)

    # Let's flag the last run to output a PyVista visualization check
    for idx, (n, h) in enumerate(zip(mesh_sizes, h_vals)):
        visualize = idx == len(mesh_sizes) - 1
        err_uncorr = test_pure_mode_2_analytical_benchmark(n, n, False)
        err_corr = test_pure_mode_2_analytical_benchmark(
            n, n, True, show_plot=visualize
        )

        errors_uncorrected.append(err_uncorr)
        errors_corrected.append(err_corr)
        print(
            f"{n:<15} | {h:<10.4f} | {err_uncorr * 100.0:>16.4f}% | {err_corr * 100.0:>14.4f}%"
        )

    log_h = np.log(h_vals)
    slope_uncorr, _ = np.polyfit(log_h, np.log(errors_uncorrected), 1)
    slope_corr, _ = np.polyfit(log_h, np.log(errors_corrected), 1)

    print("-" * 72)
    print(f"Uncorrected Mode II Rate (Slope): {slope_uncorr:.4f} (Expected: ~0.5)")
    print(f"Corrected Mode II Rate (Slope):   {slope_corr:.4f} (Expected: ~1.0)")


if __name__ == "__main__":
    run_convergence_study()
