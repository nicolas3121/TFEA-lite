import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import scipy.sparse as sps
import sympy as sp
from geomdl import knotvector
from scipy.interpolate import BSpline
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
                "arrowstyle": "->",
                "color": color,
                "shrinkA": 0,
                "shrinkB": 5,
                "alpha": 0.7,
            },
        )


def get_analytical_displacements(
    nu=0.30, plane_strain=True, xc=5.0, yc=5.0, crack_angle=0.0
):
    kappa = 3.0 - 4.0 * nu if plane_strain else (3.0 - nu) / (1.0 + nu)
    c, s = np.cos(crack_angle), np.sin(crack_angle)

    def eval_displacements(x, y):
        dx, dy = x - xc, y - yc

        # Rotate global points into local crack tip coordinate system
        x_loc = c * dx + s * dy
        y_loc = -s * dx + c * dy

        r = np.sqrt(x_loc**2 + y_loc**2)
        theta = np.arctan2(y_loc, x_loc)

        u_x_loc = np.sqrt(r) * (
            (kappa - 0.5) * np.cos(theta / 2) - 0.5 * np.cos(3 * theta / 2)
        )
        u_y_loc = np.sqrt(r) * (
            (kappa + 0.5) * np.sin(theta / 2) - 0.5 * np.sin(3 * theta / 2)
        )

        # Rotate displacements back to global system
        u_x = c * u_x_loc - s * u_y_loc
        u_y = s * u_x_loc + c * u_y_loc
        return u_x, u_y

    return eval_displacements


def derive_analytical_fields(xc_val=5.0, yc_val=5.0, crack_angle=0.0):
    x, y = sp.symbols("x y", real=True)

    E = 1.0
    nu = 0.30
    kappa = 3.0 - 4.0 * nu

    mu = E / (2.0 * (1.0 + nu))
    lmbda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

    dx, dy = x - xc_val, y - yc_val
    c, s = sp.cos(crack_angle), sp.sin(crack_angle)

    # Local coordinates mapped from global
    x_loc = c * dx + s * dy
    y_loc = -s * dx + c * dy

    r = sp.sqrt(x_loc**2 + y_loc**2)
    theta = sp.atan2(y_loc, x_loc)

    u_x_loc = sp.sqrt(r) * (
        (kappa - 0.5) * sp.cos(theta / 2) - 0.5 * sp.cos(3 * theta / 2)
    )
    u_y_loc = sp.sqrt(r) * (
        (kappa + 0.5) * sp.sin(theta / 2) - 0.5 * sp.sin(3 * theta / 2)
    )

    u_x = c * u_x_loc - s * u_y_loc
    u_y = s * u_x_loc + c * u_y_loc

    eps_xx = sp.diff(u_x, x)
    eps_yy = sp.diff(u_y, y)
    gamma_xy = sp.diff(u_x, y) + sp.diff(u_y, x)

    sig_xx = 2 * mu * eps_xx + lmbda * (eps_xx + eps_yy)
    sig_yy = 2 * mu * eps_yy + lmbda * (eps_xx + eps_yy)
    sig_xy = mu * gamma_xy

    get_strain = sp.lambdify((x, y), (eps_xx, eps_yy, gamma_xy), modules="numpy")
    get_stress = sp.lambdify((x, y), (sig_xx, sig_yy, sig_xy), modules="numpy")

    return get_strain, get_stress


# ==============================================================================
# TRIANGLE ELEMENT ENERGY NORM INTEGRATION (From Snippet 2)
# ==============================================================================
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


# ==============================================================================
# QUAD ELEMENT ENERGY NORM INTEGRATION (From Snippet 1)
# ==============================================================================
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


def test_pure_mode_1_analytical_benchmark(
    x_elem,
    y_elem,
    corrected,
    gen=tf.gen_rect_Quad4n,
    error_fn=calculate_element_energy_norm,
    geometrical_range=1.0,
    crack_angle=0.0,
    plot=False,
):
    E_mod = 1.0
    nu = 0.3

    # --- CONVERT TO EFFECTIVE PLANE STRAIN PROPERTIES ---
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

    # Dynamic starting point based on crack_angle to ensure it goes through diagonals.
    L_crack = 10.0
    tip_x, tip_y = 5.0, 5.0
    p1 = np.array(
        [tip_x - L_crack * np.cos(crack_angle), tip_y - L_crack * np.sin(crack_angle)]
    )
    p2 = np.array([tip_x, tip_y])

    control_points = np.linspace(p1, p2, 12).tolist()
    n = len(control_points)
    k = 2
    knots = knotvector.generate(k, n)
    bspline = BSpline(knots, np.array(control_points), k)

    model.insert_crack_spline(
        bspline, embedded=False, h=10 / x_elem, snapping_tolerance=0.05
    )

    model.gen_list_dof(dof_per_node=tf.IS_2D)
    elem_dict = {"Quad4n": tf.XQuad4n, "Tri3n": tf.XTri3n}
    model.cal_global_matrices(elem_dict, eval_mass=False)

    calc_disp = get_analytical_displacements(
        nu=nu, plane_strain=True, xc=tip_x, yc=tip_y, crack_angle=crack_angle
    )

    tol = 1e-8
    x_coords = model.nodes[:, 1]
    y_coords = model.nodes[:, 2]

    is_boundary = (
        (np.abs(x_coords - 0.0) < tol)
        | (np.abs(x_coords - 10.0) < tol)
        | (np.abs(y_coords - 0.0) < tol)
        | (np.abs(y_coords - 10.0) < tol)
    )
    boundary_node_ids = np.where(is_boundary)[0] + 1

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

    fix_dofs = np.array(sorted(set(fix_dofs)), dtype=int)
    model.gen_P(fix_dofs)

    F_ext = np.zeros(model.Kg.shape[0])

    F_physical_reduced = F_ext - model.Kg @ U_dir
    K_ortho = model.ortho_T.T @ model.Kg @ model.ortho_T
    K_ortho = (K_ortho + K_ortho.T) / 2
    F_ortho_reduced = model.ortho_T.T @ F_physical_reduced

    if plot:
        print("   - K_ortho evaluated.")

    K_reduced = model.P.T @ K_ortho @ model.P
    F_final = model.P.T @ F_ortho_reduced

    if plot:
        print("   - K_reduced (Lifting) evaluated.")

    D = K_reduced.diagonal()
    D_inv_sqrt = sps.diags(1.0 / np.sqrt(D))

    Kg_scaled = D_inv_sqrt @ K_reduced @ D_inv_sqrt
    Fg_scaled = D_inv_sqrt @ F_final
    # print("   - Diagonal scaling applied.")
    # print("   - Start solving for U = inv(K)F ...")
    # K_dense = Kg_scaled.toarray()
    # singular_values = svdvals(K_dense)
    # # Look at the smallest 5 singular values
    # print("singular_values")
    # print(singular_values[-5:])

    if plot:
        print("   - Diagonal scaling applied.")
    if plot:
        print("   - Start solving for U = inv(K)F ...")

    Ug_scaled = spsolve(Kg_scaled, Fg_scaled)

    Ug_reduced = D_inv_sqrt @ Ug_scaled
    Ug_tilde = model.P @ Ug_reduced + U_dir
    model.Ug = model.ortho_T @ Ug_tilde

    # ---------------------------------------------------------
    # RESTORED PYVISTA PLOTTING BLOCK
    # ---------------------------------------------------------
    if plot:
        mesh1 = my_build_Quad4n(model, mult=0.1).cast_to_unstructured_grid()
        ghosts = np.argwhere(mesh1["is_enriched"] > 0)
        mesh1.remove_cells(ghosts, inplace=True)

        mesh2 = build_XQuad4n(model, mult=0.1)

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
            show_scalar_bar=False,
        )

        pl.view_xy()
        pl.enable_anti_aliasing()
        pl.show()

    # Continue evaluating error...
    get_exact_strain, get_exact_stress = derive_analytical_fields(
        crack_angle=crack_angle
    )

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

            tip = 0
            if t_enrich:
                tip = model.tip[
                    elem_nodes[np.argmax(np.bitwise_and(elem_dofs, BRANCH_DOFS))] - 1
                ]
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

    relative_error = np.sqrt(total_error_energy_sq) / np.sqrt(total_exact_energy_sq)
    return relative_error


def run_convergence_study(crack_angle=0.0):
    # mesh_sizes = [21, 33, 41, 51, 61, 81, 101, 121, 161]
    # mesh_sizes = np.array([11, 21, 31, 41, 61, 81, 121, 161])
    # mesh_sizes = [40, 41, 42]
    mesh_sizes = [21, 33, 41, 51, 61]
    h_vals = [10.0 / n for n in mesh_sizes]

    errors_uncorrected = []
    errors_corrected = []
    # errors_corrected_tri = []

    print(
        f"Running Convergence Study at Crack Angle = {np.degrees(crack_angle):.1f}°..."
    )
    print(
        f"{'Elements':<15} | {'h':<10} | {'Uncorr. Quad':<15} | {'Corr. Quad':<15} | {'Corr. Tri'}"
    )
    print("-" * 80)

    for n, h in zip(mesh_sizes, h_vals):
        # Standard SGFEM (Uncorrected Quads)
        err_uncorr = test_pure_mode_1_analytical_benchmark(
            n, n, False, crack_angle=crack_angle, plot=False
        )

        # Stable-Corrected XFEM (Corrected Quads)
        err_corr = test_pure_mode_1_analytical_benchmark(
            n, n, True, crack_angle=crack_angle, plot=False
        )

        # Stable-Corrected XFEM (Corrected Triangles)
        # err_corr_tri = test_pure_mode_1_analytical_benchmark(
        #     n,
        #     n,
        #     True,  # Passing True here to match SGFEM behavior for tris
        #     gen=tf.gen_rect_Tri3n,
        #     error_fn=calculate_element_energy_norm_tri,
        #     geometrical_range=0.6,
        #     crack_angle=crack_angle,
        #     plot=False,
        # )

        errors_uncorrected.append(err_uncorr)
        errors_corrected.append(err_corr)
        # errors_corrected_tri.append(err_corr_tri)

        print(
            f"{n:<15} | {h:<10.4f} | {err_uncorr * 100.0:>13.4f}% | {err_corr * 100.0:>13.4f}% %"
        )

    log_h = np.log(h_vals)
    slope_uncorr, _ = np.polyfit(log_h, np.log(errors_uncorrected), 1)
    slope_corr, _ = np.polyfit(log_h, np.log(errors_corrected), 1)
    # slope_corr_tri, _ = np.polyfit(log_h, np.log(errors_corrected_tri), 1)

    print("-" * 80)
    print(f"Uncorrected Quad Rate (Slope): {slope_uncorr:.4f} (Expected: ~0.5)")
    print(f"Corrected Quad Rate (Slope):   {slope_corr:.4f} (Expected: ~1.0)")
    # print(f"Corrected Tri Rate (Slope):    {slope_corr_tri:.4f} (Expected: ~1.0)")

    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    # 1. Uncorrected Quads (Red)
    plt.loglog(
        h_vals,
        errors_uncorrected,
        marker="s",
        markersize=8,
        linestyle="-",
        linewidth=2,
        color="red",
        markerfacecolor="none",
        label="SO-XFEM",
    )
    annotate_local_slopes(
        h_vals, errors_uncorrected, ax, text_offset=(-20, 30), x_is_h=True, color="red"
    )

    # 2. Corrected Quads (Blue)
    plt.loglog(
        h_vals,
        errors_corrected,
        marker="o",
        markersize=8,
        linestyle="-",
        linewidth=2,
        color="blue",
        markerfacecolor="none",
        label="SCO-XFEM",
    )
    annotate_local_slopes(
        h_vals, errors_corrected, ax, text_offset=(20, -30), x_is_h=True, color="blue"
    )

    # 3. Corrected Triangles (Green)
    # plt.loglog(
    #     h_vals,
    #     errors_corrected_tri,
    #     marker="^",
    #     markersize=8,
    #     linestyle="-",
    #     linewidth=2,
    #     color="green",
    #     markerfacecolor="none",
    #     label=f"Stable-Corrected Triangles (Avg Slope: {slope_corr_tri:.2f})",
    # )
    # # Offset the annotations slightly so they don't perfectly overlap the blue ones
    # annotate_local_slopes(
    #     h_vals,
    #     errors_corrected_tri,
    #     ax,
    #     text_offset=(0, 25),
    #     x_is_h=True,
    #     color="green",
    # )

    # Theoretical Lines
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

    plt.xlabel(r"Element Size $h$", fontsize=14)
    plt.ylabel(r"Relative Energy Norm Error", fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.gca().invert_xaxis()
    plt.legend(
        fontsize=11,
        loc="upper right",
        framealpha=1.0,
        edgecolor="black",
        fancybox=False,
    )
    plt.xticks(h_vals, labels=[f"{h:.3f}" for h in h_vals])

    plt.tight_layout()
    plt.savefig("convergence_annotated_with_tri.pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    # Test 1: Run a single plot explicitly at exactly 45 degrees
    # to visualize the crack passing through the node diagonals.
    print(
        "Testing visualization for diagonal crack (close window to start convergence study)..."
    )
    # test_pure_mode_1_analytical_benchmark(
    #     41, 41, True, crack_angle=-np.pi / 4, plot=True
    # )

    # # Test 2: Run the full convergence study at 45 degrees
    run_convergence_study(crack_angle=-np.pi / 4)
