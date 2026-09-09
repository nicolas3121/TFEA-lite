import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import sympy as sp
from matplotlib.ticker import NullFormatter

import tfealite as tf
import tfealite.core.quadratures as qd
from tfealite.core.dofs import BRANCH_DOFS, HEAVISIDE_DOFS, DofType
from tfealite.core.level_set import CutType
from tfealite.core.quadratures import DuffySinh3D
from tfealite.core.surfaces import geomdl_to_NdBSplines, init_crack_plane_geomdl
from tfealite.elements.utils import (
    cal_B_3d_vec,
    cut_embedding_tetr_iter,
    fill_element_displacement,
    partial_cut_embedding_tetr_iter,
)
from tfealite.elements.XTetr4n import XTetr4n
from tfealite.visualization.build_mesh import (
    build_XTetr4n,
    my_build_Tetr4n,
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


def calculate_element_energy_norm_3d(element, Ue, get_exact_strain, get_exact_stress):
    exact_energy_sq = 0.0
    error_energy_sq = 0.0
    N_FN = 4

    def accumulate_error(nat_coords, w_eff):
        nonlocal exact_energy_sq, error_energy_sq

        N, dN_dxi = element.shape_functions(nat_coords)
        x_gp = N[:, :N_FN] @ element.node_coords

        J = dN_dxi[:, :, :N_FN] @ element.node_coords
        dN_dxy = np.linalg.solve(J, dN_dxi)
        B = cal_B_3d_vec(dN_dxy)

        # Numerical Strains and Stresses
        eps_h = np.einsum("gij,j->gi", B, Ue)
        sig_h = np.einsum("ij,gj->gi", element.cal_D(), eps_h)

        # Exact Strains and Stresses
        exact_strain_vals = get_exact_strain(x_gp[:, 0], x_gp[:, 1], x_gp[:, 2])
        exact_stress_vals = get_exact_stress(x_gp[:, 0], x_gp[:, 1], x_gp[:, 2])
        # Broadcast all tuple elements to have the exact same shape
        exact_strain_vals = np.broadcast_arrays(*exact_strain_vals)
        exact_stress_vals = np.broadcast_arrays(*exact_stress_vals)

        # Now column_stack will work perfectly
        eps_ex = np.column_stack(exact_strain_vals)
        sig_ex = np.column_stack(exact_stress_vals)

        # print(exact_stress_vals)
        # eps_ex = np.column_stack(exact_strain_vals)
        # sig_ex = np.column_stack(exact_stress_vals)

        # Differences
        diff_eps = eps_ex - eps_h
        diff_sig = sig_ex - sig_h

        # Energy Inner Products
        ex_energy_density = np.sum(sig_ex * eps_ex, axis=1)
        err_energy_density = np.sum(diff_sig * diff_eps, axis=1)

        # Integrate
        error_energy_sq += np.sum(err_energy_density * w_eff)
        exact_energy_sq += np.sum(ex_energy_density * w_eff)

    x_e = element.node_coords
    (rule, correction) = qd.TETR_RULES[1]

    _, dN_dxi = element.shape_functions(rule[:, :-1].T)
    J_base = dN_dxi[0, :, :N_FN] @ x_e
    detJ = np.abs(np.linalg.det(J_base))

    if (
        not getattr(element, "h_enrich", False)
        and not getattr(element, "partial_cut", False)
        and not getattr(element, "t_enrich", False)
    ):
        rule, correction = qd.TETR_RULES[1]
        nat_coords = rule[:, :3].T
        w_eff = rule[:, 3] * correction * detJ
        accumulate_error(nat_coords, w_eff)
    elif not getattr(element, "h_enrich", False) and not getattr(
        element, "partial_cut", False
    ):
        rule, correction = qd.TETR_RULES[13]
        nat_coords = rule[:, :3].T
        w_eff = rule[:, 3] * correction * detJ
        accumulate_error(nat_coords, w_eff)
    elif getattr(element, "h_enrich", False):
        Nc, kappa = element._cal_intersections()
        rule, correction = qd.TETR_RULES[13] if element.t_enrich else qd.TETR_RULES[2]

        for Ni, detJi in cut_embedding_tetr_iter(Nc, kappa):
            n_sub, _ = element._base_shape_functions(rule[:, :3].T)
            sub_nat_coords = element.NAT_COORDS.T @ Ni @ n_sub.T
            w_eff = rule[:, 3] * detJi * correction * detJ
            accumulate_error(sub_nat_coords, w_eff)

    elif getattr(element, "partial_cut", False):
        Nc, _ = element._cal_intersections()
        tip, tip_on_interface = element._cal_front_intersections()
        rule_hex, corr_hex = qd.UNIT_HEX_RULES[10]

        for Ni, detJi, n_on_interface in partial_cut_embedding_tetr_iter(
            Nc, tip, tip_on_interface
        ):
            duffy = DuffySinh3D((x_e.T @ Ni).T)
            for b1, b2 in [(2, min(n_on_interface, 2))]:
                rule_d = duffy.transform(rule_hex[:, :3].T, beta1=b1, beta2=b2)
                n_sub, _ = element._base_shape_functions(rule_d[:3])
                sub_nat_coords = element.NAT_COORDS.T @ Ni @ n_sub.T
                w_eff = rule_hex[:, 3] * rule_d[3] * detJi * corr_hex * detJ
                accumulate_error(sub_nat_coords, w_eff)

    return exact_energy_sq, error_energy_sq


def get_analytical_displacements_3d(nu=0.30, xc=5.0, yc=5.0, crack_angle=0.0):
    kappa = 3.0 - 4.0 * nu
    c, s = np.cos(crack_angle), np.sin(crack_angle)

    def eval_displacements(x, y, z):
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
        u_z_loc = np.zeros_like(x)

        # Rotate displacements back to global system
        u_x = c * u_x_loc - s * u_y_loc
        u_y = s * u_x_loc + c * u_y_loc
        u_z = u_z_loc

        return u_x, u_y, u_z

    return eval_displacements


def derive_analytical_fields_3d(xc_val=5.0, yc_val=5.0, crack_angle=0.0):
    """Generates exact strain and stress evaluators for the energy norm in 3D."""
    x, y, z = sp.symbols("x y z", real=True)

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

    # Rotate displacements back to global system
    u_x = c * u_x_loc - s * u_y_loc
    u_y = s * u_x_loc + c * u_y_loc
    u_z = sp.Float(0.0)

    # Derivatives using global coordinates (SymPy handles chain rule implicitly)
    eps_xx = sp.diff(u_x, x)
    eps_yy = sp.diff(u_y, y)
    eps_zz = sp.diff(u_z, z)
    gamma_xy = sp.diff(u_x, y) + sp.diff(u_y, x)
    gamma_xz = sp.diff(u_x, z) + sp.diff(u_z, x)
    gamma_yz = sp.diff(u_y, z) + sp.diff(u_z, y)

    strains = (eps_xx, eps_yy, eps_zz, gamma_xy, gamma_xz, gamma_yz)

    trace_eps = eps_xx + eps_yy + eps_zz
    sig_xx = lmbda * trace_eps + 2 * mu * eps_xx
    sig_yy = lmbda * trace_eps + 2 * mu * eps_yy
    sig_zz = lmbda * trace_eps + 2 * mu * eps_zz
    sig_xy = mu * gamma_xy
    sig_xz = mu * gamma_xz
    sig_yz = mu * gamma_yz

    stresses = (sig_xx, sig_yy, sig_zz, sig_xy, sig_xz, sig_yz)

    get_strain = sp.lambdify((x, y, z), strains, modules="numpy")
    get_stress = sp.lambdify((x, y, z), stresses, modules="numpy")

    return get_strain, get_stress


def test_pure_mode_1_3d_analytical_benchmark(nx, ny, nz, corrected, crack_angle=0.0):
    materials = [[1, {"E": 1.0, "nu": 0.3, "rho": 1.0}]]
    reals = [[1, {}]]
    L, H, D = 10.0, 10.0, 2
    nodes, elements = tf.core.model.gen_rect_Tetr4n(L, H, D, nx, ny, nz)

    model = tf.XFEModel(
        nodes,
        elements,
        materials,
        reals,
        tip_enrichment=True,
        geometrical_range=0.5,
        corrected=corrected,
    )

    # # Dynamic crack insertion based on angle
    # L_crack = 10.0  # Ensure it reaches the edge
    # tip_x, tip_y = 5.0, 5.0
    #
    # # p1: start of the crack line at the bottom face
    # p1 = np.array(
    #     [
    #         tip_x - L_crack * np.cos(crack_angle),
    #         tip_y - L_crack * np.sin(crack_angle),
    #         -1.0,
    #     ]
    # )
    # # p2: crack tip at the bottom face
    # p2 = np.array([tip_x, tip_y, -1.0])
    # # p3: crack tip at the top face (ensure it spans entire Z domain)
    # p3 = np.array([tip_x, tip_y, D + 1.0])
    # #
    # model.insert_planar_crack_segment(p1, p2, p3, embedded=False)
    L_crack = 10.0  # Ensure it reaches the edge
    tip_x, tip_y = 5.134, 5.0

    # 1. Calculate the dimensions
    # The old crack spans Z from -1.0 to D + 1.0, so the total width is D + 2.0
    width = D + 2.0
    length = L_crack

    # 2. Calculate the center of the crack front for translation
    # Center X, Y is the tip. Center Z is exactly halfway between -1.0 and D + 1.0
    center_z = (-1.0 + (D + 1.0)) / 2.0  # This simplifies to D / 2.0
    translation = np.array([tip_x, tip_y, center_z])

    # 3. Build the rotation matrix mapping Local to Global
    # Local X-axis aligns with the global Z-axis (the width of the crack)
    local_x = np.array([0, 0, 1])

    # Local Y-axis aligns with the direction of crack propagation
    local_y = np.array([np.cos(crack_angle), np.sin(crack_angle), 0])

    # Local Z-axis is the normal to the crack plane (cross product of X and Y)
    local_z = np.cross(local_x, local_y)

    # Combine into a rotation matrix
    rotation = np.column_stack((local_x, local_y, local_z))

    # 4. Generate the B-Spline surface
    crack = init_crack_plane_geomdl(
        width=width,
        length=length,
        rotation=rotation,
        translation=translation,
        embedded=False,
    )

    scipy_crack = geomdl_to_NdBSplines(crack)

    # Optional: check the bounds of the generated surface
    for _, _, surf in scipy_crack:
        print("Crack front bottom edge:", surf([1, 0]))
        print("Crack front top edge:", surf([1, 1]))

    # CRITICAL FIX: Snapping tolerance must be > 0 to avoid 1/sqrt(r) dividing by zero on aligned nodes.
    model.insert_ndbsplines_crack(scipy_crack, 10 / nx, snapping_tolerance=0.05)

    model.gen_list_dof(dof_per_node=tf.IS_3D)
    model.cal_global_matrices({"Tetr4n": tf.XTetr4n})

    calc_disp = get_analytical_displacements_3d(
        nu=0.3, xc=tip_x, yc=tip_y, crack_angle=crack_angle
    )
    tol = 1e-8
    x, y, z = model.nodes[:, 1], model.nodes[:, 2], model.nodes[:, 3]
    on_boundary = (
        (np.abs(x) < tol)
        | (np.abs(x - L) < tol)
        | (np.abs(y) < tol)
        | (np.abs(y - H) < tol)
    )
    boundary_nodes = np.where(on_boundary)[0] + 1

    U_dir = np.zeros(model.Kg.shape[0])
    u_x, u_y, u_z = calc_disp(
        x[boundary_nodes - 1], y[boundary_nodes - 1], z[boundary_nodes - 1]
    )

    U_dir[model.list_dof.get_elem_dof_numbers(boundary_nodes, DofType.UX).ravel()] = u_x
    U_dir[model.list_dof.get_elem_dof_numbers(boundary_nodes, DofType.UY).ravel()] = u_y

    fix_dofs = list(model.list_dof.get_elem_dof_numbers_flat(boundary_nodes, tf.IS_3D))
    z_dofs = np.concatenate(
        [
            model.list_dof.get_elem_dof_numbers_flat(
                np.array(model.nodes[:, 0], dtype=int), dof
            )
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
    fix_dofs.extend(list(z_dofs))

    model.gen_P(np.unique(sorted(fix_dofs)))

    K_reduced = model.P.T @ model.ortho_T.T @ model.Kg @ model.ortho_T @ model.P
    F_reduced = model.P.T @ model.ortho_T.T @ (-model.Kg @ U_dir)

    D_diag = K_reduced.diagonal()
    D_diag = np.where(D_diag > 0, D_diag, 1.0)
    D_inv_sqrt = sps.diags(1.0 / np.sqrt(D_diag))

    A_scaled = D_inv_sqrt @ K_reduced @ D_inv_sqrt
    b_scaled = D_inv_sqrt @ F_reduced

    Ug_scaled, info = spla.cg(A_scaled, b_scaled, rtol=1e-12, atol=1e-12, maxiter=10000)

    if info == 0:
        pass  # print(f"CG converged successfully. Matrix size: {K_reduced.shape[0]}")
    elif info > 0:
        print(
            f"Warning: CG failed to converge after {info} iterations. Check conditioning."
        )
    else:
        print("Error: CG illegal input or breakdown.")

    model.Ug = model.ortho_T @ (model.P @ (D_inv_sqrt @ Ug_scaled) + U_dir)

    plot = True
    if plot:
        mult = 0.1

        # mult = 1e5
        # model.Ug = np.zeros(len(model.list_dof))
        output_file_png = f"von_mises_render_{nx}_{corrected}.png"
        mesh1 = my_build_Tetr4n(model, mult=mult).cast_to_unstructured_grid()
        ghosts = np.argwhere(mesh1["is_enriched"] > 0)
        mesh1.remove_cells(ghosts, inplace=True)
        mesh2 = build_XTetr4n(model, mult=mult)
        blocks = pv.MultiBlock([mesh1, mesh2])
        # blocks.plot(show_edges=True, color="lightblue")
        vm_1 = mesh1.point_data["von_mises"]
        vm_2 = mesh2.point_data["von_mises"]
        all_vm = np.concatenate([vm_1, vm_2])
        v_max = np.percentile(all_vm, 90)
        # pl = pv.Plotter()
        pl = pv.Plotter(window_size=(3000, 3000), off_screen=True)
        # pl.add_mesh(
        #     blocks,
        #     scalars="von_mises",  # The exact string key in your point_data dict
        #     cmap="turbo",  # A great colormap for stress fields (or use "jet", "viridis")
        #     show_edges=True,  # Shows the mesh grid (including your sub-triangulations!)
        #     clim=[0, v_max],
        #     scalar_bar_args={"title": "Von Mises Stress"},
        # )
        # pl.show()
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
        # Rotate the camera slightly to reveal the 3D depth
        pl.camera.azimuth -= 25  # Rotates the camera to the left/right
        pl.reset_camera_clipping_range()
        # pl.camera.elevation += 15  # Tilts the camera up slightly
        # pl.view_isometric()
        pl.enable_anti_aliasing()

        # Optional: Zoom out slightly (e.g., 0.9 or 0.85) so the square mesh
        # doesn't cramp the legends and scalar bars at the edges.
        pl.camera.zoom(0.9)

        # Render and save (no need to pass window_size here, it's defined at init)
        pl.show(screenshot=output_file_png)
        print(f"Saved high-res raster graphic to: {output_file_png}")

    get_exact_strain, get_exact_stress = derive_analytical_fields_3d(
        xc_val=tip_x, yc_val=tip_y, crack_angle=crack_angle
    )
    total_ex, total_err = 0.0, 0.0

    cut_info = model.cut_info
    for i_e, ele_info in enumerate(model.elements):
        elem_func = XTetr4n
        mat_id = ele_info[2]
        real_ie = ele_info[3]
        n_dofs = model.dof_per_node.bit_count()
        elem_nodes = np.array(ele_info[4], dtype=np.uint32)
        elem_vertices = model.nodes[elem_nodes - 1, 1 : 1 + n_dofs]
        elem_dofs = model.list_dof.get_elem_dofs(elem_nodes)
        material = model.materials[mat_id - 1][1]
        real = model.reals[real_ie - 1][1]
        local_dofs_per_node = np.bitwise_or.reduce(elem_dofs)
        h_enrich = local_dofs_per_node & HEAVISIDE_DOFS != 0
        if corrected:
            in_range = model.in_range[elem_nodes - 1]
        else:
            in_range = np.ones(len(ele_info[4]))
        t_enrich = local_dofs_per_node & BRANCH_DOFS != 0
        if h_enrich or t_enrich:
            tip = 0
            ls = model.ls[
                elem_nodes[
                    np.argmax(np.bitwise_and(elem_dofs, BRANCH_DOFS | HEAVISIDE_DOFS))
                ]
                - 1
            ]
            if t_enrich:
                tip = model.tip[
                    elem_nodes[np.argmax(np.bitwise_and(elem_dofs, BRANCH_DOFS))] - 1
                ]
            phi_n, phi_t = model.level_sets[ls].get(elem_nodes, tip)

            assert cut_info
            ci = cut_info.get(ele_info[0])
            partial_cut = False
            if ci is not None:
                _, cut_type, _ = ci
                partial_cut = cut_type == CutType.PARTIAL

            elem = elem_func(
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
        else:
            elem = elem_func(elem_vertices, material, real)

        Ue = fill_element_displacement(elem_nodes, model.list_dof, model.Ug)
        ex_sq, err_sq = calculate_element_energy_norm_3d(
            elem, Ue, get_exact_strain, get_exact_stress
        )
        total_ex += ex_sq
        total_err += err_sq

    return np.sqrt(total_err) / np.sqrt(total_ex)


def run_convergence_study_3d(crack_angle=0.0):
    mesh_sizes = [51]
    # mesh_sizes = np.array([13, 21, 31, 51, 71, 101])
    h_vals = [10.0 / n for n in mesh_sizes]
    errs_uncorr, errs_corr = [], []

    print(
        f"Running 3D Convergence Study (Tetr4n) at Crack Angle = {np.degrees(crack_angle):.1f}°..."
    )
    print(f"{'Mesh (NxNx2)':<15} | {'h':<10} | {'Uncorrected':<15} | {'Corrected'}")
    print("-" * 65)

    for n in mesh_sizes:
        u, c = (
            test_pure_mode_1_3d_analytical_benchmark(
                n, n, 2, False, crack_angle=crack_angle
            ),
            test_pure_mode_1_3d_analytical_benchmark(
                n, n, 2, True, crack_angle=crack_angle
            ),
        )
        errs_uncorr.append(u)
        errs_corr.append(c)
        print(
            f"{f'{n}x{n}x{n}':<15} | {10.0 / n:<10.4f} | {u * 100:>14.4f}% | {c * 100:>12.4f}%"
        )

    s_u, s_c = (
        np.polyfit(np.log(h_vals), np.log(errs_uncorr), 1)[0],
        np.polyfit(np.log(h_vals), np.log(errs_corr), 1)[0],
    )
    print("-" * 65)
    print(
        f"Rates: Uncorrected = {s_u:.4f} (exp. 0.5), Corrected = {s_c:.4f} (exp. 1.0)"
    )

    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    # 1. Uncorrected Quads (Red)
    plt.loglog(
        h_vals,
        errs_uncorr,
        marker="s",
        markersize=8,
        linestyle="-",
        linewidth=2,
        color="red",
        markerfacecolor="none",
        label="SO-XFEM",
    )
    annotate_local_slopes(
        h_vals, errs_uncorr, ax, text_offset=(-20, 30), x_is_h=True, color="red"
    )

    # 2. Corrected Quads (Blue)
    plt.loglog(
        h_vals,
        errs_corr,
        marker="o",
        markersize=8,
        linestyle="-",
        linewidth=2,
        color="blue",
        markerfacecolor="none",
        label="SCO-XFEM",
    )
    annotate_local_slopes(
        h_vals, errs_corr, ax, text_offset=(20, -30), x_is_h=True, color="blue"
    )

    C_1 = errs_corr[0] / (h_vals[0] ** 1.0)
    theo_1 = [C_1 * (h**1.0) for h in h_vals]
    plt.loglog(
        h_vals,
        theo_1,
        linestyle="--",
        color="black",
        alpha=0.6,
        label=r"Optimal $\mathcal{O}(h^{1.0})$",
    )

    C_05 = errs_uncorr[0] / (h_vals[0] ** 0.5)
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
    ax.xaxis.set_minor_formatter(NullFormatter())

    plt.tight_layout()

    plt.savefig("backup_convergence_test_3d.pdf")
    plt.show()


if __name__ == "__main__":
    tip_x, tip_y = 5.134, 5.0
    x = 0
    y = np.e
    dx = tip_x
    dy = tip_y - y
    theta = np.atan2(dy, dx)
    run_convergence_study_3d(theta)

    # Plotting code remains identical (omitted for brevity) ...
