import numpy as np
from scipy.interpolate import BSpline, splprep

import tfealite as tf
from tfealite import DofType
from tfealite.core.sif import DisplacementCorrelationMethodSIF as DCMSIF


def test_pure_mode_1():
    E_mod = 200e9
    nu = 0.3
    kappa = (3 - nu) / (3 + nu)
    G_mod = (E_mod) / (2 * (1 + nu))
    a = 0.115
    W = 0.5
    x_elem = 100
    y_elem = int(1.5 * x_elem) + 1
    nodes, elements = tf.gen_rect_Quad4n(2 * W, 1.5, x_elem, y_elem)
    materials = [[1, {"E": E_mod, "nu": nu, "rho": 7850}]]
    reals = [[1, {"t": 1}]]
    model = tf.XFEModel(
        nodes,
        elements,
        materials,
        reals,
        tip_enrichment=True,
        geometrical_range=0.07,
        corrected=True,
    )
    p1 = np.array([0.5 - a, 0.75])
    p2 = np.array([0.5 + a, 0.75])

    pts = np.linspace(p1, p2, 4).T
    tck, u = splprep(pts, s=0, k=3)
    bspline = BSpline(tck[0], np.transpose(tck[1]), tck[2])
    h = 2 * W / x_elem
    model.insert_crack_spline(bspline, embedded=True, h=h, snapping_tolerance=0.03)

    model.gen_list_dof(dof_per_node=tf.IS_2D)

    model.cal_global_matrices(tf.XQuad4n)

    def sel_condition(x, y, z):
        return y - 0.0

    to_constrain = sel_condition(None, model.nodes[:, 2], None) < 1e-8
    nodes_to_constrain = 1 + np.where(to_constrain)[0]
    fix_dofs = list(
        model.list_dof.get_elem_dof_numbers_flat(nodes_to_constrain, DofType.UY)
    )
    fix_dofs.append(model.list_dof[(x_elem // 2 + 1, DofType.UX)])

    model.gen_P(np.array(sorted(set(fix_dofs)), dtype=int))

    def sel_condition(x, y, z):
        return y - 1.5

    def force_expression(x, y, z):
        return 0.0, 1, 0.0

    model.gen_surface_tractions(sel_condition, force_expression, tf.Quad4n, 2)
    model.solve_static()
    # model.Ug = np.zeros_like(model.Ug)
    kappa = (3 - 0.3) / (1 + 0.3)

    dcm = DCMSIF(
        kappa, G_mod, np.array([0.011, 0.021, 0.025, 0.031, 0.041, 0.045]), None
    )
    K_I, K_II = dcm.cal_sif(model.level_sets[0], model, model.cut_info, 1.0)

    # analytical solutions https://doi.org/10.1016/0013-7944(68)90027-1
    K_I_analytical = 1 * np.sqrt(np.pi * a) * np.sqrt(1 / np.cos((np.pi * a) / (2 * W)))
    assert np.isclose(K_I, K_I_analytical, rtol=0.01)
    assert np.isclose(K_II / K_I, 0.0, atol=0.001)

    # model.Ug *= 1e10
    # mesh1 = my_build_Quad4n(model).cast_to_unstructured_grid()
    # ghosts = np.argwhere(mesh1["is_cut"] > 0)
    # mesh1.remove_cells(ghosts, inplace=True)
    # mesh2 = build_XQuad4n(model)
    # blocks = pv.MultiBlock([mesh1, mesh2])
    # # blocks.plot(show_edges=True, color="lightblue")
    # pl = pv.Plotter()
    # pl.add_mesh(blocks, color="lightblue", show_edges=True)
    # pl.view_xy()
    # # pl.enable_parallel_projection()
    # pl.show()
    # assert False


def test_mixed_mode():
    E_mod = 200e9
    nu = 0.3
    kappa = (3 - nu) / (3 + nu)
    G_mod = (E_mod) / (2 * (1 + nu))
    a = 0.115
    W = 0.5
    angle = np.pi / 4
    x_elem = 100
    y_elem = int(1.5 * x_elem) + 1
    nodes, elements = tf.gen_rect_Quad4n(2 * W, 1.5, x_elem, y_elem)
    materials = [[1, {"E": E_mod, "nu": nu, "rho": 7850}]]
    reals = [[1, {"t": 1}]]
    model = tf.XFEModel(
        nodes,
        elements,
        materials,
        reals,
        tip_enrichment=True,
        geometrical_range=0.07,
        corrected=True,
    )
    p1 = np.array([0.5 - a * np.cos(angle), 0.75 - a * np.sin(angle)])
    p2 = np.array([0.5 + a * np.cos(angle), 0.75 + a * np.sin(angle)])

    pts = np.linspace(p1, p2, 4).T
    tck, u = splprep(pts, s=0, k=3)
    bspline = BSpline(tck[0], np.transpose(tck[1]), tck[2])
    h = 2 * W / x_elem
    model.insert_crack_spline(bspline, embedded=True, h=h, snapping_tolerance=0.03)

    model.gen_list_dof(dof_per_node=tf.IS_2D)

    model.cal_global_matrices(tf.XQuad4n)

    def sel_condition(x, y, z):
        return y - 0.0

    to_constrain = sel_condition(None, model.nodes[:, 2], None) < 1e-8
    nodes_to_constrain = 1 + np.where(to_constrain)[0]
    fix_dofs = list(
        model.list_dof.get_elem_dof_numbers_flat(nodes_to_constrain, DofType.UY)
    )
    fix_dofs.append(model.list_dof[(x_elem // 2 + 1, DofType.UX)])

    model.gen_P(np.array(sorted(set(fix_dofs)), dtype=int))

    def sel_condition(x, y, z):
        return y - 1.5

    def force_expression(x, y, z):
        return 0.0, 1, 0.0

    model.gen_surface_tractions(sel_condition, force_expression, tf.Quad4n, 2)
    model.solve_static()
    # model.Ug = np.zeros_like(model.Ug)
    kappa = (3 - 0.3) / (1 + 0.3)

    dcm = DCMSIF(kappa, G_mod, np.array([0.035, 0.038, 0.043, 0.048, 0.055]), None)
    K_I, K_II = dcm.cal_sif(model.level_sets[0], model, model.cut_info, 1.0)
    print(K_I, K_II)

    K_I_analytical = 1 * np.sqrt(np.pi * a) * np.sin(angle) ** 2
    K_II_analytical = 1 * np.sqrt(np.pi * a) * np.sin(angle) * np.cos(angle)

    assert np.isclose(K_I / K_II, 1.0, atol=0.01)
    assert np.isclose(K_I, K_I_analytical, rtol=0.05)
    assert np.isclose(K_II, K_II_analytical, rtol=0.05)

    # model.Ug *= 1e10
    # mesh1 = my_build_Quad4n(model).cast_to_unstructured_grid()
    # ghosts = np.argwhere(mesh1["is_cut"] > 0)
    # mesh1.remove_cells(ghosts, inplace=True)
    # mesh2 = build_XQuad4n(model)
    # blocks = pv.MultiBlock([mesh1, mesh2])
    # # blocks.plot(show_edges=True, color="lightblue")
    # pl = pv.Plotter()
    # pl.add_mesh(blocks, color="lightblue", show_edges=True)
    # pl.view_xy()
    # # pl.enable_parallel_projection()
    # pl.show()
    # assert False
