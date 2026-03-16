import tfealite as tf
import tfealite.core.bc as bc
import numpy as np


def test_fully_cut_XTri3n():
    nodes, elements = tf.gen_rect_Tri3n(1, 1, 1, 1)
    print(nodes)
    print(elements)
    materials = [[1, {"E": 1, "nu": 0.33, "rho": 7850}]]
    reals = [[1, {"t": 1}]]

    model = tf.XFEModel(nodes, elements, materials, reals, tip_enrichment=False)

    model.insert_crack_segment(
        np.array([-0.1, 0.6]), np.array([1.1, 0.6]), embedded=True
    )
    # print(model.level_sets)

    model.gen_list_dof(dof_per_node=tf.IS_2D)
    print("full phi_n", model.level_sets[0].phi_n)
    print("full phi_t", model.level_sets[0].phi_t)
    model.cal_global_matrices(tf.XTri3n)

    def sel_condition(x, y, z):
        return y - 0.0

    model.gen_dirichlet_bc(sel_condition)

    def sel_condition(x, y, z):
        return y - 1.0

    def force_expression(x, y, z):
        return 0.0, 10**9 * 0.5, 0.0

    model.gen_nodal_forces(sel_condition, force_expression)
    tol = 1e-8
    for node in model.nodes:
        nid = int(node[0])
        x, y, z = map(float, node[1:4])

        if abs(sel_condition(x, y, z)) < tol:
            dof_number = model.list_dof[(nid, tf.DofType.UX)]
            model.Kg[dof_number, dof_number] += 10**9
            dof_number = model.list_dof[(nid, tf.DofType.UY)]
            model.Kg[dof_number, dof_number] += 10**9

    model.solve_static()

    Xg = model.Ug[
        model.list_dof.get_elem_dof_numbers(
            1 + np.arange(model.n_nodes, dtype=int), tf.DofType.UX
        ).flatten()
    ]
    assert np.allclose(Xg, 0.0)
    Yg = model.Ug[
        model.list_dof.get_elem_dof_numbers(
            1 + np.arange(model.n_nodes, dtype=int), tf.DofType.UY
        ).flatten()
    ]
    assert np.allclose(Yg, np.array([0.0, 0.0, 0.5, 0.5]))
    print(Xg)
    print(Yg)

    HXg = model.Ug[
        model.list_dof.get_elem_dof_numbers(
            1 + np.arange(model.n_nodes, dtype=int), tf.DofType.HX
        ).flatten()
    ]
    assert np.allclose(HXg, 0.0)
    HYg = model.Ug[
        model.list_dof.get_elem_dof_numbers(
            1 + np.arange(model.n_nodes, dtype=int), tf.DofType.HY
        ).flatten()
    ]
    assert np.allclose(HYg, 0.5)
    print(HXg)
    print(HYg)


def test_edge_cut_XTri3n():
    nodes, elements = tf.gen_rect_Tri3n(1, 1, 1, 2)
    print(nodes)
    print(elements)
    materials = [[1, {"E": 1, "nu": 0.33, "rho": 7850}]]
    reals = [[1, {"t": 1}]]

    model = tf.XFEModel(nodes, elements, materials, reals, tip_enrichment=False)

    model.insert_crack_segment(
        np.array([-0.1, 0.5]), np.array([1.1, 0.5]), embedded=True
    )
    # print(model.level_sets)

    model.gen_list_dof(dof_per_node=tf.IS_2D)
    print(model.list_dof.list_dof)
    model.cal_global_matrices(tf.XTri3n)

    def sel_condition(x, y, z):
        return y - 0.0

    bc.gen_dirichlet_bc(model, sel_condition)

    def sel_condition(x, y, z):
        return y - 1.0

    def force_expression(x, y, z):
        return 0.0, 10**9 * 1, 0.0

    model.gen_nodal_forces(sel_condition, force_expression)
    tol = 1e-8
    for node in model.nodes:
        nid = int(node[0])
        x, y, z = map(float, node[1:4])

        if abs(sel_condition(x, y, z)) < tol:
            dof_number = model.list_dof[(nid, tf.DofType.UX)]
            model.Kg[dof_number, dof_number] += 10**9
            dof_number = model.list_dof[(nid, tf.DofType.UY)]
            model.Kg[dof_number, dof_number] += 10**9

    model.solve_static()
    Xg = model.Ug[
        model.list_dof.get_elem_dof_numbers(
            1 + np.arange(model.n_nodes, dtype=int), tf.DofType.UX
        ).flatten()
    ]
    assert np.allclose(Xg, 0.0)
    Yg = model.Ug[
        model.list_dof.get_elem_dof_numbers(
            1 + np.arange(model.n_nodes, dtype=int), tf.DofType.UY
        ).flatten()
    ]
    assert np.allclose(Yg, np.array([0.0, 0.0, 0.5, 0.5, 1.0, 1.0]))
    print(Xg)
    print(Yg)

    HXg = model.Ug[
        model.list_dof.get_elem_dof_numbers_flat(
            1 + np.arange(model.n_nodes, dtype=int), tf.DofType.HX
        ).flatten()
    ]
    assert np.allclose(HXg, 0.0)
    HYg = model.Ug[
        model.list_dof.get_elem_dof_numbers_flat(
            1 + np.arange(model.n_nodes, dtype=int), tf.DofType.HY
        ).flatten()
    ]
    assert np.allclose(HYg, 1)
    print(HXg)
    print(HYg)
    print(model.list_dof.list_dof)
    # assert False


def test_fully_cut_XQuad4n():
    nodes, elements = tf.gen_rect_Quad4n(1, 1, 2, 1)
    print(nodes)
    print(elements)
    materials = [[1, {"E": 1, "nu": 0.0, "rho": 7850}]]
    reals = [[1, {"t": 1}]]

    model = tf.XFEModel(nodes, elements, materials, reals, tip_enrichment=False)

    model.insert_crack_segment(
        np.array([-0.1, 0.25]), np.array([1.1, 0.25]), embedded=True
    )
    # print(model.level_sets)

    model.gen_list_dof(dof_per_node=tf.IS_2D)
    print("full phi_n", model.level_sets[0].phi_n)
    print("full phi_t", model.level_sets[0].phi_t)
    model.cal_global_matrices(tf.XQuad4n)

    def sel_condition(x, y, z):
        return y - 0.0

    model.gen_dirichlet_bc(sel_condition)

    def sel_condition(x, y, z):
        return y - 1.0

    def force_expression(x, y, z):
        return 0.0, 10**12 * 0.5, 0.0

    model.gen_nodal_forces(sel_condition, force_expression)
    tol = 1e-8
    for node in model.nodes:
        nid = int(node[0])
        x, y, z = map(float, node[1:4])

        if abs(sel_condition(x, y, z)) < tol:
            dof_number = model.list_dof[(nid, tf.DofType.UX)]
            model.Kg[dof_number, dof_number] += 10**12
            dof_number = model.list_dof[(nid, tf.DofType.UY)]
            model.Kg[dof_number, dof_number] += 10**12

    model.solve_static()

    Xg = model.Ug[
        model.list_dof.get_elem_dof_numbers_flat(
            1 + np.arange(model.n_nodes, dtype=int), tf.DofType.UX
        )
    ]
    assert np.allclose(Xg, 0.0)
    Yg = model.Ug[
        model.list_dof.get_elem_dof_numbers_flat(
            1 + np.arange(model.n_nodes, dtype=int), tf.DofType.UY
        )
    ]
    assert np.allclose(Yg, np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5]))
    print(Xg)
    print(Yg)

    HXg = model.Ug[
        model.list_dof.get_elem_dof_numbers_flat(
            1 + np.arange(model.n_nodes, dtype=int), tf.DofType.HX
        )
    ]
    assert np.allclose(HXg, 0.0)
    HYg = model.Ug[
        model.list_dof.get_elem_dof_numbers_flat(
            1 + np.arange(model.n_nodes, dtype=int), tf.DofType.HY
        )
    ]
    assert np.allclose(HYg, 0.5)
    print(HXg)
    print(HYg)
    # print(model.Ug)
    # assert False


def test_edge_cut_XQuad4n():
    nodes, elements = tf.gen_rect_Quad4n(1, 1, 1, 2)
    print(nodes)
    print(elements)
    materials = [[1, {"E": 1, "nu": 0.33, "rho": 7850}]]
    reals = [[1, {"t": 1}]]

    model = tf.XFEModel(nodes, elements, materials, reals, tip_enrichment=False)

    model.insert_crack_segment(
        np.array([-0.1, 0.5]), np.array([1.1, 0.5]), embedded=True
    )
    # print(model.level_sets)

    model.gen_list_dof(dof_per_node=tf.IS_2D)
    print(model.list_dof.list_dof)
    model.cal_global_matrices(tf.XQuad4n)

    def sel_condition(x, y, z):
        return y - 0.0

    bc.gen_dirichlet_bc(model, sel_condition)

    def sel_condition(x, y, z):
        return y - 1.0

    def force_expression(x, y, z):
        return 0.0, 10**9 * 1, 0.0

    model.gen_nodal_forces(sel_condition, force_expression)
    tol = 1e-8
    for node in model.nodes:
        nid = int(node[0])
        x, y, z = map(float, node[1:4])

        if abs(sel_condition(x, y, z)) < tol:
            dof_number = model.list_dof[(nid, tf.DofType.UX)]
            model.Kg[dof_number, dof_number] += 10**9
            dof_number = model.list_dof[(nid, tf.DofType.UY)]
            model.Kg[dof_number, dof_number] += 10**9

    model.solve_static()
    Xg = model.Ug[
        model.list_dof.get_elem_dof_numbers(
            1 + np.arange(model.n_nodes, dtype=int), tf.DofType.UX
        ).flatten()
    ]
    assert np.allclose(Xg, 0.0)
    Yg = model.Ug[
        model.list_dof.get_elem_dof_numbers(
            1 + np.arange(model.n_nodes, dtype=int), tf.DofType.UY
        ).flatten()
    ]
    assert np.allclose(Yg, np.array([0.0, 0.0, 0.5, 0.5, 1.0, 1.0]))
    print(Xg)
    print(Yg)

    HXg = model.Ug[
        model.list_dof.get_elem_dof_numbers_flat(
            1 + np.arange(model.n_nodes, dtype=int), tf.DofType.HX
        ).flatten()
    ]
    assert np.allclose(HXg, 0.0)
    HYg = model.Ug[
        model.list_dof.get_elem_dof_numbers_flat(
            1 + np.arange(model.n_nodes, dtype=int), tf.DofType.HY
        ).flatten()
    ]
    assert np.allclose(HYg, 1)
    print(HXg)
    print(HYg)


def test_element_edge_edge_crack_no_tip_XQuad4n():
    nodes = np.array(
        [
            [1, 0, 0, 0],
            [2, 1, 0, 0],
            [3, 2, 0, 0],
            [4, 0, 1, 0],
            [5, 0, 1, 0],
            [6, 1, 1, 0],
            [7, 2, 1, 0],
            [8, 0, 2, 0],
            [9, 1, 2, 0],
            [10, 2, 2, 0],
        ]
    )
    elements = [
        [1, "Quad4n", 1, 1, (1, 2, 6, 4)],
        [2, "Quad4n", 1, 1, (2, 3, 7, 6)],
        [3, "Quad4n", 1, 1, (5, 6, 9, 8)],
        [4, "Quad4n", 1, 1, (6, 7, 10, 9)],
    ]

    materials = [[1, {"E": 1, "nu": 0.3, "rho": 7850}]]
    reals = [[1, {"t": 1}]]
    model = tf.FEModel(nodes, elements, materials, reals)
    model.gen_list_dof(dof_per_node=tf.IS_2D)
    model.cal_global_matrices(tf.Quad4n)
    fix_dofs = []
    fix_dofs.append(model.list_dof[(1, tf.DofType.UX)])
    fix_dofs.append(model.list_dof[(1, tf.DofType.UY)])
    fix_dofs.append(model.list_dof[(2, tf.DofType.UY)])
    fix_dofs.append(model.list_dof[(3, tf.DofType.UY)])
    model.gen_P(fix_dofs)

    def sel_condition(x, y, z):
        return y - 2

    def force_expression(x, y, z):
        return 0.0, 1, 0.0

    model.gen_nodal_forces(sel_condition, force_expression)
    model.solve_static()
    Ug1 = model.Ug[
        model.list_dof.get_elem_dof_numbers(
            1 + np.arange(model.n_nodes, dtype=int), tf.IS_2D
        ).flatten()
    ]
    model.compute_quad4n_nodal_stresses()
    print(Ug1.reshape((-1, 2)))
    # model.Ug *= 1e5
    # model.show(
    #     node_size=10,
    #     nbc_size=15,
    #     load_size=(0.8, 0.15),
    #     Ug=0.1 * Ug1,
    #     node_stress=stress[:, 1],
    #     colorbar_title="s_yy",
    # )
    nodes, elements = tf.gen_rect_Quad4n(2, 2, 2, 2)
    materials = [[1, {"E": 1, "nu": 0.3, "rho": 7850}]]
    reals = [[1, {"t": 1}]]
    model = tf.XFEModel(
        nodes, elements, materials, reals, tip_enrichment=False, geometrical_range=0.0
    )
    model.insert_crack_segment(
        np.array([-0.1, 1.0]), np.array([1.1, 1.0]), embedded=False
    )
    model.gen_list_dof(dof_per_node=tf.IS_2D)
    model.cal_global_matrices(tf.XQuad4n)
    fix_dofs = []
    fix_dofs.append(model.list_dof[(1, tf.DofType.UX)])
    fix_dofs.append(model.list_dof[(1, tf.DofType.UY)])
    fix_dofs.append(model.list_dof[(2, tf.DofType.UY)])
    fix_dofs.append(model.list_dof[(3, tf.DofType.UY)])
    model.gen_P(fix_dofs)

    def sel_condition(x, y, z):
        return y - 2

    def force_expression(x, y, z):
        return 0.0, 1, 0.0

    model.gen_nodal_forces(sel_condition, force_expression)
    model.solve_static()
    Ug2 = model.Ug[
        model.list_dof.get_elem_dof_numbers(
            1 + np.arange(model.n_nodes, dtype=int), tf.IS_2D
        ).flatten()
    ]
    assert np.all(
        np.isclose(
            Ug1.reshape((-1, 2))[[0, 1, 2, -1, -2, -3], :],
            Ug2.reshape((-1, 2))[[0, 1, 2, -1, -2, -3], :],
            atol=1e-13,
        ),
    )


# def test_abaqus_edge_cut_small_mesh():
#     nodes, elements = tf.gen_rect_Quad4n(2, 2, 3, 3)
#     print(nodes)
#     print(elements)
#     materials = [[1, {"E": 1, "nu": 0.3, "rho": 7850}]]
#     reals = [[1, {"t": 1}]]
#
#     model = tf.XFEModel(
#         nodes, elements, materials, reals, tip_enrichment=True, geometrical_range=0.0
#     )
#
#     model.insert_crack_segment(
#         np.array([-0.1, 1.0]), np.array([1.0, 1.0]), embedded=False
#     )
#     # print(model.level_sets)
#
#     model.gen_list_dof(dof_per_node=tf.IS_2D)
#     model.cal_global_matrices(tf.XQuad4n)
#
#     fix_dofs = []
#     fix_dofs.append(model.list_dof[(1, tf.DofType.UX)])
#     fix_dofs.append(model.list_dof[(1, tf.DofType.UY)])
#     fix_dofs.append(model.list_dof[(2, tf.DofType.UY)])
#     fix_dofs.append(model.list_dof[(3, tf.DofType.UY)])
#     fix_dofs.append(model.list_dof[(4, tf.DofType.UY)])
#     model.gen_P(fix_dofs)
#
#     def sel_condition(x, y, z):
#         return y - 2
#
#     def force_expression(x, y, z):
#         return 0.0, 2 / 3, 0.0
#
#     model.gen_nodal_forces(sel_condition, force_expression)
#     model.Fg[-1] = 2 / 6
#     model.Fg[-7] = 2 / 6
#
#     model.solve_static()
#     Ug = model.Ug[
#         model.list_dof.get_elem_dof_numbers(
#             1 + np.arange(model.n_nodes, dtype=int), tf.IS_2D
#         ).flatten()
#     ]
#     print(Ug)
#
#     # model.show(
#     #     node_size=10,
#     #     nbc_size=15,
#     #     load_size=(0.8, 0.15),
#     #     Ug=0.1 * Ug,
#     #     node_stress=None,
#     #     colorbar_title="s_yy",
#     # # )
#     # model.Ug *= 0.02
#     # mesh1 = my_build_Quad4n(model).cast_to_unstructured_grid()
#     # ghosts = np.argwhere(mesh1["is_cut"] > 0)
#     # mesh1.remove_cells(ghosts, inplace=True)
#     # mesh2 = build_XQuad4n(model)
#     # blocks = pv.MultiBlock([mesh1, mesh2])
#     # blocks.plot(show_edges=True, color="lightblue")
#     assert False
#
#
# def test_abaqus_edge_cut_large_mesh():
#     nodes, elements = tf.gen_rect_Quad4n(2, 2, 15, 15)
#     print(nodes)
#     print(elements)
#     materials = [[1, {"E": 1, "nu": 0.3, "rho": 7850}]]
#     reals = [[1, {"t": 1}]]
#
#     model = tf.XFEModel(
#         nodes, elements, materials, reals, tip_enrichment=True, geometrical_range=0.0
#     )
#
#     model.insert_crack_segment(
#         np.array([-0.1, 1.0]), np.array([1.0, 1.0]), embedded=False
#     )
#     # print(model.level_sets)
#
#     model.gen_list_dof(dof_per_node=tf.IS_2D)
#     model.cal_global_matrices(tf.XQuad4n)
#
#     fix_dofs = []
#     fix_dofs.append(model.list_dof[(1, tf.DofType.UX)])
#     fix_dofs.append(model.list_dof[(1, tf.DofType.UY)])
#     fix_dofs.append(model.list_dof[(2, tf.DofType.UY)])
#     fix_dofs.append(model.list_dof[(3, tf.DofType.UY)])
#     fix_dofs.append(model.list_dof[(4, tf.DofType.UY)])
#     fix_dofs.append(model.list_dof[(5, tf.DofType.UY)])
#     fix_dofs.append(model.list_dof[(6, tf.DofType.UY)])
#     fix_dofs.append(model.list_dof[(7, tf.DofType.UY)])
#     fix_dofs.append(model.list_dof[(8, tf.DofType.UY)])
#     fix_dofs.append(model.list_dof[(9, tf.DofType.UY)])
#     fix_dofs.append(model.list_dof[(10, tf.DofType.UY)])
#     fix_dofs.append(model.list_dof[(11, tf.DofType.UY)])
#     fix_dofs.append(model.list_dof[(12, tf.DofType.UY)])
#     fix_dofs.append(model.list_dof[(13, tf.DofType.UY)])
#     fix_dofs.append(model.list_dof[(14, tf.DofType.UY)])
#     fix_dofs.append(model.list_dof[(15, tf.DofType.UY)])
#     fix_dofs.append(model.list_dof[(16, tf.DofType.UY)])
#     model.gen_P(fix_dofs)
#
#     def sel_condition(x, y, z):
#         return y - 2
#
#     def force_expression(x, y, z):
#         return 0.0, 2 / 15, 0.0
#
#     model.gen_nodal_forces(sel_condition, force_expression)
#     model.Fg[-1] = 2 / 30
#     model.Fg[-31] = 2 / 30
#
#     model.solve_static()
#     Ug = model.Ug[
#         model.list_dof.get_elem_dof_numbers(
#             1 + np.arange(model.n_nodes, dtype=int), tf.IS_2D
#         ).flatten()
#     ]
#     print(Ug.reshape((-1, 2)))
#
#     # model.show(
#     #     node_size=10,
#     #     nbc_size=15,
#     #     load_size=(0.8, 0.15),
#     #     Ug=0.1 * Ug,
#     #     node_stress=None,
#     #     colorbar_title="s_yy",
#     # )
#     model.Ug *= 0.01
#     mesh1 = my_build_Quad4n(model).cast_to_unstructured_grid()
#     ghosts = np.argwhere(mesh1["is_cut"] > 0)
#     mesh1.remove_cells(ghosts, inplace=True)
#     mesh2 = build_XQuad4n(model)
#     blocks = pv.MultiBlock([mesh1, mesh2])
#     blocks.plot(show_edges=True, color="lightblue")
#     assert False


# def test_moose_edge_crack_2d():
#     nodes, elements = tf.gen_rect_Quad4n(1, 2, 5, 9)
#     print(nodes)
#     print(elements)
#     materials = [[1, {"E": 1e6, "nu": 0.3, "rho": 7850}]]
#     reals = [[1, {"t": 1}]]
#
#     model = tf.XFEModel(
#         nodes, elements, materials, reals, tip_enrichment=False, geometrical_range=0.0
#     )
#
#     model.insert_crack_segment(
#         np.array([-0.1, 1.0]), np.array([0.5, 1.0]), embedded=False
#     )
#     # print(model.level_sets)
#
#     model.gen_list_dof(dof_per_node=tf.IS_2D)
#     model.cal_global_matrices(tf.XQuad4n)
#
#     fix_dofs = []
#     fix_dofs.append(model.list_dof[(6, tf.DofType.UX)])
#     fix_dofs.append(model.list_dof[(6, tf.DofType.UY)])
#     fix_dofs.append(model.list_dof[(60, tf.DofType.UX)])
#     model.gen_P(fix_dofs)
#
#     def sel_condition(x, y, z):
#         return y - 2
#
#     def force_expression(x, y, z):
#         return 0.0, 1 / 5, 0.0
#
#     model.gen_nodal_forces(sel_condition, force_expression)
#
#     def sel_condition(x, y, z):
#         return y
#
#     def force_expression(x, y, z):
#         return 0.0, -1 / 5, 0.0
#
#     model.Fg[1] -= 1 / 5 / 2
#     model.Fg[11] -= 1 / 5 / 2
#     model.Fg[-1] = 1 / 5 / 2
#     model.Fg[-11] = 1 / 5 / 2
#
#     model.gen_nodal_forces(sel_condition, force_expression, reset=False)
#
#     model.solve_static()
#     U = model.Ug[
#         model.list_dof.get_elem_dof_numbers_flat(
#             1 + np.arange(model.n_nodes, dtype=int), tf.IS_2D
#         )
#     ]
#
#     Xg = model.Ug[
#         model.list_dof.get_elem_dof_numbers_flat(
#             1 + np.arange(model.n_nodes, dtype=int), tf.DofType.UX
#         )
#     ]
#     Yg = model.Ug[
#         model.list_dof.get_elem_dof_numbers_flat(
#             1 + np.arange(model.n_nodes, dtype=int), tf.DofType.UY
#         )
#     ]
#     print(U.reshape((-1, 2)))
#     # Ug = model.Ug[
#     #     model.list_dof.get_elem_dof_numbers(
#     #         1 + np.arange(model.n_nodes, dtype=int), tf.IS_2D
#     #     ).flatten()
#     # ]
#     model.Ug *= 1e4
#     mesh1 = my_build_Quad4n(model).cast_to_unstructured_grid()
#     ghosts = np.argwhere(mesh1["is_cut"] > 0)
#     mesh1.remove_cells(ghosts, inplace=True)
#     mesh2 = build_XQuad4n(model)
#     blocks = pv.MultiBlock([mesh1, mesh2])
#     blocks.plot(show_edges=True, color="lightblue")
#     # model.show(
#     #     node_size=10,
#     #     nbc_size=15,
#     #     load_size=(0.8, 0.15),
#     #     Ug=1e4 * Ug,
#     #     node_stress=None,
#     #     colorbar_title="s_yy",
#     # )
#     # print(Yg)
#     assert False
