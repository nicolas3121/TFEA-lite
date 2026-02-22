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
