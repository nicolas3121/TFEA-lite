import numpy as np

import tfealite as tf


def test_single_quad():
    nodes, elements = tf.gen_rect_Quad4n(1, 1, 1, 2)
    materials = [[1, {"E": 1, "nu": 0.3, "rho": 7850}]]
    reals = [[1, {"t": 1}]]
    model = tf.FEModel(nodes, elements, materials, reals)
    model.gen_list_dof(dof_per_node=tf.IS_2D)
    model.cal_global_matrices({"Quad4n": tf.XQuad4n})
    fix_dofs = []
    fix_dofs.append(model.list_dof[(1, tf.DofType.UX)])
    fix_dofs.append(model.list_dof[(1, tf.DofType.UY)])
    fix_dofs.append(model.list_dof[(2, tf.DofType.UY)])
    model.gen_P(fix_dofs)

    def sel_condition(x, y, z):
        return y - 1

    def force_expression(x, y, z):
        return 0.0, 1 / 2, 0.0

    model.gen_nodal_forces(sel_condition, force_expression)
    model.solve_static()
    Ug = model.Ug[
        model.list_dof.get_elem_dof_numbers(
            1 + np.arange(model.n_nodes, dtype=int), tf.IS_2D
        ).flatten()
    ]
    model.compute_quad4n_nodal_stresses()
    print(Ug.reshape((-1, 2)))
    print(model.Kg)
    # model.Ug *= 1e5
    # model.show(
    #     node_size=10,
    #     nbc_size=15,
    #     load_size=(0.8, 0.15),
    #     Ug=Ug,
    #     node_stress=stress[:, 1],
    #     colorbar_title="s_yy",
    # )
    answer = np.array(
        [
            [0.00000000e00, 0.00000000e00],
            [-3.00000000e-01, 0.00000000e00],
            [-3.60822483e-16, 5.00000000e-01],
            [-3.00000000e-01, 5.00000000e-01],
            [-9.85461382e-16, 1.00000000e00],
            [-3.00000000e-01, 1.00000000e00],
        ],
    ).flatten()
    assert np.all(np.isclose(Ug, answer))


def test_edge_crack():
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
    model.cal_global_matrices({"Quad4n": tf.XQuad4n})
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
    model.cal_global_matrices({"Quad4n": tf.XQuad4n})
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
