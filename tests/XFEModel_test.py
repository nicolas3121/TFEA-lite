import tfealite as tf
import numpy as np


# imposes vertical displacement of 0.5 at top 2 nodes with a fully cut element
# should be no deformation as the two halves are not connected
# coefficient Heaviside functions should be equal to the displacement of 0.5
def test_fully_cut_XTri3n():
    nodes, elements = tf.gen_rect_Tri3n(1, 1, 1, 1)
    print(nodes)
    print(elements)
    materials = [[1, {"E": 1, "nu": 0.33, "rho": 7850}]]
    reals = [[1, {"t": 1}]]

    model = tf.XFEModel(nodes, elements, materials, reals)

    model.insert_crack_segment(np.array([-0.1, 0.6]), np.array([1.1, 0.6]))
    print(model.level_set)

    model.gen_list_dof(dof_per_node=tf.IS_2D)
    model.cal_global_matrices(tf.XTri3n, tip_enrich=False)

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
