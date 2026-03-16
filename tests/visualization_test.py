import tfealite as tf
from tfealite.visualization.build_mesh import (
    my_build_Quad4n,
    build_XQuad4n,
)
from tfealite.core.level_set import CutType
import numpy as np
import pyvista as pv


# def test_1():
#     nodes, elements = tf.gen_rect_Quad4n(1, 1, 2, 1)
#     print(nodes)
#     print(elements)
#     materials = [[1, {"E": 1, "nu": 0.0, "rho": 7850}]]
#     reals = [[1, {"t": 1}]]
#
#     model = tf.XFEModel(nodes, elements, materials, reals, tip_enrichment=False)
#
#     model.insert_crack_segment(
#         np.array([-0.1, 0.6]), np.array([1.1, 0.3]), embedded=True
#     )
#     # print(model.level_sets)
#
#     model.gen_list_dof(dof_per_node=tf.IS_2D)
#     # print("full phi_n", model.level_sets[0].phi_n)
#     # print("full phi_t", model.level_sets[0].phi_t)
#     model.cal_global_matrices(tf.XQuad4n)
#
#     def sel_condition(x, y, z):
#         return y - 0.0
#
#     model.gen_dirichlet_bc(sel_condition)
#
#     def sel_condition(x, y, z):
#         return y - 1.0
#
#     def force_expression(x, y, z):
#         return 0.0, 10**12 * 0.5, 0.0
#
#     model.gen_nodal_forces(sel_condition, force_expression)
#     tol = 1e-8
#     for node in model.nodes:
#         nid = int(node[0])
#         x, y, z = map(float, node[1:4])
#
#         if abs(sel_condition(x, y, z)) < tol:
#             dof_number = model.list_dof[(nid, tf.DofType.UX)]
#             model.Kg[dof_number, dof_number] += 10**12
#             dof_number = model.list_dof[(nid, tf.DofType.UY)]
#             model.Kg[dof_number, dof_number] += 10**12
#
#     model.solve_static()
#     XQuad4n_split_mesh2(model, model.elements)
#     assert False
#


# def test_2():
#     nodes, elements = tf.gen_rect_Quad4n(1, 1, 1, 1)
#     materials = [[1, {"E": 1, "nu": 0.0, "rho": 7850}]]
#     reals = [[1, {"t": 1}]]
#
#     model = tf.XFEModel(nodes, elements, materials, reals, tip_enrichment=True)
#
#     model.insert_crack_segment(
#         np.array([-0.1, 0.5]), np.array([0.5, 0.5]), embedded=False
#     )
#     # print(model.level_sets)
#
#     model.gen_list_dof(dof_per_node=tf.IS_2D)
#     # print("full phi_n", model.level_sets[0].phi_n)
#     # print("full phi_t", model.level_sets[0].phi_t)
#     model.cal_global_matrices(tf.XQuad4n)
#
#     def sel_condition(x, y, z):
#         return y - 0.0
#
#     model.gen_dirichlet_bc(sel_condition)
#
#     def sel_condition(x, y, z):
#         return y - 1.0
#
#     def force_expression(x, y, z):
#         return 0.0, 1, 0.0
#
#     model.gen_nodal_forces(sel_condition, force_expression)
#     # tol = 1e-8
#     # for node in model.nodes:
#     #     nid = int(node[0])
#     #     x, y, z = map(float, node[1:4])
#     #
#     #     if abs(sel_condition(x, y, z)) < tol:
#     #         dof_number = model.list_dof[(nid, tf.DofType.UX)]
#     #         model.Kg[dof_number, dof_number] += 10**12
#     #         dof_number = model.list_dof[(nid, tf.DofType.UY)]
#     #         model.Kg[dof_number, dof_number] += 10**12
#
#     model.solve_static()
#     model.Ug *= 0.001
#     XQuad4n_partial_split_mesh2(model, model.elements)
#     assert False
#


def test_3():
    nodes, elements = tf.gen_rect_Quad4n(1, 1, 11, 11)
    materials = [[1, {"E": 1, "nu": 0.33, "rho": 7850}]]
    reals = [[1, {"t": 1}]]
    model = tf.XFEModel(
        nodes, elements, materials, reals, tip_enrichment=True, geometrical_range=0.2
    )
    p1 = np.array([-0.1, 0.5])
    p2 = np.array([0.5, 0.5])
    model.insert_crack_segment(p1, p2, embedded=False)
    # p1 = np.array([5.1, 0.711])
    # p2 = np.array([3.71, 0.267])
    # model.insert_crack_segment(p1, p2, embedded=False)
    # p1 = np.array([5.1, 0.711])
    # p2 = np.array([3.71, 0.267])
    # model.insert_crack_segment(p1, p2, embedded=False)

    model.gen_list_dof(dof_per_node=tf.IS_2D)

    # print(model.cut_info)
    # print(model.list_dof.list_dof)
    for elem, info in model.cut_info.items():
        ls, cut_type, tip = info
        if tip == 2 and (cut_type == CutType.PARTIAL or cut_type == CutType.CUT):
            element = model.elements[elem]
            nodes = element[4]
            phi_n, phi_t = model.level_sets[0].get(nodes, 2)
            print("phi_n", phi_n)
            print("phi_t", phi_t)

    # model.cut_info.pop(8)
    # node_numbers = model.list_dof.get_elem_dof_numbers(
    #     to_delete, mask=tf.DofType.HX | tf.DofType.HY
    # ).flatten()
    model.cal_global_matrices(tf.XQuad4n)

    def sel_condition(x, y, z):
        return y - 0.0

    # bc.my_gen_dirichlet_bc(model, sel_condition, to_delete)
    model.gen_dirichlet_bc(sel_condition)

    def sel_condition(x, y, z):
        return y - 1

    def force_expression(x, y, z):
        return 0.0, 1, 0.0

    model.gen_nodal_forces(sel_condition, force_expression)

    # to_delete_dof_numbers = model.list_dof.get_elem_dof_numbers(
    #     np.array(to_delete), tf.DofType.HX | tf.DofType.HY
    # ).flatten()
    #
    # for dof_number in to_delete_dof_numbers:

    model.solve_static()
    model.Ug *= 0.001
    mesh1 = my_build_Quad4n(model).cast_to_unstructured_grid()
    ghosts = np.argwhere(mesh1["is_cut"] > 0)
    mesh1.remove_cells(ghosts, inplace=True)
    mesh2 = build_XQuad4n(model)
    blocks = pv.MultiBlock([mesh1, mesh2])
    # blocks.plot(show_edges=True, color="lightblue")
    pl = pv.Plotter()
    pl.add_mesh(blocks, color="lightblue", show_edges=True)
    pl.view_xy()
    # pl.enable_parallel_projection()
    pl.show()
    assert False


# def test_3():
#     nodes, elements = tf.gen_rect_Quad4n(5, 1, 35, 13)
#     materials = [[1, {"E": 1, "nu": 0.33, "rho": 7850}]]
#     reals = [[1, {"t": 1}]]
#     model = tf.XFEModel(
#         nodes, elements, materials, reals, tip_enrichment=True, geometrical_range=0.3
#     )
#     p1 = np.array([0.711, 0.711])
#     p2 = np.array([2.51, 0.267])
#     model.insert_crack_segment(p1, p2, embedded=True)
#     p1 = np.array([5.1, 0.711])
#     p2 = np.array([3.71, 0.267])
#     model.insert_crack_segment(p1, p2, embedded=False)
#     # p1 = np.array([5.1, 0.711])
#     # p2 = np.array([3.71, 0.267])
#     # model.insert_crack_segment(p1, p2, embedded=False)
#
#     model.gen_list_dof(dof_per_node=tf.IS_2D)
#     print(model.cut_info)
#     print(model.list_dof.list_dof)
#     # model.cut_info.pop(8)
#     # node_numbers = model.list_dof.get_elem_dof_numbers(
#     #     to_delete, mask=tf.DofType.HX | tf.DofType.HY
#     # ).flatten()
#     model.cal_global_matrices(tf.XQuad4n)
#
#     def sel_condition(x, y, z):
#         return y - 0.0
#
#     to_delete = [2, 5]
#     # bc.my_gen_dirichlet_bc(model, sel_condition, to_delete)
#     model.gen_dirichlet_bc(sel_condition)
#
#     def sel_condition(x, y, z):
#         return y - 1
#
#     def force_expression(x, y, z):
#         return 0.0, 1, 0.0
#
#     model.gen_nodal_forces(sel_condition, force_expression)
#
#     # to_delete_dof_numbers = model.list_dof.get_elem_dof_numbers(
#     #     np.array(to_delete), tf.DofType.HX | tf.DofType.HY
#     # ).flatten()
#     #
#     # for dof_number in to_delete_dof_numbers:
#
#     model.solve_static()
#     model.Ug *= 0.001
#     mesh1 = my_build_Quad4n(model).cast_to_unstructured_grid()
#     ghosts = np.argwhere(mesh1["is_cut"] > 0)
#     mesh1.remove_cells(ghosts, inplace=True)
#     mesh2 = build_XQuad4n(model)
#     blocks = pv.MultiBlock([mesh1, mesh2])
#     blocks.plot(show_edges=True, color="lightblue")
#     assert False


# def test_4():
#     nodes, elements = tf.gen_rect_Quad4n(1, 1, 4, 4)
#     materials = [[1, {"E": 1, "nu": 0.33, "rho": 7850}]]
#     reals = [[1, {"t": 1}]]
#     model = tf.XFEModel(
#         nodes, elements, materials, reals, tip_enrichment=True, geometrical_range=0.0
#     )
#     p1 = np.array([-0.1, 0.44])
#     p2 = np.array([0.75, 0.44])
#     model.insert_crack_segment(p1, p2, embedded=False)
#     # p1 = np.array([5.1, 0.711])
#     # p2 = np.array([3.71, 0.267])
#     # model.insert_crack_segment(p1, p2, embedded=False)
#
#     model.gen_list_dof(dof_per_node=IS_2D)
#     dofs1 = model.list_dof.get_elem_dof_numbers(
#         1 + np.arange(model.n_nodes, dtype=int), IS_2D
#     ).flatten()
