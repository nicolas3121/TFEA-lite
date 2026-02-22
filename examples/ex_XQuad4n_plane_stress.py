import tfealite as tf
import numpy as np


nodes, elements = tf.gen_rect_Quad4n(5, 1, 31, 11)
materials = [[1, {"E": 1, "nu": 0.33, "rho": 7850}]]
reals = [[1, {"t": 1}]]
model = tf.XFEModel(
    nodes, elements, materials, reals, tip_enrichment=True, geometrical_range=0.3
)
p1 = np.array([0.711, 0.711])
p2 = np.array([2.51, 0.267])
model.insert_crack_segment(p1, p2, embedded=True)
p1 = np.array([5.1, 0.711])
p2 = np.array([3.71, 0.267])
model.insert_crack_segment(p1, p2, embedded=False)
# p1 = np.array([5.1, 0.711])
# p2 = np.array([3.71, 0.267])
# model.insert_crack_segment(p1, p2, embedded=False)

model.gen_list_dof(dof_per_node=tf.IS_2D)
print(model.cut_info)
print(model.list_dof.list_dof)
# model.cut_info.pop(8)
# node_numbers = model.list_dof.get_elem_dof_numbers(
#     to_delete, mask=tf.DofType.HX | tf.DofType.HY
# ).flatten()
model.cal_global_matrices(tf.XQuad4n)


def sel_condition(x, y, z):
    return y - 0.0


to_delete = [2, 5]
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
Ug = model.Ug[
    model.list_dof.get_elem_dof_numbers(
        1 + np.arange(model.n_nodes, dtype=int), tf.IS_2D
    ).flatten()
]
print(model.tip)
# print(Ug)

# print()
model.show(
    node_size=10,
    nbc_size=15,
    load_size=(0.8, 0.15),
    Ug=0.001 * Ug,
    node_stress=None,
    colorbar_title="s_yy",
)
