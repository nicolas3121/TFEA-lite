import tfealite as tf
import numpy as np

nodes, elements = tf.gen_rect_Tri3n(1, 1, 2, 2)
materials = [[1, {"E": 1, "nu": 0.33, "rho": 7850}]]
reals = [[1, {"t": 1}]]
model = tf.XFEModel(nodes, elements, materials, reals)
model.insert_crack_segment(np.array([0, 0.25]), np.array([0.51, 0.25]))

model.gen_list_dof(dof_per_node=tf.IS_2D)
model.cal_global_matrices(tf.XTri3n)

print(model.list_dof)


def sel_condition(x, y, z):
    return y - 0.0


model.gen_dirichlet_bc(sel_condition)


def sel_condition(x, y, z):
    return y - 1.0


def force_expression(x, y, z):
    return 0.0, 1, 0.0


model.gen_nodal_forces(sel_condition, force_expression)


model.solve_static()
Ug = model.Ug[
    model.list_dof.get_elem_dof_numbers(
        1 + np.arange(model.n_nodes, dtype=int), tf.IS_2D
    ).flatten()
]

print()
model.show(
    node_size=10,
    nbc_size=15,
    load_size=(0.8, 0.15),
    Ug=0.01 * Ug,
    node_stress=None,
    colorbar_title="s_yy",
)
