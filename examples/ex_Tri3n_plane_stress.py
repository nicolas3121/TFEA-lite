import tfealite as tf
import numpy as np


# %% Model creation
nodes, elements = tf.gen_rect_Tri3n(L=1.0, H=1.0)
materials = [[1, {"E": 2e11, "nu": 0.33, "rho": 7850}]]
reals = [[1, {"t": 1}]]
model = tf.FEModel(nodes, elements, materials, reals)


# %% Finite element formulation
model.gen_list_dof(dof_per_node=tf.DofType.UX | tf.DofType.UY)
model.cal_global_matrices(tf.Tri3n)


# %% B.C. definition
def sel_condition(x, y, z):
    return y - 0.0


model.gen_dirichlet_bc(sel_condition)


# %% Load definition
def sel_condition(x, y, z):
    return y - 1.0


def force_expression(x, y, z):
    return 0.0, np.sin(np.pi * x / 1.0), 0.0


model.gen_nodal_forces(sel_condition, force_expression)

# %% Solve
model.solve_static()
stress = model.compute_tri3n_nodal_stresses()

# %% Visualization
model.show(
    node_size=10,
    nbc_size=15,
    load_size=(0.8, 0.15),
    Ug=1e9 * model.Ug,
    node_stress=stress[:, 1],
    colorbar_title="s_yy",
)
