import tfealite as tf

# %% Model creation
nodes, elements = tf.gen_ibeam_Tetr4n(L=1.0, h=0.3, bf=0.15, tw=0.02, tf=0.02)
materials = [[1, {"E": 2e11, "nu": 0.33, "rho": 7850}]]
reals = [[1, {}]]
model = tf.FEModel(nodes, elements, materials, reals)
# model.show()

# %% Finite element formulation
model.gen_list_dof(dof_per_node=["ux", "uy", "uz"])
model.cal_global_matrices()


# %% B.C. definition
def sel_condition(x, y, z):
    return x - 0.0


model.gen_dirichlet_bc(sel_condition)
# model.show(nbc_size = 10)


# %% Load definition
def sel_condition(x, y, z):
    return x - 1.0


def force_expression(x, y, z):
    return 0.0, 0.0, -1.0


model.gen_nodal_forces(sel_condition, force_expression)
# model.show(load_size = (0.50, 0.05))

# %% Solve
model.solve_static()
e_stress = model.cal_Tetr4n_stresses()
n_stress = model.eval_node_average(e_stress[:, 0])

# %% Visualization
model.show(
    nbc_size=10,
    load_size=(0.5, 0.05),
    Ug=model.Ug * 1e5,
    node_stress=n_stress,
    colorbar_title="s_xx",
)
