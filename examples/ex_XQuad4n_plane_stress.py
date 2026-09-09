import numpy as np
import scipy as sp
from scipy.linalg import svdvals

import tfealite as tf

nodes, elements = tf.gen_rect_Quad4n(10, 10, 25, 25)
materials = [[1, {"E": 1, "nu": 0.33, "rho": 7850}]]
reals = [[1, {"t": 1}]]
model = tf.XFEModel(
    nodes,
    elements,
    materials,
    reals,
    tip_enrichment=True,
    geometrical_range=2,
    corrected=True,
)
p1 = np.array([0.0, 5])
p2 = np.array([5, 5])
model.insert_crack_segment(p1, p2, embedded=False)

model.gen_list_dof(dof_per_node=tf.IS_2D)
model.cal_global_matrices({"Quad4n": tf.XQuad4n})


def sel_condition(x, y, z):
    return y - 0.0


model.gen_dirichlet_bc(sel_condition)

# blending_nodes = (model.in_range == 0) & (model.list_dof.list_dof & BRANCH_DOFS != 0)
#
# extra_fix_dofs = model.list_dof.get_elem_dof_numbers_flat(
#     np.where(blending_nodes)[0][:2] + 1, BRANCH_4_DOFS
# )
#
# bc.my_gen_dirichlet_bc(model, sel_condition, extra_fix_dofs)


def sel_condition(x, y, z):
    return y - 10


def force_expression(x, y, z):
    return 0.0, 1, 0.0


model.gen_nodal_forces(sel_condition, force_expression)

Fg = model.Fg

Kg_bc = model.ortho_T.T @ model.Kg @ model.ortho_T
# Kg_bc = model.ortho_T.T @ model.Kg @ model.ortho_T
print("   - Kg_bc evaluated.")
# Fg_bc = model.P.T @ model.ortho_T.T @ Fg
print("   - Fg_bc evaluated.")
D = Kg_bc.diagonal()

# 2. Create the sparse diagonal scaling matrix (D^-1/2)
D_inv_sqrt = sp.sparse.diags(1.0 / np.sqrt(D))

# 3. Scale both the matrix and the load vector
Kg_scaled = D_inv_sqrt @ Kg_bc @ D_inv_sqrt
# Fg_scaled = D_inv_sqrt @ Fg_bc
print("   - Diagonal scaling applied.")
print("   - Start solving for U = inv(K)F ...")
K_dense = Kg_scaled.toarray()
singular_values = svdvals(K_dense)
# Look at the smallest 5 singular values
print("singular_values")
print(singular_values[-5:])
# Ug_scaled = sp.sparse.linalg.spsolve(Kg_scaled, Fg_scaled)
# Ug_bc = D_inv_sqrt @ Ug_scaled
# print("   - Ug_bc evaluated.")
# model.Ug = model.ortho_T @ model.P @ Ug_bc

# model.solve_static()
# Ug = model.Ug[
#     model.list_dof.get_elem_dof_numbers(
#         1 + np.arange(model.n_nodes, dtype=int), tf.IS_2D
#     ).flatten()
# ]
# print(model.tip)
# print(Ug)
#
# mult = 0.001
# # model.Ug = np.zeros(len(model.list_dof))
# mesh1 = my_build_Quad4n(model, mult=mult).cast_to_unstructured_grid()
# ghosts = np.argwhere(mesh1["is_enriched"] > 0)
# mesh1.remove_cells(ghosts, inplace=True)
# mesh2 = build_XQuad4n(model, mult=mult)
# blocks = pv.MultiBlock([mesh1, mesh2])
# pl = pv.Plotter()
# pl.add_mesh(
#     blocks,
#     color="lightblue",
#     # scalars="von_mises",  # The exact string key in your point_data dict
#     # cmap="turbo",  # A great colormap for stress fields (or use "jet", "viridis")
#     show_edges=True,  # Shows the mesh grid (including your sub-triangulations!)
# )
# pl.view_xy()
# # pl.enable_parallel_projection()
# pl.show()
