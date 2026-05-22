import tfealite as tf
import numpy as np
from scipy.linalg import svdvals
from tfealite.core.surfaces import init_crack_plane_geomdl, geomdl_to_NdBSplines

x_elem = 25

# %% Model creation
nodes, elements = tf.gen_rect_Tetr4n(L=0.2, H=1.0, D=1.0, nx=2, ny=x_elem, nz=x_elem)
materials = [[1, {"E": 2e11, "nu": 0.33, "rho": 7850}]]
reals = [[1, {}]]
model = tf.XFEModel(
    nodes,
    elements,
    materials,
    reals,
    tip_enrichment=True,
    geometrical_range=0.3,
    corrected=True,
)
translation = np.array([0.5, 0.55, 0.5])
theta = 30 / 180 * np.pi
s = np.sin(theta)
c = np.cos(theta)
# rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
# rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
crack = init_crack_plane_geomdl(
    2,
    0.5,
    # rotation=rotation,
    translation=translation,
    embedded=False,
)
scipy_crack = geomdl_to_NdBSplines(crack)

for _, _, surf in scipy_crack:
    print(surf([1, 0]), surf([1, 1]))

model.insert_ndbsplines_crack(scipy_crack, 1 / x_elem, snapping_tolerance=0.01)
model.gen_list_dof(dof_per_node=tf.IS_3D)
model.cal_global_matrices({"Tetr4n": tf.XTetr4n})


def sel_condition(x, y, z):
    return z - 0.0


model.gen_dirichlet_bc(sel_condition)
# model.show(nbc_size = 10)


# %% Load definition
def sel_condition(x, y, z):
    return z - 1.0


def force_expression(x, y, z):
    return 0.0, 0.0, 1e3


model.gen_nodal_forces(sel_condition, force_expression)

Fg = model.Fg

Kg_bc = model.Kg
print("   - Kg_bc evaluated.")
# Fg_bc = model.P.T @ model.ortho_T.T @ Fg
print("   - Fg_bc evaluated.")
# D = Kg_bc.diagonal()

# 2. Create the sparse diagonal scaling matrix (D^-1/2)
# D_inv_sqrt = sp.sparse.diags(1.0 / np.sqrt(D))

# 3. Scale both the matrix and the load vector
# Kg_scaled = D_inv_sqrt @ Kg_bc @ D_inv_sqrt
# Fg_scaled = D_inv_sqrt @ Fg_bc
print("   - Diagonal scaling applied.")
print("   - Start solving for U = inv(K)F ...")
K_dense = Kg_bc.toarray()
singular_values = svdvals(K_dense)
# Look at the smallest 5 singular values
print("singular_values")
print(singular_values)
# Ug_scaled = sp.sparse.linalg.spsolve(Kg_scaled, Fg_scaled)
# Ug_bc = D_inv_sqrt @ Ug_scaled
# print("   - Ug_bc evaluated.")
# model.Ug = model.ortho_T @ model.P @ Ug_bc
#
# # model.solve_static()
# # model.Ug = np.zeros_like(model.Ug)
#
# mult = 1e5
# # model.Ug = np.zeros(len(model.list_dof))
# mesh1 = my_build_Tetr4n(model, mult=mult).cast_to_unstructured_grid()
# ghosts = np.argwhere(mesh1["is_enriched"] > 0)
# mesh1.remove_cells(ghosts, inplace=True)
# mesh2 = build_XTetr4n(model, mult=mult)
# blocks = pv.MultiBlock([mesh1, mesh2])
# blocks.plot(show_edges=True, color="lightblue")
# # vm_1 = mesh1.point_data["von_mises"]
# # vm_2 = mesh2.point_data["von_mises"]
# # all_vm = np.concatenate([vm_1, vm_2])
# # v_max = np.percentile(all_vm, 90)
# # pl = pv.Plotter()
# # pl.add_mesh(
# #     blocks,
# #     scalars="von_mises",  # The exact string key in your point_data dict
# #     cmap="turbo",  # A great colormap for stress fields (or use "jet", "viridis")
# #     show_edges=True,  # Shows the mesh grid (including your sub-triangulations!)
# #     clim=[0, v_max],
# #     scalar_bar_args={"title": "Von Mises Stress"},
# # )
# # pl.show()
