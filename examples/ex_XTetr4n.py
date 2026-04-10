import tfealite as tf
from tfealite.visualization.build_mesh import (
    my_build_Tetr4n,
    build_XTetr4n,
)
import numpy as np
import pyvista as pv

# %% Model creation
nodes, elements = tf.gen_rect_Tetr4n(L=1.0, H=1.0, D=1.0, nx=10, ny=10, nz=10)
materials = [[1, {"E": 2e11, "nu": 0.33, "rho": 7850}]]
reals = [[1, {}]]
model = tf.XFEModel(
    nodes,
    elements,
    materials,
    reals,
    tip_enrichment=True,
    geometrical_range=0.2,
    corrected=True,
)
p1 = np.array([-0.1, 0.0, 0.5])
p2 = np.array([0.5, 0.0, 0.5])
p3 = np.array([0.5, 1.3, 0.5])
model.insert_planar_crack_segment(p1, p2, p3, embedded=False)
model.gen_list_dof(dof_per_node=tf.IS_3D)
model.cal_global_matrices(tf.XTetr4n)


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


model.solve_static()
# model.Ug = np.zeros_like(model.Ug)

mult = 5e5
# model.Ug = np.zeros(len(model.list_dof))
mesh1 = my_build_Tetr4n(model, mult=mult).cast_to_unstructured_grid()
ghosts = np.argwhere(mesh1["is_enriched"] > 0)
mesh1.remove_cells(ghosts, inplace=True)
mesh2 = build_XTetr4n(model, mult=mult)
blocks = pv.MultiBlock([mesh1, mesh2])
blocks.plot(show_edges=True, color="lightblue")
# vm_1 = mesh1.point_data["von_mises"]
# vm_2 = mesh2.point_data["von_mises"]
# all_vm = np.concatenate([vm_1, vm_2])
# v_max = np.percentile(all_vm, 90)
# pl = pv.Plotter()
# pl.add_mesh(
#     blocks,
#     scalars="von_mises",  # The exact string key in your point_data dict
#     cmap="turbo",  # A great colormap for stress fields (or use "jet", "viridis")
#     show_edges=True,  # Shows the mesh grid (including your sub-triangulations!)
#     clim=[0, v_max],
#     scalar_bar_args={"title": "Von Mises Stress"},
# )
# pl.show()
