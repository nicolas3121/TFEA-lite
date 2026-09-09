import numpy as np
import pyvista as pv

import tfealite as tf
from tfealite.core.level_set import LevelSet
from tfealite.core.surfaces import (
    geomdl_to_NdBSplines,
    # init_half_coin_crack_geomdl,
    init_crack_plane_geomdl,
)
from tfealite.visualization.build_mesh import (
    build_XTetr4n,
    my_build_Tetr4n,
)

# %% Model creation
nodes, elements = tf.gen_rect_Tetr4n(L=1.0, H=1.0, D=1.0, nx=20, ny=20, nz=20)
materials = [[1, {"E": 2e11, "nu": 0.33, "rho": 7850}]]
reals = [[1, {}]]
model = tf.XFEModel(
    nodes,
    elements,
    materials,
    reals,
    tip_enrichment=True,
    geometrical_range=0.13,
    corrected=True,
)
# model.show()


theta = np.pi / 7
# rotation = np.array(
#     [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
# )
# geomdl_crack = init_half_coin_crack_geomdl(0.3, translation=np.array([0.5, 0, 0.43]))
geomdl_crack = init_crack_plane_geomdl(
    2, 0.7, translation=np.array([0.5, 0.5, 0.53]), embedded=True
)
scipy_crack = geomdl_to_NdBSplines(geomdl_crack)

ls = LevelSet()
ls.gen_from_ndbsplines(nodes, scipy_crack, 0.05, 0.13)

model.level_sets.append(ls)

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

mult = 1e4
# model.Ug = np.zeros(len(model.list_dof))
mesh1 = my_build_Tetr4n(model, mult=mult).cast_to_unstructured_grid()
ghosts = np.argwhere(mesh1["is_enriched"] > 0)
mesh1.remove_cells(ghosts, inplace=True)
mesh2 = build_XTetr4n(model, mult=mult)
blocks = pv.MultiBlock([mesh1, mesh2])
blocks.plot(show_edges=True, color="lightblue")
