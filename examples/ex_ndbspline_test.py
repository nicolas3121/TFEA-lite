import tfealite as tf
from tfealite.core.surfaces import (
    init_half_coin_crack_geomdl,
    geomdl_to_NdBSplines,
)
from tfealite.core.level_set import LevelSet
import numpy as np
from tfealite.visualization.build_mesh import (
    my_build_Tetr4n,
    build_XTetr4n,
)
import pyvista as pv

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
    geometrical_range=0.1,
    corrected=True,
)
# model.show()


theta = np.pi / 7
rotation = np.array(
    [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
)
geomdl_crack = init_half_coin_crack_geomdl(
    0.3, rotation=rotation, translation=np.array([0.5, 0, 0.43])
)
scipy_crack = geomdl_to_NdBSplines(geomdl_crack)
test_coords = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
print("test_coords", scipy_crack[0](test_coords, nu=(1, 0)))
test_coords = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
print("test_coords", scipy_crack[0](test_coords, nu=(0, 1)))

ls = LevelSet()
ls.gen_from_ndbsplines(nodes, scipy_crack, 0.4, 0.0)

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

mult = 1e5
# model.Ug = np.zeros(len(model.list_dof))
mesh1 = my_build_Tetr4n(model, mult=mult).cast_to_unstructured_grid()
ghosts = np.argwhere(mesh1["is_enriched"] > 0)
mesh1.remove_cells(ghosts, inplace=True)
mesh2 = build_XTetr4n(model, mult=mult)
blocks = pv.MultiBlock([mesh1, mesh2])
blocks.plot(show_edges=True, color="lightblue")
