import numpy as np
import pyvista as pv

import tfealite as tf
from tfealite.core.sif import DisplacementCorrelationMethodSIF3D as DCMSIF3D
from tfealite.core.surfaces import geomdl_to_NdBSplines, init_crack_plane_geomdl
from tfealite.visualization.build_mesh import (
    build_XTetr4n,
    my_build_Tetr4n,
)

E_mod = 200e9
nu = 0.3
kappa = (3 - nu) / (3 + nu)
G_mod = (E_mod) / (2 * (1 + nu))
a = 0.115
W = 0.5
D = 0.02
x_elem = 33
y_elem = int(1.5 * x_elem) + 1
z_elem = 2
nodes, elements = tf.gen_rect_Tetr4n(2 * W, 1.5 * 2 * W, D, x_elem, y_elem, z_elem)
materials = [[1, {"E": E_mod, "nu": nu, "rho": 7850}]]
reals = [[1, {"t": 1}]]
model = tf.XFEModel(
    nodes,
    elements,
    materials,
    reals,
    tip_enrichment=True,
    geometrical_range=0.0,
    corrected=True,
)
c90 = np.cos(np.pi / 2)
s90 = np.sin(np.pi / 2)
rotation_x = np.array([[1, 0, 0], [0, c90, -s90], [0, s90, c90]])
rotation_y = np.array([[c90, 0, s90], [0, 1, 0], [-s90, 0, c90]])
rotation = rotation_y @ rotation_x
rotation = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
translation = np.array([0.5, 0.75, D / 2])
crack = init_crack_plane_geomdl(
    D,
    2 * a,
    rotation=rotation,
    translation=translation,
    embedded=True,
)
scipy_crack = geomdl_to_NdBSplines(crack)
for _, _, surf in scipy_crack:
    print(surf([1, 0]), surf([1, 1]))

model.insert_ndbsplines_crack(scipy_crack, 2 * W / x_elem, snapping_tolerance=0.05)

model.gen_list_dof(dof_per_node=tf.IS_3D)
model.cal_global_matrices({"Tetr4n": tf.XTetr4n})


def sel_condition(x, y, z):
    return y - 0.0


model.gen_dirichlet_bc(sel_condition)


# %% Load definition
def sel_condition(x, y, z):
    return y - 1.5 * 2 * W


def force_expression(x, y, z):
    return 0.0, 1e3, 0.0


# print(model.Fg)
# assert False

model.gen_nodal_forces(sel_condition, force_expression)

model.solve_static()

mult = 1e3
# model.Ug = np.zeros(len(model.list_dof))
mesh1 = my_build_Tetr4n(model, mult=mult).cast_to_unstructured_grid()
ghosts = np.argwhere(mesh1["is_enriched"] > 0)
mesh1.remove_cells(ghosts, inplace=True)
mesh2 = build_XTetr4n(model, mult=mult)
blocks = pv.MultiBlock([mesh1, mesh2])
blocks.plot(show_edges=True, color="lightblue")

dcm = DCMSIF3D(kappa, G_mod, np.array([0.05, 0.1, 0.15]), None)
K_I, K_II, K_III = dcm.cal_sif(
    model.level_sets[0],
    model,
    model.cut_info,
    tip_index=1,
    v_tip=np.linspace(0, 1, 5),
)
print("K_I", K_I)
print("K_II", K_II)
print("K_III", K_III)
