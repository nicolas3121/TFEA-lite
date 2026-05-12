import meshio
import numpy as np
import pyvista as pv
from scipy.interpolate import BSpline, splprep

import tfealite as tf
from tfealite.visualization.build_mesh import (
    build_XQuad4n,
    my_build_Quad4n,
)

mesh = meshio.read("./mcts_xfem_mesh.msh")

nodes = np.hstack([np.arange(1, mesh.points.shape[0] + 1)[:, None], mesh.points])


nu = 0.3
nu_eff = nu / (1.0 - nu)

# --- 1. Define Base Material (Plate) ---
E_plate = 1.0
E_eff_plate = E_plate / (1.0 - nu**2)

E_pin = 100.0 * E_plate
E_eff_pin = E_pin / (1.0 - nu**2)

materials = [
    [1, {"E": E_eff_plate, "nu": nu_eff, "rho": 7850}],  # Material 1: Plate
    [2, {"E": E_eff_pin, "nu": nu_eff, "rho": 7850}],  # Material 2: Stiff Pin
]
reals = [[1, {"t": 1}]]

quad_tags = mesh.cell_data_dict["gmsh:physical"]["quad"]

elements = []
for i, (element_nodes, phys_tag) in enumerate(zip(mesh.cells_dict["quad"], quad_tags)):
    elements.append(
        [
            i + 1,  # Element ID
            "Quad4n",  # Element Type
            phys_tag,  # Material ID (1 for Plate, 2 for Pin based on Gmsh)
            1,  # Real constant ID
            element_nodes + 1,  # Node indices (1-based)
        ]
    )


vertex_cells = mesh.cells_dict.get("vertex", [])
vertex_tags = mesh.cell_data_dict["gmsh:physical"].get("vertex", [])

node_top_id = None
node_bot_id = None

for i, tag in enumerate(vertex_tags):
    if tag == 100:
        node_top_id = vertex_cells[i][0] + 1
    elif tag == 200:
        node_bot_id = vertex_cells[i][0] + 1

assert node_top_id is not None
assert node_bot_id is not None

model = tf.XFEModel(
    nodes,
    elements,
    materials,
    reals,
    tip_enrichment=True,
    geometrical_range=1.2,
    corrected=True,
)


W = 40.0
H = 40.0

dist_opp_edge = 23.0

p1 = np.array([W - dist_opp_edge - 3, 0.0])
p2 = np.array([W - dist_opp_edge + 2.5, 0.0])

pts = np.linspace(p1, p2, 4).T
tck, u = splprep(pts, s=0, k=2)
bspline = BSpline(tck[0], np.transpose(tck[1]), tck[2])
h = 0.2
model.insert_crack_spline(bspline, embedded=False, h=h, snapping_tolerance=0.03)

model.gen_list_dof(dof_per_node=tf.IS_2D)
model.cal_global_matrices({"Quad4n": tf.XQuad4n})

fix_dofs = [
    model.list_dof[(node_bot_id, tf.DofType.UX)],
    model.list_dof[(node_bot_id, tf.DofType.UY)],
    model.list_dof[(node_top_id, tf.DofType.UX)],
]
model.gen_P(np.array(sorted(set(fix_dofs)), dtype=int))

Fg = np.zeros(len(model.list_dof), dtype=float)
Fg[model.list_dof[(node_top_id, tf.DofType.UY)]] = 1000.0
model.Fg = Fg


model.solve_static()
mult = 0.00005
# model.Ug = np.zeros(len(model.list_dof))
mesh1 = my_build_Quad4n(model, mult=mult).cast_to_unstructured_grid()
ghosts = np.argwhere(mesh1["is_enriched"] > 0)
mesh1.remove_cells(ghosts, inplace=True)
mesh2 = build_XQuad4n(model, mult=mult)
blocks = pv.MultiBlock([mesh1, mesh2])
# blocks.plot(show_edges=True, color="lightblue")

# pv.set_plot_theme("document")
pl = pv.Plotter()
pl.enable_anti_aliasing("ssaa")

vm_1 = mesh1.point_data["von_mises"]
vm_2 = mesh2.point_data["von_mises"]
all_vm = np.concatenate([vm_1, vm_2])

v_max = np.percentile(all_vm, 98)

sbar_args = {
    "title": "Von Mises Stress (Pa)",
    "color": "black",
    "font_family": "arial",
    "fmt": "%.2e",
    "title_font_size": 24,
    "label_font_size": 20,
    "vertical": True,
}

pl.add_mesh(
    blocks,
    scalars="von_mises",
    cmap="turbo",
    show_edges=True,
    lighting=False,
    clim=[0, v_max],
    show_scalar_bar=False,
    # scalar_bar_args=sbar_args,
)

pl.zoom_camera("tight")
pl.view_xy()
# export_filename = "xfem_stress_presentation.png"
# pl.screenshot(export_filename, transparent_background=False)
# pl.close()
pl.show()
