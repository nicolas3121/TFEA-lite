import meshio
import numpy as np
from geomdl import knotvector
from scipy.interpolate import BSpline
import pyvista as pv
from tfealite.core.sif import DisplacementCorrelationMethodSIF as DCMSIF

import tfealite as tf
from tfealite.visualization.build_mesh import (
    build_XQuad4n,
    my_build_Quad4n,
)


def calculate_growth_direction(K_I, K_II):
    # Added **2 to K_II in both the numerator and denominator
    numerator = 3 * K_II**2 + np.sqrt(K_I**4 + 8 * K_I**2 * K_II**2)
    denominator = K_I**2 + 9 * K_II**2

    # Clip ensures floating point errors don't push the ratio to 1.00000001
    argument = np.clip(numerator / denominator, -1.0, 1.0)

    theta = np.sign(-K_II) * np.acos(argument)
    return theta


nu = 0.29
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

W = 40.0
H = 40.0

dist_opp_edge = 23.0

p1 = np.array([W - dist_opp_edge - 3, 0.0])
p2 = np.array([W - dist_opp_edge + 3, 0.0])

control_points = np.linspace(p1, p2, 12).tolist()
n = len(control_points)
k = 2
knots = knotvector.generate(k, n)
bspline = BSpline(knots, np.array(control_points), k)

h = 0.2
da = 0.15

kappa = (3 - nu) / (3 + nu)
G_mod = (E_plate) / (2 * (1 + nu))
dcm = DCMSIF(
    kappa,
    G_mod,
    4 * h + 0.1 * h * np.arange(11),
    None,
)

mesh = meshio.read("./mcts_xfem_mesh.msh")

nodes = np.hstack([np.arange(1, mesh.points.shape[0] + 1)[:, None], mesh.points])

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
vertex_tags = mesh.cell_data_dict.get("gmsh:physical", {}).get("vertex", [])

node_top_id = None
node_bot_id = None

for i, tag in enumerate(vertex_tags):
    if tag == 100:
        node_top_id = vertex_cells[i][0] + 1
    elif tag == 200:
        node_bot_id = vertex_cells[i][0] + 1

assert node_top_id is not None
assert node_bot_id is not None

# --- Define Callbacks for GrowthController ---


def bc_fn(model):
    fix_dofs = [
        model.list_dof[(node_bot_id, tf.DofType.UX)],
        model.list_dof[(node_bot_id, tf.DofType.UY)],
        model.list_dof[(node_top_id, tf.DofType.UX)],
    ]
    model.gen_P(np.array(sorted(set(fix_dofs)), dtype=int))


def force_fn(model):
    Fg = np.zeros(len(model.list_dof), dtype=float)
    Fg[model.list_dof[(node_top_id, tf.DofType.UY)]] = 1000.0
    model.Fg = Fg


def plot_fn(model, bspline, i):
    displacement_mult = 0
    mesh1 = my_build_Quad4n(model, mult=displacement_mult).cast_to_unstructured_grid()
    ghosts = np.argwhere(mesh1["is_enriched"] > 0)
    mesh1.remove_cells(ghosts, inplace=True)

    mesh2 = build_XQuad4n(model, mult=displacement_mult)
    blocks = pv.MultiBlock([mesh1, mesh2])

    pv.set_plot_theme("document")
    # pl = pv.Plotter()
    pl = pv.Plotter(off_screen=True, window_size=[3840, 3840])
    pl.enable_anti_aliasing("ssaa")

    vm_1 = mesh1.point_data["von_mises"]
    vm_2 = mesh2.point_data["von_mises"]
    all_vm = np.concatenate([vm_1, vm_2])

    v_max = np.percentile(all_vm, 98)

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

    u_new = np.linspace(0, 1, 1000)
    spline_pts_2d = bspline(u_new)  # Output shape is (100, 2)
    z_coords = np.zeros((spline_pts_2d.shape[0], 1))
    spline_pts_3d = np.hstack((spline_pts_2d, z_coords))
    spline_pv_mesh = pv.lines_from_points(spline_pts_3d)
    pl.add_mesh(spline_pv_mesh, color="red", line_width=4, label="Crack Spline")
    pl.zoom_camera("tight")
    pl.view_xy()
    export_filename = f"CT_hole_{i}.png"
    pl.screenshot(export_filename, transparent_background=False)
    pl.close()


# --- Initialize and Run GrowthController ---

growth_controller = tf.GrowthController(
    nodes,
    elements,
    materials,
    reals,
    True,  # tip_enrichment
    2,  # geometrical_range
    False,  # corrected
    h,  # h
    False,  # embedded
    tf.IS_2D,  # dof_per_node
    {"Quad4n": tf.XQuad4n},  # element mapping
    dcm,  # sif calculation method
    calculate_growth_direction,  # direction function
    da,  # crack increment
    control_points,  # crack initial control points
    k,  # spline degree
    bc_fn,  # boundary condition function
    force_fn,  # force application function
    plot_fn,  # plotting function
)

growth_controller.run(max_iter=100)
