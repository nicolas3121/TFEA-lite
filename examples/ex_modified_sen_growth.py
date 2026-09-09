import meshio
import numpy as np
import pyvista as pv

import tfealite as tf
from tfealite.core.sif import DisplacementCorrelationMethodSIF as DCMSIF
from tfealite.visualization.build_mesh import (
    build_XQuad4n,
    my_build_mixed_mesh,
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

h = 0.15
da = 0.15

materials = [
    [1, {"E": E_eff_plate, "nu": nu_eff, "rho": 7850}],  # Material 1: Plate
    [2, {"E": E_eff_pin, "nu": nu_eff, "rho": 7850}],  # Material 2: Stiff Pin
]

reals = [[1, {"t": 1}]]

kappa = (3 - nu) / (3 + nu)
G_mod = (E_plate) / (2 * (1 + nu))
dcm = DCMSIF(
    kappa,
    G_mod,
    4 * h + 0.1 * h * np.arange(11),
    None,
)

p1 = np.array([62.5, -0.1])
p2 = np.array([62.5, 2.5])

control_points = np.linspace(p1, p2, 12).tolist()
n = len(control_points)
k = 2

mesh = meshio.read("./modified_sen_hole_2d.msh")

vertex_cells = mesh.cells_dict.get("vertex", [])
vertex_tags = mesh.cell_data_dict.get("gmsh:physical", {}).get("vertex", [])

# Build nodes array (ID, X, Y, Z)
nodes = np.hstack([np.arange(1, mesh.points.shape[0] + 1)[:, None], mesh.points])

elements = []
elem_id = 1

# Map meshio cell string to your solver's element string
supported_elements = {"quad": "Quad4n", "triangle": "Tri3n"}

# Loop through supported types and extract them if they exist in the mesh
for cell_type, elem_string in supported_elements.items():
    if cell_type in mesh.cells_dict:
        cells = mesh.cells_dict[cell_type]

        # Safely get physical tags, default to an array of 1s if they are missing
        phys_tags = mesh.cell_data_dict.get("gmsh:physical", {}).get(
            cell_type, np.ones(len(cells), dtype=int)
        )

        for element_nodes, phys_tag in zip(cells, phys_tags):
            elements.append(
                [
                    elem_id,
                    elem_string,
                    phys_tag,
                    1,  # Real ID
                    element_nodes + 1,  # 1-based node indexing
                ]
            )
            elem_id += 1

load_node_ids = []
support_node_ids = []

for i, tag in enumerate(vertex_tags):
    if tag == 10:
        load_node_ids.append(vertex_cells[i][0] + 1)

    elif tag == 11:
        support_node_ids.append(vertex_cells[i][0] + 1)

assert len(load_node_ids) == 2, f"Expected 2 load nodes, found {len(load_node_ids)}"
assert len(support_node_ids) == 2, (
    f"Expected 2 support nodes, found {len(support_node_ids)}"
)


def bc_fn(model):
    fix_dofs = []

    fix_dofs.append(
        model.list_dof[(support_node_ids[0], tf.DofType.UX)],
    )
    fix_dofs.append(
        model.list_dof[(support_node_ids[0], tf.DofType.UY)],
    )
    fix_dofs.append(
        model.list_dof[(support_node_ids[1], tf.DofType.UY)],
    )

    model.gen_P(np.array(sorted(set(fix_dofs)), dtype=int))


def force_fn(model):
    Fg = np.zeros(len(model.list_dof), dtype=float)
    for n in load_node_ids:
        Fg[model.list_dof[(n, tf.DofType.UY)]] = -10.0
        model.Fg = Fg


def plot_fn(model, bspline, i):
    displacement_mult = 0.00
    mesh1 = my_build_mixed_mesh(
        model, mult=displacement_mult
    ).cast_to_unstructured_grid()
    ghosts = np.argwhere(mesh1["is_enriched"] > 0)
    mesh1.remove_cells(ghosts, inplace=True)

    mesh2 = build_XQuad4n(model, mult=displacement_mult)
    # mesh3 = build_XTri3n(model, mult=displacement_mult)
    blocks = pv.MultiBlock([mesh1, mesh2])

    pv.set_plot_theme("document")
    # pl = pv.Plotter()
    pl = pv.Plotter(off_screen=True, window_size=[3840, 3840 // 2])
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
    pl.view_xy()
    pl.zoom_camera("tight")
    export_filename = f"SEN_hole_{i}.png"
    pl.screenshot(export_filename, transparent_background=False)
    pl.close()


growth_controller = tf.GrowthController(
    nodes,
    elements,
    materials,
    reals,
    True,
    1.5,
    True,
    0.2,
    False,
    tf.IS_2D,
    {"Quad4n": tf.XQuad4n, "Tri3n": tf.XTri3n},
    dcm,
    calculate_growth_direction,
    da,
    control_points,
    2,
    bc_fn,
    force_fn,
    plot_fn,
)

growth_controller.run(max_iter=100)
