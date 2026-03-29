import tfealite as tf
from tfealite.visualization.build_mesh import (
    my_build_Quad4n,
    build_XQuad4n,
)
import numpy as np
import pyvista as pv


def clean_and_average(mesh, decimals=5):
    rounded_points = np.round(mesh.points, decimals=decimals)
    unique_points, inv_idx = np.unique(rounded_points, axis=0, return_inverse=True)
    num_merged = len(unique_points)

    new_point_data = {}
    for key, array in mesh.point_data.items():
        if array.ndim == 1:
            new_array = np.zeros(num_merged)
            counts = np.zeros(num_merged, dtype=int)
        else:
            new_array = np.zeros((num_merged, array.shape[1]))
            counts = np.zeros(num_merged, dtype=int)

        np.add.at(new_array, inv_idx, array)
        np.add.at(counts, inv_idx, 1)

        if array.ndim > 1:
            new_point_data[key] = new_array / counts[:, None]
        else:
            new_point_data[key] = new_array / counts

    faces = mesh.faces
    new_faces = []
    idx = 0
    while idx < len(faces):
        n_pts = faces[idx]
        new_faces.append(n_pts)
        for j in range(n_pts):
            new_faces.append(inv_idx[faces[idx + 1 + j]])
        idx += n_pts + 1

    merged_mesh = pv.PolyData(unique_points, new_faces)
    for key, array in new_point_data.items():
        merged_mesh.point_data[key] = array

    return merged_mesh


materials = [[1, {"E": 2e11, "nu": 0.33, "rho": 7850}]]
reals = [[1, {"t": 1}]]
nodes, elements = tf.gen_rect_Quad4n(1, 1, 77, 77)
model = tf.XFEModel(
    nodes,
    elements,
    materials,
    reals,
    tip_enrichment=True,
    geometrical_range=0.1,
    corrected=True,
)
p1 = np.array([-0.1, 0.5])
p2 = np.array([0.48, 0.5])
model.insert_crack_segment(p1, p2, embedded=False)
model.gen_list_dof(dof_per_node=tf.IS_2D)
model.cal_global_matrices(tf.XQuad4n)


def sel_condition(x, y, z):
    return y - 0.0


model.gen_dirichlet_bc(sel_condition)


def sel_condition(x, y, z):
    return y - 1


def force_expression(x, y, z):
    return 0.0, 1e7 * np.sin(np.pi * x / 1.0), 0.0


displacement_mult = 0.5e3
model.gen_surface_tractions(sel_condition, force_expression, tf.Quad4n, 10)
model.solve_static()
mesh1 = my_build_Quad4n(model, mult=displacement_mult).cast_to_unstructured_grid()
ghosts = np.argwhere(mesh1["is_enriched"] > 0)
mesh1.remove_cells(ghosts, inplace=True)

mesh2 = build_XQuad4n(model, mult=displacement_mult)
mesh2 = clean_and_average(mesh2, 4)

blocks = pv.MultiBlock([mesh1, mesh2])

pv.set_plot_theme("document")
pl = pv.Plotter(off_screen=True, window_size=[3840, 3840])
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
export_filename = "xfem_stress_presentation.png"
pl.screenshot(export_filename, transparent_background=False)
pl.close()
