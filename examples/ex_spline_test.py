import tfealite as tf
import numpy as np
from scipy.interpolate import splprep, BSpline
from tfealite.visualization.build_mesh import (
    my_build_Quad4n,
    build_XQuad4n,
)
import pyvista as pv


nodes, elements = tf.gen_rect_Quad4n(1, 1, 40, 40)
materials = [[1, {"E": 1, "nu": 0.33, "rho": 7850}]]
reals = [[1, {"t": 1}]]
model = tf.XFEModel(
    nodes,
    elements,
    materials,
    reals,
    tip_enrichment=True,
    geometrical_range=0.15,
    corrected=True,
)
pts = np.array([[0.0, 0.55], [0.2, 0.45], [0.3, 0.35], [0.72, 0.54]]).T
# pts = np.array([[0.22, 0.23], [0.3, 0.35], [0.72, 0.54]]).T

tck, u = splprep(pts, s=0, k=2)

crack_spline = BSpline(tck[0], np.transpose(tck[1]), tck[2])
model.insert_crack_spline(crack_spline, embedded=False)
print(model.level_sets[0].phi_t_list)
# model.insert_crack_segment(p1, p2, embedded=False)
# print(model.level_sets[0].phi_n)

model.gen_list_dof(dof_per_node=tf.IS_2D)
# for elem, info in model.cut_info.items():
#     ls, cut_type, tip = info
#     if cut_type == CutType.PARTIAL or cut_type == CutType.CUT:
#         element = model.elements[elem - 1]
#         nodes = element[4]
#         phi_n, phi_t = model.level_sets[0].get(nodes, 1)
#         print(
#             "elem",
#             elem,
#             cut_type,
#             "phi_n",
#             np.array_repr(phi_n),
#             "phi_t",
#             np.array_repr(phi_t),
#         )
#         # print("phi_t", phi_t)
# print(model.cut_info)
# print(model.list_dof.list_dof)
# model.cut_info.pop(8)
# node_numbers = model.list_dof.get_elem_dof_numbers(
#     to_delete, mask=tf.DofType.HX | tf.DofType.HY
# ).flatten()
model.cal_global_matrices({"Quad4n": tf.XQuad4n})


def sel_condition(x, y, z):
    return y - 0.0


to_delete = [2, 5]
# bc.my_gen_dirichlet_bc(model, sel_condition, to_delete)
model.gen_dirichlet_bc(sel_condition)


def sel_condition(x, y, z):
    return y - 1


def force_expression(x, y, z):
    return 0.0, 0.5, 0.0


model.gen_nodal_forces(sel_condition, force_expression)

# to_delete_dof_numbers = model.list_dof.get_elem_dof_numbers(
#     np.array(to_delete), tf.DofType.HX | tf.DofType.HY
# ).flatten()
#
# for dof_number in to_delete_dof_numbers:

model.solve_static()


# Ug = model.Ug[
#     model.list_dof.get_elem_dof_numbers(
#         1 + np.arange(model.n_nodes, dtype=int), tf.IS_2D
#     ).flatten()
# ]
# print(model.tip)
# # print(Ug)
# #
# model.show(
#     node_size=10,
#     nbc_size=15,
#     load_size=(0.8, 0.15),
#     Ug=0.001 * Ug,
#     node_stress=None,
#     colorbar_title="s_yy",
# )

# model.Ug *= 0.00005
mult = 0.00005
# model.Ug = np.zeros(len(model.list_dof))
mesh1 = my_build_Quad4n(model, mult=mult).cast_to_unstructured_grid()
ghosts = np.argwhere(mesh1["is_enriched"] > 0)
mesh1.remove_cells(ghosts, inplace=True)
mesh2 = build_XQuad4n(model, mult=mult)
blocks = pv.MultiBlock([mesh1, mesh2])
# blocks.plot(show_edges=True, color="lightblue")
vm_1 = mesh1.point_data["von_mises"]
vm_2 = mesh2.point_data["von_mises"]
all_vm = np.concatenate([vm_1, vm_2])
v_max = np.percentile(all_vm, 98)
pl = pv.Plotter()
# pl.add_mesh(blocks, color="lightblue", show_edges=True)
pl.add_mesh(
    blocks,
    color="lightblue",
    # scalars="von_mises",  # The exact string key in your point_data dict
    # cmap="turbo",  # A great colormap for stress fields (or use "jet", "viridis")
    show_edges=True,  # Shows the mesh grid (including your sub-triangulations!)
    clim=[0, v_max],
    scalar_bar_args={"title": "Von Mises Stress"},
)
u_new = np.linspace(0, 1, 1000)
spline_pts_2d = crack_spline(u_new)  # Output shape is (100, 2)
z_coords = np.zeros((spline_pts_2d.shape[0], 1))
spline_pts_3d = np.hstack((spline_pts_2d, z_coords))
spline_pv_mesh = pv.lines_from_points(spline_pts_3d)
pl.add_mesh(spline_pv_mesh, color="red", line_width=4, label="Crack Spline")
u_box = np.linspace(0.0, 1.0, 30 + 1)
curve_pts_box = crack_spline(u_box)

min_coords = np.minimum(curve_pts_box[:-1], curve_pts_box[1:]) - 0.1
max_coords = np.maximum(curve_pts_box[:-1], curve_pts_box[1:]) + 0.1

# 2. Iterate through each segment to draw its box
# for i in range(30):
#     xmin, ymin = min_coords[i]
#     xmax, ymax = max_coords[i]
#
#     # Define the corners: Bottom-Left, Bottom-Right, Top-Right, Top-Left, Bottom-Left (to close the loop)
#     box_corners = np.array(
#         [
#             [xmin, ymin, 0.0],
#             [xmax, ymin, 0.0],
#             [xmax, ymax, 0.0],
#             [xmin, ymax, 0.0],
#             [xmin, ymin, 0.0],
#         ]
#     )
#
#     # Convert points into a PyVista line mesh
#     box_mesh = pv.lines_from_points(box_corners)
#
#     # Add to the plotter. We only assign the label to the first box so the legend stays clean.
#     box_label = "Bounding Boxes" if i == 0 else None
#     pl.add_mesh(box_mesh, color="blue", line_width=2, label=box_label)
#
# heaviside_nodes = np.where(model.list_dof.list_dof & HEAVISIDE_DOFS != 0)[0]
# heaviside_cloud = pv.PolyData(model.nodes[heaviside_nodes, 1:])
# pl.add_mesh(
#     heaviside_cloud,
#     color="orange",  # High contrast against light blue
#     point_size=15,  # Adjust this to make them bigger/smaller
#     render_points_as_spheres=True,  # Renders as nice 3D dots
#     label="Heaviside Nodes",
# )
#
pl.view_xy()
# pl.enable_parallel_projection()
pl.show()
