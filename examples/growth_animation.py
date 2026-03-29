import tfealite as tf
from tfealite.visualization.build_mesh import (
    my_build_Quad4n,
    build_XQuad4n,
)
from tfealite.core.dofs import BRANCH_DOFS
from tfealite import DofType
from tfealite.core.sif import DisplacementCorrelationMethodSIF as DCMSIF
from scipy.interpolate import splprep, BSpline
import numpy as np
import pyvista as pv


def calculate_growth_direction(K_I, K_II):
    return 2 * np.arctan2(-2 * K_II, K_I + np.sqrt(K_I**2 + 8 * K_II**2))


E_mod = 200e9
nu = 0.3
kappa = (3 - nu) / (3 + nu)
G_mod = (E_mod) / (2 * (1 + nu))
a = 0.115
W = 0.75
H = W
angle = np.pi / 4
x_elem = 150
y_elem = 153
h = 2 * W / x_elem

nodes, elements = tf.gen_rect_Quad4n(2 * W, 2 * H, x_elem, y_elem)
materials = [[1, {"E": E_mod, "nu": nu, "rho": 7850}]]
reals = [[1, {"t": 1}]]

da_dn = 0.013

# pts_list = [[0.0, 0.5], [0.05, 0.48], [0.08, 0.475], [0.09, 0.47], [0.1, 0.46]]

# p1 = np.array([0, 0.5])
# p2 = np.array([0.0 + a, 0.5])
p1 = np.array([W - a * np.cos(angle), H - a * np.sin(angle)])
p2 = np.array([W + a * np.cos(angle), H + a * np.sin(angle)])

pts_list = list(np.linspace(p1, p2, 20))

dcm = DCMSIF(
    kappa,
    G_mod,
    np.array([0.025, 0.031, 0.035, 0.038, 0.043, 0.048, 0.055, 0.063]),
    None,
)

# dcm = DCMSIF(kappa, G_mod, np.array([0.035, 0.038, 0.043, 0.048]), None)

for i in range(100):
    model = tf.XFEModel(
        nodes,
        elements,
        materials,
        reals,
        tip_enrichment=True,
        geometrical_range=0.09,
        corrected=True,
    )
    pts_array = np.array(pts_list)

    diffs = np.linalg.norm(np.diff(pts_array, axis=0), axis=1)
    valid_mask = np.insert(diffs > 1e-8, 0, True)
    pts_clean = pts_array[valid_mask]

    tck, u = splprep(pts_clean.T, s=1e-5, k=2)
    bspline = BSpline(tck[0], np.transpose(tck[1]), tck[2])
    model.insert_crack_spline(bspline, embedded=True, h=h, snapping_tolerance=0.03)

    model.gen_list_dof(dof_per_node=tf.IS_2D)

    model.cal_global_matrices(tf.XQuad4n)

    def sel_condition(x, y, z):
        return y - 0.0

    to_constrain = sel_condition(None, model.nodes[:, 2], None) < 1e-8
    nodes_to_constrain = 1 + np.where(to_constrain)[0]
    fix_dofs = list(
        model.list_dof.get_elem_dof_numbers_flat(nodes_to_constrain, DofType.UY)
    )
    fix_dofs.append(model.list_dof[(x_elem // 2 + 1, DofType.UX)])

    model.gen_P(np.array(sorted(set(fix_dofs)), dtype=int))

    def sel_condition(x, y, z):
        return y - 1.5

    def force_expression(x, y, z):
        return 0.0, 1e6, 0.0

    model.gen_surface_tractions(sel_condition, force_expression, tf.Quad4n, 2)

    model.solve_static()

    tip = model.level_sets[0].bspline(1)
    t_tip = model.level_sets[0].dbspline(1)
    t_mag = np.linalg.norm(t_tip)
    t_unit = t_tip / t_mag
    K_I, K_II = dcm.cal_sif(model.level_sets[0], model, model.cut_info, 1.0)
    theta = calculate_growth_direction(K_I, K_II)
    print("K_I", K_I, "K_II", K_II)
    print("angle", np.degrees(theta))

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    new_direction = rotation_matrix @ t_unit
    # if i == 0:
    #     new_tip = tip + 2 * da_dn * new_direction
    # else:
    new_tip = tip + da_dn * new_direction
    kink_threshold = np.radians(50.0)

    if np.abs(theta) > kink_threshold:
        print(
            f"Sharp kink of {np.degrees(theta):.1f}° detected! Applying spline clamp."
        )
        num_clamping_points = 4
        print(list(np.linspace(tip, new_tip, num_clamping_points + 1)[1:, :]))

        breadcrumbs = list(np.linspace(tip, new_tip, num_clamping_points + 1)[1:, :])
        pts_list.extend(breadcrumbs)
    else:
        pts_list.append(new_tip)
    tip = model.level_sets[0].bspline(0)
    t_tip = model.level_sets[0].dbspline(0)
    t_tip *= -1
    t_mag = np.linalg.norm(t_tip)
    t_unit = t_tip / t_mag
    K_I, K_II = dcm.cal_sif(model.level_sets[0], model, model.cut_info, 0.0)
    theta = -calculate_growth_direction(K_I, K_II)
    print("K_I", K_I, "K_II", K_II)
    print("angle", np.degrees(theta))

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    new_direction = rotation_matrix @ t_unit
    # if i == 0:
    #     new_tip = tip + 2 * da_dn * new_direction
    # else:
    new_tip = tip + da_dn * new_direction
    kink_threshold = np.radians(50)

    if np.abs(theta) > kink_threshold:
        print(
            f"Sharp kink of {np.degrees(theta):.1f}° detected at u=0! Applying spline clamp."
        )
        num_clamping_points = 4

        # Generate breadcrumbs from new_tip to tip, excluding the old tip
        # This gives us the points in the correct order: [new_tip, breadcrumb1, breadcrumb2...]
        new_pts = list(np.linspace(new_tip, tip, num_clamping_points + 1)[:-1, :])

        # Prepend to the front of the list
        pts_list = new_pts + pts_list
    else:
        # Prepend the single new tip to the front
        pts_list.insert(0, new_tip)

    multiplier = 5000
    mesh1 = my_build_Quad4n(model, multiplier).cast_to_unstructured_grid()
    ghosts = np.argwhere(mesh1["is_enriched"] > 0)
    mesh1.remove_cells(ghosts, inplace=True)

    mesh2 = build_XQuad4n(model, multiplier)
    # mesh2 = clean_and_average(mesh2, 4)

    blocks = pv.MultiBlock([mesh1, mesh2])

    # pl = pv.Plotter()
    pl = pv.Plotter(off_screen=True, window_size=[3 * 1920, 3 * 1080])
    vm_1 = mesh1.point_data["von_mises"]
    vm_2 = mesh2.point_data["von_mises"]
    mesh1.point_data["von_mises"] = np.nan_to_num(
        mesh1.point_data["von_mises"], nan=0.0, posinf=0.0, neginf=0.0
    )
    mesh2.point_data["von_mises"] = np.nan_to_num(
        mesh2.point_data["von_mises"], nan=0.0, posinf=0.0, neginf=0.0
    )
    all_vm = np.concatenate([vm_1, vm_2])

    # Cap the color scale at the 95th percentile (ignores the top 5% singularity spike)
    v_max = np.percentile(all_vm, 99)

    pl.add_mesh(
        blocks,
        # color="lightblue",
        scalars="von_mises",  # The exact string key in your point_data dict
        cmap="turbo",  # A great colormap for stress fields (or use "jet", "viridis")
        show_edges=True,  # Shows the mesh grid (including your sub-triangulations!)
        clim=[0, v_max],
        scalar_bar_args={"title": "Von Mises Stress"},
    )
    u_new = np.linspace(0, 1, 10000)
    spline_pts_2d = bspline(u_new)  # Output shape is (100, 2)
    z_coords = np.zeros((spline_pts_2d.shape[0], 1))
    spline_pts_3d = np.hstack((spline_pts_2d, z_coords))
    spline_pv_mesh = pv.lines_from_points(spline_pts_3d)
    pl.add_mesh(spline_pv_mesh, color="red", line_width=4, label="Crack Spline")
    enriched_nodes = np.where(model.list_dof.list_dof & BRANCH_DOFS)[0]
    enriched_nodes_coords = model.nodes[enriched_nodes, 1:]
    enriched_cloud = pv.PolyData(enriched_nodes_coords)
    pl.add_mesh(
        enriched_cloud,
        color="green",  # High contrast against light blue
        point_size=3,  # Adjust this to make them bigger/smaller
        render_points_as_spheres=True,  # Renders as nice 3D dots
        label="Heaviside Nodes",
    )

    pl.view_xy()
    # pl.show()
    pl.screenshot(f"crack_step_{i}.png", transparent_background=False)
    pl.close()
