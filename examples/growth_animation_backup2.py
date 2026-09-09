import numpy as np
import pyvista as pv
from scipy.interpolate import BSpline, splprep

import tfealite as tf
from tfealite import DofType
from tfealite.core.sif import DisplacementCorrelationMethodSIF as DCMSIF
from tfealite.visualization.build_mesh import (
    build_XQuad4n,
    my_build_Quad4n,
)


def calculate_growth_direction(K_I, K_II):
    return 2 * np.arctan2(-2 * K_II, K_I + np.sqrt(K_I**2 + 8 * K_II**2))


E_mod = 200e9
nu = 0.3
kappa = (3 - nu) / (3 + nu)
G_mod = (E_mod) / (2 * (1 + nu))
a = 0.5
W = 0.5


da_dn = 0.013

# pts_list = [[0.0, 0.5], [0.05, 0.48], [0.08, 0.475], [0.09, 0.47], [0.1, 0.46]]

# p1 = np.array([0, 0.5])
# p2 = np.array([0.0 + a, 0.5])
p1 = np.array([1.0, 0.0])
p2 = np.array([1.0, 0.25])

pts_list = list(np.linspace(p1, p2, 10))

dcm = DCMSIF(
    kappa,
    G_mod,
    np.array([0.025, 0.031, 0.035, 0.038, 0.043, 0.048, 0.055, 0.063]),
    None,
)

for i in range(100):
    print(pts_list)
    x_elem, y_elem = 161, 41

    nodes, elements = tf.gen_rect_Quad4n(2.0, 0.5, x_elem, y_elem)
    materials = [[1, {"E": E_mod, "nu": nu, "rho": 7850}]]
    reals = [[1, {"t": 1}]]
    model = tf.XFEModel(
        nodes,
        elements,
        materials,
        reals,
        tip_enrichment=True,
        geometrical_range=0.1,
        corrected=False,
    )
    pts_array = np.array(pts_list)

    diffs = np.linalg.norm(np.diff(pts_array, axis=0), axis=1)
    valid_mask = np.insert(diffs > 1e-8, 0, True)
    pts_clean = pts_array[valid_mask]

    tck, u = splprep(pts_clean.T, s=1e-5, k=3)
    bspline = BSpline(tck[0], np.transpose(tck[1]), tck[2])
    h = 2 * W / x_elem
    model.insert_crack_spline(bspline, embedded=False, h=h, snapping_tolerance=0.03)

    model.gen_list_dof(dof_per_node=tf.IS_2D)

    model.cal_global_matrices(tf.XQuad4n)

    # def sel_condition(x, y, z):
    #     return y - 1.0

    def sel_bottom_left(x, y, z):
        return (y - 0.0) ** 2 + (x - 0.5) ** 2

    def sel_bottom_right(x, y, z):
        return (y - 0.0) ** 2 + (x - 1.5) ** 2

    # 1. Pinned support on the bottom left (Fix UX and UY)
    fix_dofs = []
    node_bl = np.argmin(sel_bottom_left(model.nodes[:, 1], model.nodes[:, 2], 0)) + 1
    fix_dofs.append(model.list_dof[(node_bl, DofType.UX)])
    fix_dofs.append(model.list_dof[(node_bl, DofType.UY)])
    node_br = 1 + np.argmin(sel_bottom_right(model.nodes[:, 1], model.nodes[:, 2], 0))
    fix_dofs.append(model.list_dof[(node_br, DofType.UY)])

    model.gen_P(np.array(sorted(set(fix_dofs)), dtype=int))

    def sel_top_left(x, y, z):
        return np.isclose(y, 0.5) & np.isclose(x, 0.5, atol=0.05)

    def sel_top_right(x, y, z):
        return np.isclose(y, 0.5) & np.isclose(x, 1.5, atol=0.05)

    model.gen_surface_tractions(
        sel_top_left, lambda x, y, z: (0.0, 1e6, 0.0), tf.Quad4n, 2, reset=False
    )
    model.gen_surface_tractions(
        sel_top_right, lambda x, y, z: (0.0, -1e6, 0.0), tf.Quad4n, 2, reset=False
    )

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
    kink_threshold = np.radians(15.0)

    if np.abs(theta) > kink_threshold:
        print(
            f"Sharp kink of {np.degrees(theta):.1f}° detected! Applying spline clamp."
        )
        num_clamping_points = 2
        print(list(np.linspace(tip, new_tip, num_clamping_points + 1)[1:, :]))

        breadcrumbs = list(np.linspace(tip, new_tip, num_clamping_points + 1)[1:, :])
        pts_list.extend(breadcrumbs)
    else:
        pts_list.append(new_tip)
    multiplier = 10
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
    all_vm = np.concatenate([vm_1, vm_2])

    # Cap the color scale at the 95th percentile (ignores the top 5% singularity spike)
    v_max = np.percentile(all_vm, 99)

    pl.add_mesh(
        blocks,
        scalars="von_mises",  # The exact string key in your point_data dict
        cmap="turbo",  # A great colormap for stress fields (or use "jet", "viridis")
        show_edges=True,  # Shows the mesh grid (including your sub-triangulations!)
        clim=[0, v_max],
        scalar_bar_args={"title": "Von Mises Stress"},
    )

    pl.view_xy()
    # pl.show()
    pl.screenshot(f"crack_step_{i}.png", transparent_background=False)
    pl.close()
