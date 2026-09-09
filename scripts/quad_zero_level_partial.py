import matplotlib.pyplot as plt
import numpy as np

from tfealite.core import quadratures as qd
from tfealite.core.quadratures import DuffyDistance
from tfealite.elements.utils import partial_cut_embedding_tri_iter

# Import your element and utilities
from tfealite.elements.XQuad4n import XQuad4n


def main():
    # 1. Setup the Worst-Case "Massive Bulge" scenario
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    phi_n = np.array([-0.17246507, 0.01231707, 0.17246507, 0.01231707])
    phi_t = np.array([0.0, 0.17246507, 0.0, -0.17246507])

    phi_n = [-0.16444344, 0.02348837, 0.16444344, 0.02348837]
    phi_t = [0.0, 0.16444344, 0.0, -0.16444344]
    phi_n = [-0.11591914, -0.01655738, 0.11591914, -0.01655738]
    phi_t = [0.0, 0.11591914, 0.0, -0.11591914]

    phi_n = [-0.2634543, -0.03535534, 0.19274362, -0.03535534]
    phi_t = [-0.03535534, 0.19274362, -0.03535534, -0.2634543]

    material = {"E": 1, "nu": 0.3, "rho": 1}
    real = {"t": 1}

    # Instantiate your actual class
    elem = XQuad4n(
        node_coords=node_coords,
        material=material,
        real=real,
        phi_n=phi_n,
        phi_t=phi_t,
        h_enrich=True,
        t_enrich=False,
        partial_cut=False,
    )

    # 2. Extract the parent triangle sub-divisions using your exact logic
    Nc1, Nc2 = elem._cal_intersections()

    xi_tip, eta_tip = elem._cal_tip_nat_coords()
    # tri1_coords = np.array([[-1, 1, 1], [-1, -1, 1], [1, 1, 1]])
    tri1_coords = np.vstack([elem.NAT_1.T, np.ones(3)])
    tip1 = np.linalg.solve(tri1_coords, [xi_tip, eta_tip, 1.0])

    tri2_coords = np.vstack([elem.NAT_2.T, np.ones(3)])
    tip2 = np.linalg.solve(tri2_coords, [xi_tip, eta_tip, 1.0])

    (rule, correction) = qd.QUAD_RULES[20]
    rule = rule.copy()
    rule[:, 0:2] = (1 + rule[:, 0:2]) / 2
    rule[:, 2] /= 4

    # NATIVE SIZING: 6.5 inches wide (leaves room for legend) by 4.0 inches tall
    plt.figure(figsize=(8, 4.0))

    # Helper function to process and plot the sub-triangles
    def plot_parent_tri(Nc, nat_x_e, tip, range):
        for Ni, detJi in partial_cut_embedding_tri_iter(Nc, tip, range):
            # Map sub-triangle to natural quad coordinates
            nat_sub_x_e = nat_x_e.T @ Ni

            poly = np.column_stack([nat_sub_x_e, nat_sub_x_e[:, 0]])
            plt.plot(
                poly[0, :],
                poly[1, :],
                color="gray",
                linewidth=1.0,
                linestyle=":",
                zorder=3,
                label="Sub-triangle edge",
            )

            N, _ = elem._base_shape_functions(nat_sub_x_e)
            sub_phi_n = N @ elem.phi_n
            sub_phi_t = N @ elem.phi_t
            behind_tip = sub_phi_t < 1e-10
            on_crack = np.isclose(sub_phi_n, 0.0, atol=1e-10)
            is_on_crack = np.sum(on_crack & behind_tip) == 2
            p1_idx = np.where(~on_crack)[0][0]
            sign = np.sign(sub_phi_n[p1_idx])

            if is_on_crack:
                # Highlight the sub-triangle containing the crack
                plt.fill(
                    nat_sub_x_e[0, :],
                    nat_sub_x_e[1, :],
                    color="orange",
                    alpha=0.15,
                    zorder=1,
                    label="Crack-adjacent sub-triangle",
                )

                on_crack_indices = np.where(on_crack)[0]
                p2_idx, p3_idx = on_crack_indices

                # Enforce CCW
                if p2_idx == 0 and p3_idx == 2:
                    p2_idx, p3_idx = 2, 0

                p2 = nat_sub_x_e[:, p2_idx]
                p3 = nat_sub_x_e[:, p3_idx]

                # Use your exact Newton-Raphson solver for the curved node
                p4 = elem._cal_curved_edge_node(p2, p3)

                # Generate quadratic curve points
                t_vals = np.linspace(0, 1, 50)
                quad_curve = np.array(
                    [
                        2 * (t - 1) * (t - 0.5) * p2
                        + 4 * t * (1 - t) * p4
                        + 2 * t * (t - 0.5) * p3
                        for t in t_vals
                    ]
                )

                # Plot your calculated approximations (scaled down widths/markers)
                plt.plot(
                    quad_curve[:, 0],
                    quad_curve[:, 1],
                    "b-",
                    linewidth=2.0,
                    zorder=5,
                    label="Quadratic fit",
                )
                plt.plot(
                    p4[0],
                    p4[1],
                    "bo",
                    markersize=5,
                    zorder=6,
                    label="Quadratic edge node",
                )
                plt.plot(
                    [p2[0], p3[0]],
                    [p2[1], p3[1]],
                    "r--",
                    linewidth=1.5,
                    zorder=4,
                    label="Standard linear fit",
                )

            x_e_i = elem._base_shape_functions(nat_sub_x_e)[0] @ elem.node_coords
            duffy = DuffyDistance(x_e_i)
            u, v = rule[:, 0], rule[:, 1]
            xi_d, eta_d, w_d = duffy.transform(u, v, beta=1)
            print(xi_d.shape)

            nat_coords_sub, detJi_mod, sign = elem._get_mapped_coords(
                xi_d,
                eta_d,
                sub_phi_n,
                behind_tip,
                nat_sub_x_e,
                detJi,
                False,
            )

            N_sub, dN_dxi_sub = elem._base_shape_functions(nat_coords_sub)
            phi_n = np.sum(elem.phi_n * N_sub, axis=1)
            phi_t = np.sum(elem.phi_t * N_sub, axis=1)
            if np.sum(behind_tip) > 1:
                to_flip = np.sign(phi_n) != sign
            else:
                to_flip = np.zeros_like(phi_n, dtype=bool)
            print(np.sum(to_flip))

            mask_true = to_flip
            mask_false = ~to_flip

            if np.any(mask_false):
                plt.scatter(
                    nat_coords_sub[0, mask_false],
                    nat_coords_sub[1, mask_false],
                    color="green",
                    s=8,  # marker size
                    zorder=8,
                    label="Mapped points (to_flip=False)",
                )

            # Plot points where to_flip is True (Red)
            if np.any(mask_true):
                plt.scatter(
                    nat_coords_sub[0, mask_true],
                    nat_coords_sub[1, mask_true],
                    color="red",
                    s=8,
                    zorder=8,
                    label="Mapped points (to_flip=True)",
                )

    # Run the plotting logic on both parent triangles using your dynamic NATs
    plot_parent_tri(Nc1, elem.NAT_1, tip1, range(4))
    plot_parent_tri(Nc2, elem.NAT_2, tip2, range(2, 6))

    grid_res = 300
    X, Y = np.meshgrid(np.linspace(-1, 1, grid_res), np.linspace(-1, 1, grid_res))
    Phi = np.zeros_like(X)

    # NOTE: With grid_res=1000, this loop will run 1,000,000 times.
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            N, _ = elem._base_shape_functions(np.array([X[i, j], Y[i, j]]))
            Phi[i, j] = np.dot(N[0], elem.phi_n)
    # Dashed, zorder 7, scaled down linewidth
    plt.contour(
        X,
        Y,
        Phi,
        levels=[0],
        colors="black",
        linewidths=1.8,
        linestyles="dashed",
        zorder=7,
    )
    plt.plot(
        [],
        [],
        color="black",
        linewidth=1.8,
        linestyle="--",
        label="True bilinear interface",
    )

    # 4. Presentation Formatting
    plt.plot(
        [-1, 1, 1, -1, -1],
        [-1, -1, 1, 1, -1],
        "k--",
        linewidth=1.2,
        zorder=0,
        label="Quad4n boundary",
    )

    # DYNAMIC DIAGONAL DETECTION
    shared_nodes = []
    for n1 in elem.NAT_1:
        for n2 in elem.NAT_2:
            if np.allclose(n1, n2):
                shared_nodes.append(n1)
    shared_nodes = np.array(shared_nodes)

    if len(shared_nodes) == 2:
        plt.plot(
            shared_nodes[:, 0],
            shared_nodes[:, 1],
            "k--",
            linewidth=1.0,
            zorder=0,
            label="Parent diagonal split",
        )

    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")

    # Remove duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Custom ordering for the legend
    order = [
        "Quad4n boundary",
        "Parent diagonal split",
        "Sub-triangle edge",
        "Crack-adjacent sub-triangle",
        "True bilinear interface",
        "Standard linear fit",
        "Quadratic fit",
        "Quadratic edge node",
    ]
    ordered_handles = [by_label[k] for k in order if k in by_label]
    ordered_labels = [k for k in order if k in by_label]

    # Native 12pt font size
    plt.legend(
        ordered_handles,
        ordered_labels,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),  # Pushed slightly further right to avoid clipping
        fontsize=12,
        framealpha=0.9,
    )

    plt.subplots_adjust(left=0.02, right=0.55, top=0.98, bottom=0.02)

    plt.savefig("Quad_folding_sub_element.pdf")
    plt.show()

    # plt.tight_layout()
    # plt.savefig("Quad_folding_sub_element.pdf")
    # plt.show()


if __name__ == "__main__":
    main()
