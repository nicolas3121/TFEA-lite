import matplotlib.pyplot as plt
import numpy as np

from tfealite.elements.utils import cut_embedding_tri_iter

# Import your element and utilities
from tfealite.elements.XQuad4n import XQuad4n


def main():
    # 1. Setup the Worst-Case "Massive Bulge" scenario
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    # phi_n = [-0.11, 0.05579515, 0.08, 0.00756507]
    # phi_n = [
    #     np.float64(-0.17246506858208627),
    #     np.float64(-0.02463414634146342),
    #     np.float64(0.1724650685820831),
    #     np.float64(-0.02463414634146342),
    # ]
    phi_n = [-0.17246507, 0.02463415, 0.17246507, 0.02463415]

    phi_t = np.ones(4)  # Dummy array, not used for h_enrich

    # phi_n = [-0.68986027, -0.51739521, -0.34493014, -0.51739521]
    # phi_t = [-0.17246507, 0.0, -0.17246507, -0.34493014]

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

    # NATIVE SIZING: 6.5 inches wide (leaves room for legend) by 4.0 inches tall
    plt.figure(figsize=(8, 4.0))

    # Helper function to process and plot the sub-triangles
    def plot_parent_tri(Nc, nat_x_e):
        for Ni, detJi in cut_embedding_tri_iter(Nc):
            # Map sub-triangle to natural quad coordinates
            nat_sub_x_e = nat_x_e.T @ Ni

            # Plot the sub-triangle edges (scaled down linewidth)
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

            # Use your exact logic to check if this sub-triangle is on the crack
            N, _ = elem._base_shape_functions(nat_sub_x_e)
            sub_phi_n = N @ elem.phi_n
            on_crack = np.isclose(sub_phi_n, 0.0, atol=1e-10)

            if np.sum(on_crack) == 2:
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

    # Run the plotting logic on both parent triangles using your dynamic NATs
    plot_parent_tri(Nc1, elem.NAT_1)
    plot_parent_tri(Nc2, elem.NAT_2)

    # 3. Plot the True Bilinear Contour for reference
    X, Y = np.meshgrid(np.linspace(-1, 1, 300), np.linspace(-1, 1, 300))
    Phi = np.zeros_like(X)
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
