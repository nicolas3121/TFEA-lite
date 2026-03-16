# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def plot_element_level_set(phi_nodes):
#     """
#     Plots the bilinear interpolation of level set values inside a Q4 element.
#     phi_nodes: list of 4 values [Bottom-Left, Bottom-Right, Top-Right, Top-Left]
#     """
#     # 1. Create a dense grid of local coordinates (xi, eta) from -1 to 1
#     xi = np.linspace(-1, 1, 200)
#     eta = np.linspace(-1, 1, 200)
#     XI, ETA = np.meshgrid(xi, eta)
#
#     # 2. Standard Bilinear Shape Functions for a Q4 element
#     N1 = 0.25 * (1 - XI) * (1 - ETA)  # Bottom-Left
#     N2 = 0.25 * (1 + XI) * (1 - ETA)  # Bottom-Right
#     N3 = 0.25 * (1 + XI) * (1 + ETA)  # Top-Right
#     N4 = 0.25 * (1 - XI) * (1 + ETA)  # Top-Left
#
#     # 3. Interpolate the nodal values across the whole element
#     PHI = N1 * phi_nodes[0] + N2 * phi_nodes[1] + N3 * phi_nodes[2] + N4 * phi_nodes[3]
#
#     # --- PLOTTING ---
#     plt.figure(figsize=(7, 6))
#
#     # Create the Heatmap (Red is positive, Blue is negative)
#     # We find the max absolute value to center the colormap white at exactly 0
#     max_val = np.max(np.abs(phi_nodes))
#     if max_val == 0:
#         max_val = 1
#
#     cp = plt.contourf(
#         XI, ETA, PHI, levels=50, cmap="RdBu_r", vmin=-max_val, vmax=max_val
#     )
#     plt.colorbar(cp, label="Interpolated Normal Distance ($\phi_n$)")
#
#     # Draw the Exact Zero-Crossing (The Crack)
#     plt.contour(XI, ETA, PHI, levels=[0.0], colors="black", linewidths=3)
#
#     # Mark the 4 nodes with their values
#     node_coords = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
#     for i, (x, y) in enumerate(node_coords):
#         plt.plot(x, y, "ko", markersize=8)
#
#         # Determine label position to avoid overlapping the box
#         ha = "right" if x < 0 else "left"
#         va = "top" if y < 0 else "bottom"
#
#         plt.text(
#             x * 1.05,
#             y * 1.05,
#             f"Node {i + 1}\n$\phi$ = {phi_nodes[i]:.3f}",
#             ha=ha,
#             va=va,
#             fontsize=10,
#             bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"),
#         )
#
#     plt.xlim(-1.4, 1.4)
#     plt.ylim(-1.4, 1.4)
#     plt.title("Q4 Element Zero-Crossing Interpolation")
#     plt.axhline(0, color="gray", linestyle="--", alpha=0.3)
#     plt.axvline(0, color="gray", linestyle="--", alpha=0.3)
#     plt.gca().set_aspect("equal")
#     plt.show()
#
#
# # ==========================================
# # PLUG YOUR 4 NODAL VALUES IN HERE
# # Order: [Bottom-Left, Bottom-Right, Top-Right, Top-Left]
# # ==========================================
#
# # Example 1: A perfect straight line (Values form a perfect plane)
# # plot_element_level_set([-1.0, -0.5, 0.5, 1.0])
#
# # Example 2: The "Sign Bug" (Creating an artificial zero-crossing)
# # plot_element_level_set([-1.0, 1.0, 1.0, -1.0])
#
# # Your actual values here:
# # my_element_nodes = [-0.12, 0.45, 0.60, -0.05]
# my_element_nodes = [-0.02925957, -0.01879596, 0.03026976, 0.01933582]
# plot_element_level_set(my_element_nodes)


import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. YOUR XFEM ALGORITHMS (Adapted for standalone use)
# ==========================================


def cut_embedding_iter(Nc, iter_range=range(4)):
    """Generates the Ni matrices for sub-triangles."""
    for i in iter_range:
        if i != 3:
            Ni = np.eye(3)
            Ni[:, (i + 1) % 3] = Nc[:, i]
            Ni[:, (i + 2) % 3] = Nc[:, (i + 2) % 3]
        else:
            Ni = Nc.copy()

        detJi = np.linalg.det(Ni)
        if not np.isclose(detJi, 0.0):
            yield Ni, detJi


def cal_intersections(phi_n):
    num = np.empty_like(phi_n)
    num[:-1] = phi_n[1:]
    num[-1] = phi_n[0]
    denom = num - phi_n
    unsolvable = np.isclose(denom, 0)
    on_crack = np.isclose(phi_n, 0)
    N1 = np.clip(
        np.divide(
            num,
            denom,
            out=np.ones_like(num),
            where=~unsolvable & ~on_crack,
        ),
        0,
        1,
    )
    [phi1, phi2, phi3, phi4] = phi_n
    A = phi1 - phi2 + phi3 - phi4
    B = -2.0 * phi1 + phi2 + phi4
    C = phi1
    num_diag = phi_n[0]
    denom_diag = num_diag - phi_n[2]
    unsolvable_diag = np.isclose(denom_diag, 0)
    s_linear = np.divide(
        num_diag, denom_diag, out=np.ones_like(phi1), where=~unsolvable_diag
    )
    discriminant = B**2 - 4 * A * C
    use_linear = np.isclose(A, 0) | (discriminant < 0)
    safe_A = np.where(use_linear, 1.0, A)
    sqrt_disc = np.sqrt(np.maximum(discriminant, 0))
    root1 = (-B + sqrt_disc) / (2.0 * safe_A)
    root2 = (-B - sqrt_disc) / (2.0 * safe_A)

    valid_root1 = (root1 >= 0.0) & (root1 <= 1.0)
    valid_root2 = (root2 >= 0.0) & (root2 <= 1.0)

    s_quad = np.where(valid_root1, root1, np.where(valid_root2, root2, s_linear))

    N1_diag = np.where(use_linear, s_linear, s_quad)

    N1_diag = np.where(unsolvable_diag | np.isclose(phi3, 0), 1.0, N1_diag)
    N1_diag = np.clip(N1_diag, 0, 1)

    Nc1 = np.array(
        [
            [N1[0], 0, 1 - N1_diag],
            [1 - N1[0], N1[1], 0],
            [0, 1 - N1[1], N1_diag],
        ]
    )
    Nc2 = np.array(
        [
            [1 - N1_diag, 0, 1 - N1[3]],
            [N1_diag, N1[2], 0],
            [0, 1 - N1[2], N1[3]],
        ]
    )
    return Nc1, Nc2


# ==========================================
# 2. PLOTTING FUNCTION
# ==========================================


def plot_element_with_subtriangles(phi_nodes):
    # --- 1. Base Element Heatmap & Contour ---
    xi = np.linspace(-1, 1, 200)
    eta = np.linspace(-1, 1, 200)
    XI, ETA = np.meshgrid(xi, eta)

    # Q4 shape functions
    N1 = 0.25 * (1 - XI) * (1 - ETA)
    N2 = 0.25 * (1 + XI) * (1 - ETA)
    N3 = 0.25 * (1 + XI) * (1 + ETA)
    N4 = 0.25 * (1 - XI) * (1 + ETA)

    PHI = N1 * phi_nodes[0] + N2 * phi_nodes[1] + N3 * phi_nodes[2] + N4 * phi_nodes[3]

    plt.figure(figsize=(8, 7))
    max_val = np.max(np.abs(phi_nodes)) if np.max(np.abs(phi_nodes)) != 0 else 1
    cp = plt.contourf(
        XI, ETA, PHI, levels=50, cmap="RdBu_r", vmin=-max_val, vmax=max_val, alpha=0.5
    )
    plt.colorbar(cp, label="Interpolated Normal Distance ($\phi_n$)")

    # The true mathematical zero-crossing
    plt.contour(XI, ETA, PHI, levels=[0.0], colors="black", linewidths=4, zorder=2)

    # --- 2. Generate and Plot Sub-triangles ---
    # Define the base triangles in natural coordinates
    # Triangle 1: Nodes 1, 2, 3 (Indices 0, 1, 2)
    nat_x_e_1 = np.array([[-1, -1], [1, -1], [1, 1]])

    # Triangle 2: Nodes 1, 3, 4 (Indices 0, 2, 3)
    nat_x_e_2 = np.array([[-1, -1], [1, 1], [-1, 1]])

    Nc1, Nc2 = cal_intersections(phi_nodes)

    sub_triangles = []

    # Generate sub-triangles from Nc1 (Base Triangle 1)
    for Ni, _ in cut_embedding_iter(Nc1):
        sub_nat_x_e = Ni.T @ nat_x_e_1
        sub_triangles.append(sub_nat_x_e)

    # Generate sub-triangles from Nc2 (Base Triangle 2)
    for Ni, _ in cut_embedding_iter(Nc2):
        sub_nat_x_e = Ni.T @ nat_x_e_2
        sub_triangles.append(sub_nat_x_e)

    # Draw them
    for i, tri in enumerate(sub_triangles):
        # Repeat the first point to close the triangle loop
        closed_tri = np.vstack((tri, tri[0]))
        plt.plot(
            closed_tri[:, 0],
            closed_tri[:, 1],
            color="#00FF00",
            linewidth=2.5,
            linestyle="-",
            marker="o",
            markersize=6,
            markerfacecolor="yellow",
            markeredgecolor="black",
            zorder=3,
        )

        # Calculate centroid to label the sub-triangle
        cx, cy = np.mean(tri, axis=0)
        plt.text(
            cx,
            cy,
            f"T{i + 1}",
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
        )

    # --- 3. Format Plot ---
    node_coords = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for i, (x, y) in enumerate(node_coords):
        plt.plot(x, y, "ks", markersize=10, zorder=4)
        ha = "right" if x < 0 else "left"
        va = "top" if y < 0 else "bottom"
        plt.text(
            x * 1.1,
            y * 1.1,
            f"Node {i + 1}\n$\phi$={phi_nodes[i]:.2f}",
            ha=ha,
            va=va,
            fontsize=10,
            weight="bold",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"),
        )

    plt.xlim(-1.4, 1.4)
    plt.ylim(-1.4, 1.4)
    plt.title("Validation: Generated Sub-Triangles vs. True 0-Contour", pad=20)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.gca().set_aspect("equal")

    # Add a custom legend
    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], color="black", lw=4),
        Line2D([0], [0], color="#00FF00", lw=2.5, marker="o", markerfacecolor="yellow"),
    ]
    plt.legend(
        custom_lines,
        ["True FEA 0-Contour", "Generated Sub-Triangles"],
        loc="upper right",
    )

    plt.show()


# ==========================================
# TEST IT WITH YOUR NODAL VALUES
# ==========================================

# Test with your sample values (Bottom-Left, Bottom-Right, Top-Right, Top-Left)
# my_nodal_values = [-0.12, 0.45, 0.60, -0.05]
my_nodal_values = [-0.02925957, -0.01879596, 0.03026976, 0.01933582]
plot_element_with_subtriangles(my_nodal_values)
