import matplotlib.pyplot as plt
import numpy as np


def generate_3d_nudge_error_frontal_plot():
    # Setup aesthetic plot parameters for a 3.15-inch wide figure
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "serif",
            "axes.titlesize": 12,
        }
    )

    # --- 1. Define the 3D Geometry ---
    # Crack Surface: A twisted plane (z = k*x*y)
    k = 0.5
    u = np.linspace(0, 1.8, 15)
    v = np.linspace(0, 2.2, 15)
    U, V = np.meshgrid(u, v)
    Z_surf = k * U * V

    # Crack Front: The boundary where U = 0 (the Y-axis)
    u0, v0 = 1.2, 1.0
    P_surf = np.array([u0, v0, k * u0 * v0])

    # Compute tangent vectors and normal vector at P_surf
    T_u = np.array([1, 0, k * v0])
    T_v = np.array([0, 1, k * u0])
    N_unnormalized = np.cross(T_u, T_v)
    N_hat = N_unnormalized / np.linalg.norm(N_unnormalized)

    # --- 2. Define the Nodes and Projections ---
    dist_orig = 0.5
    dist_nudge = 1.6

    # Node 0 (Original) and Node 1 (Nudged)
    N0 = P_surf + dist_orig * N_hat
    N1 = P_surf + dist_nudge * N_hat

    # Orthogonal projections onto the Crack Front (y-axis)
    F0 = np.array([0, N0[1], 0])
    F1 = np.array([0, N1[1], 0])

    # --- 3. Create the Plot ---
    fig = plt.figure(figsize=(4.0, 4.0))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the twisted crack surface
    ax.plot_surface(U, V, Z_surf, color="#0072B2", alpha=0.3, edgecolor="white", lw=0.5)

    # Plot the crack front (Frontal view means this goes horizontally across the screen)
    ax.plot([0, 0], [0, 2.5], [0, 0], color="black", lw=3)

    # Plot Surface Normal Line
    normal_end = P_surf + 2.2 * N_hat
    ax.plot(
        [P_surf[0], normal_end[0]],
        [P_surf[1], normal_end[1]],
        [P_surf[2], normal_end[2]],
        "k--",
        lw=1.2,
    )

    # Draw the "Nudge" arrow along the normal
    ax.quiver(
        N0[0],
        N0[1],
        N0[2],
        N1[0] - N0[0],
        N1[1] - N0[1],
        N1[2] - N0[2],
        color="#D55E00",
        lw=2,
        arrow_length_ratio=0.15,
    )

    # Draw projection paths to the crack front (These will look perfectly vertical now)
    ax.plot(
        [N0[0], F0[0]],
        [N0[1], F0[1]],
        [N0[2], F0[2]],
        color="gray",
        linestyle=":",
        lw=1.5,
    )
    ax.plot(
        [N1[0], F1[0]],
        [N1[1], F1[1]],
        [N1[2], F1[2]],
        color="#009E73",
        linestyle=":",
        lw=1.5,
    )

    # Highlight the projection error on the crack front
    ax.plot(
        [F0[0], F1[0]], [F0[1], F1[1]], [F0[2], F1[2]], color="#D55E00", lw=5, zorder=5
    )

    # Plot key points
    ax.scatter(*N0, color="gray", s=40, edgecolor="k", depthshade=False)
    ax.scatter(*N1, color="#009E73", s=40, edgecolor="k", depthshade=False)
    ax.scatter(*F0, color="black", marker="s", s=25, depthshade=False)
    ax.scatter(*F1, color="black", marker="s", s=25, depthshade=False)
    ax.scatter(*P_surf, color="#0072B2", marker="o", s=15, depthshade=False)

    # --- 4. Direct Labeling in 3D (Optimized for Frontal View) ---

    # Anchor text closely to the nodes.
    ax.text(
        1.5,
        N0[1] + 0.15,
        N0[2] - 0.1,
        "Original\nNode",
        color="gray",
        fontsize=9,
        ha="right",
    )
    ax.text(
        N1[0],
        N1[1] + 0.2,
        N1[2],
        "Nudged\nNode",
        color="#009E73",
        fontsize=9,
        ha="left",
    )

    # Shift label next to the arrow
    mid_nudge = (N0 + N1) / 2
    ax.text(
        1.5,
        mid_nudge[1] + -0.3,
        mid_nudge[2],
        "Normal\nShift",
        color="#D55E00",
        fontsize=9,
        fontweight="bold",
    )

    # --- FIXED: Pushed Projection labels further apart ---
    # Subtracted more from F0[1] and added more to F1[1]
    ax.text(
        F0[0],
        F0[1] - 0.3,
        F0[2] - 0.2,
        "Original\nProj.",
        color="black",
        fontsize=9,
        ha="right",
    )
    ax.text(
        F1[0],
        F1[1] + 0.3,
        F1[2] - 0.2,
        "New\nProj.",
        color="#009E73",
        fontsize=9,
        ha="left",
    )

    # Highlight the gap below the crack front
    mid_F = (F0 + F1) / 2
    ax.text(
        mid_F[0],
        mid_F[1],
        mid_F[2] - 0.65,
        "Projection\nDrift Error",
        color="#D55E00",
        fontsize=10,
        ha="center",
        fontweight="bold",
    )

    # Surface Labels
    ax.text(
        1.5,
        2.3,
        1.0,
        "Curved\nCrack\nSurface",
        color="#0072B2",
        fontsize=10,
        alpha=0.8,
        ha="center",
    )

    # --- FIXED: Lowered the Crack Front Label ---
    # Changed Z from 0.2 to -0.2 to pull it safely below the thick black line
    ax.text(0, 1, 0.3, "Crack Front", color="black", fontsize=11, fontweight="bold")

    # --- 5. Formatting ---
    ax.view_init(elev=-185, azim=20)

    # Remove all background panes, axes, and grids for a clean diagram
    ax.set_axis_off()

    # Adjust limits to frame the drawing tightly
    ax.set_xlim(-0.2, 1.8)
    ax.set_ylim(-0.2, 2.5)
    ax.set_zlim(-0.8, 2.5)

    plt.tight_layout(pad=0.0)

    # Save as PDF
    plt.savefig(
        "nudge_3D_drift_frontal.pdf", format="pdf", dpi=300, bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    generate_3d_nudge_error_frontal_plot()
