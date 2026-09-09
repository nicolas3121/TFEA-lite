import matplotlib.pyplot as plt
import numpy as np


def generate_nudge_invariance_plot():
    # Setup aesthetic plot parameters for a 3.15-inch wide figure
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "serif",
            "axes.labelsize": 12,
            "axes.titlesize": 12,
        }
    )

    # --- 1. Define the geometry of the crack (a simple parabolic curve) ---
    def crack_curve(t):
        return np.array([t, 0.15 * t**2])

    def crack_tangent(t):
        vec = np.array([1.0, 0.3 * t])
        return vec / np.linalg.norm(vec)

    def crack_normal(t):
        T = crack_tangent(t)
        return np.array([-T[1], T[0]])  # 90-degree CCW rotation

    # --- 2. Define the key points ---
    t_tip = 0.0
    t_proj = 2.5
    t_end = 4.5

    p_tip = crack_curve(t_tip)
    p_proj = crack_curve(t_proj)
    n_vec = crack_normal(t_proj)
    t_vec = crack_tangent(t_proj)

    # Define node positions along the SAME normal line
    dist_original = 0.8
    dist_nudge = 1.8

    node_orig = p_proj + dist_original * n_vec
    node_nudge = p_proj + dist_nudge * n_vec

    # Generate arrays for drawing the lines
    t_vals_full = np.linspace(t_tip, t_end, 100)
    crack_path = np.array([crack_curve(t) for t in t_vals_full])

    t_vals_arc = np.linspace(t_tip, t_proj, 50)
    arc_path = np.array([crack_curve(t) for t in t_vals_arc])

    # --- 3. Create the Plot ---
    # Width = 3.15 inches (~50% A4 linewidth), Height = 3.5 inches
    fig, ax = plt.subplots(figsize=(3.15, 3.5))

    # Plot the full crack surface
    ax.plot(crack_path[:, 0], crack_path[:, 1], "k-", lw=1.5)

    # Highlight the Tangential Distance (Arc length phi_t)
    ax.plot(
        arc_path[:, 0],
        arc_path[:, 1],
        color="#0072B2",
        lw=2.5,
        linestyle="-",
        alpha=0.7,
    )

    # Plot the normal projection line
    extended_normal = p_proj + (dist_nudge + 0.3) * n_vec
    ax.plot(
        [p_proj[0], extended_normal[0]],
        [p_proj[1], extended_normal[1]],
        "k--",
        lw=1.0,
        alpha=0.6,
    )

    # Draw a visual arrow showing the "Nudge" shift between nodes
    ax.annotate(
        "",
        xy=node_nudge,
        xytext=node_orig,
        arrowprops={"arrowstyle": "->", "color": "#D55E00", "lw": 1.5},
    )
    # Nudge label positioned carefully near the shift arrow
    ax.text(
        node_orig[0] + 0.15,
        node_orig[1] + 0.1,
        "Nudge",
        color="#D55E00",
        fontsize=10,
        fontweight="bold",
        rotation=32,
    )

    # Draw the right-angle perpendicular symbol at the projection point
    square_size = 0.2
    p1 = p_proj + square_size * t_vec
    p2 = p1 + square_size * n_vec
    p3 = p_proj + square_size * n_vec
    ax.plot([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], "k-", lw=0.8)

    # Plot the key points
    ax.plot(*p_tip, marker="*", color="#E69F00", markersize=10, markeredgecolor="k")
    ax.plot(*p_proj, marker="s", color="black", markersize=4)
    ax.plot(*node_orig, marker="o", color="gray", markersize=6, markeredgecolor="k")
    ax.plot(*node_nudge, marker="o", color="#009E73", markersize=6, markeredgecolor="k")

    # --- 4. DIRECT LABELING (Replacing the Legend) ---

    # Original Node Label
    ax.annotate(
        "Original Node",
        xy=node_orig,
        xytext=(node_orig[0] - 2.5, node_orig[1] - 0.2),
        arrowprops={"arrowstyle": "-", "color": "gray", "lw": 1.0},
        fontsize=10,
        color="gray",
        ha="left",
    )

    # Nudged Node Label
    ax.annotate(
        "Nudged Node",
        xy=node_nudge,
        xytext=(node_nudge[0] + 0.8, node_nudge[1] + 0.3),
        arrowprops={"arrowstyle": "-", "color": "#009E73", "lw": 1.0},
        fontsize=10,
        color="#009E73",
        ha="left",
    )

    # Crack Tip Label
    ax.annotate(
        "Crack Tip",
        xy=p_tip,
        xytext=(p_tip[0], p_tip[1] - 0.7),
        arrowprops={"arrowstyle": "-", "color": "#E69F00", "lw": 1.0},
        fontsize=10,
        color="#D55E00",  # Using the darker orange for text readability
        ha="left",
    )

    # Crack Surface Label
    ax.annotate(
        "Crack Surface",
        xy=(3.8, crack_curve(3.8)[1]),
        xytext=(4.5, crack_curve(3.8)[1] - 0.6),
        arrowprops={"arrowstyle": "-", "color": "black", "lw": 1.0},
        fontsize=10,
        ha="center",
    )

    # Tangential Arc Label (combines the old constant arc + legend entry)
    ax.annotate(
        r"Tangential Dist ($\phi_t$)",
        xy=(crack_curve(1.25)),
        xytext=(3, -0.5),
        arrowprops={"facecolor": "#0072B2", "arrowstyle": "->", "lw": 1.0, "color": "#0072B2"},
        color="#0072B2",
        fontsize=10,
        fontweight="bold",
        ha="center",
    )

    # Projection Point Label
    ax.annotate(
        "Projection",
        xy=p_proj,
        xytext=(p_proj[0] + 0.5, p_proj[1] - 0.6),
        arrowprops={"facecolor": "black", "arrowstyle": "-", "lw": 0.8},
        fontsize=10,
        ha="center",
    )

    # Normal Line Label (aligned with the dashed line angle)
    ax.text(
        extended_normal[0] - 0.1,
        extended_normal[1] + 0.1,
        "Surface Normal",
        color="black",
        fontsize=10,
        rotation=-53,  # Approximate angle of the normal vector
        ha="right",
        alpha=0.7,
    )

    # --- 5. Formatting ---
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.2, 5.0)
    ax.set_ylim(
        -0.8, 3.2
    )  # Extended y-min slightly to give bottom labels breathing room

    # Clean up axes for a "diagram" look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout(pad=0.1)

    # Save as PDF
    plt.savefig("nudge_invariance.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    generate_nudge_invariance_plot()
