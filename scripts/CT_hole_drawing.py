import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path


def draw_mirrored_specimen_diagram():
    # --- 1. MIRRORED GEOMETRY DEFINITION ---
    W, H = 40.0, 40.0

    # Key X coordinates (Mirrored, so they are negative to flip along Y axis)
    x_left = 0.0  # Was 0, now the right-most edge
    x_right = -W  # Was 40, now the left-most edge
    x_load = -29.5
    x_notch_tip = -23.0
    x_notch_shoulder = -24.5

    # NEW: Crack extends 2.5mm past the notch tip into the uncut material
    # Because it is mirrored, the uncut material is in the positive X direction relative to the tip
    x_crack_tip = x_notch_tip + 2.5  # -20.5

    # Key Y coordinates
    y_top = H / 2
    y_bot = -H / 2
    y_load_top = y_top - 9.2
    y_load_bot = y_bot + 9.2

    # Hole parameters
    r_load = 9.5 / 2
    r_mod = 7.0 / 2

    # Internal parameters A and B
    A_val = 8.0
    B_val = 7.0
    x_mod = x_notch_tip + A_val  # -15.0
    y_mod = B_val

    fig, ax = plt.subplots(figsize=(10, 10))

    # --- 2. DRAW MAIN SPECIMEN OUTLINE ---
    boundary_pts = [
        (x_left, y_top),
        (x_right, y_top),
        (x_right, 1.5),
        (x_notch_shoulder, 1.5),
        (x_notch_tip, 0),
        (x_notch_shoulder, -1.5),
        (x_right, -1.5),
        (x_right, y_bot),
        (x_left, y_bot),
        (x_left, y_top),
    ]

    ax.add_patch(
        PathPatch(
            Path(boundary_pts), facecolor="#f9f9f9", edgecolor="black", lw=1.5, zorder=1
        )
    )

    # Draw the Sharp Crack extending from the notch
    ax.plot([x_notch_tip, x_crack_tip], [0, 0], "k-", lw=1.5, zorder=2)

    # Load Holes & Modified Hole
    ax.add_patch(
        Circle(
            (x_load, y_load_top),
            r_load,
            facecolor="white",
            edgecolor="black",
            lw=1.5,
            zorder=2,
        )
    )
    ax.add_patch(
        Circle(
            (x_load, y_load_bot),
            r_load,
            facecolor="white",
            edgecolor="black",
            lw=1.5,
            zorder=2,
        )
    )
    ax.add_patch(
        Circle(
            (x_mod, y_mod),
            r_mod,
            facecolor="white",
            edgecolor="black",
            lw=1.5,
            zorder=2,
        )
    )

    # --- 3. HELPER FUNCTIONS FOR DRAFTING ---
    def dim_line(p1, p2, text, offset=(0, 0), rot=0):
        # The main arrow line
        ax.annotate(
            "",
            xy=p1,
            xytext=p2,
            arrowprops={"arrowstyle": "<|-|>", "color": "black", "mutation_scale": 12, "lw": 1},
        )
        # The text with a white bounding box to hide lines underneath
        mid_x = (p1[0] + p2[0]) / 2 + offset[0]
        mid_y = (p1[1] + p2[1]) / 2 + offset[1]
        ax.text(
            mid_x,
            mid_y,
            text,
            ha="center",
            va="center",
            rotation=rot,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
            fontsize=18,
        )

    def ext_line(p1, p2):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", lw=0.5)

    # --- 4. ADD DIMENSIONS ---
    # Width dimensions (Bottom)
    ext_line((x_left, y_bot), (x_left, y_bot - 10))
    ext_line((x_load, y_load_bot - r_load), (x_load, y_bot - 8))
    ext_line((x_right, y_bot), (x_right, y_bot - 10))

    # dim_line((x_left, y_bot - 8), (x_right, y_bot - 8), "40")
    dim_line((x_left, y_bot - 4), (x_load, y_bot - 4), "w = 29.5")
    dim_line((x_load, y_bot - 4), (x_right, y_bot - 4), "10.5")

    # Height dimensions (Now on the Left Side, offset negatively)
    ext_line((x_right, y_top), (x_right - 8, y_top))
    ext_line((x_load - r_load, y_load_top), (x_right - 6, y_load_top))
    ext_line((x_right, 1.5), (x_right - 6, 1.5))
    ext_line((x_right, -1.5), (x_right - 6, -1.5))
    ext_line((x_load - r_load, y_load_bot), (x_right - 6, y_load_bot))
    ext_line((x_right, y_bot), (x_right - 8, y_bot))

    dim_line((x_right - 8, y_bot), (x_right - 8, y_top), "40", rot=90)
    dim_line((x_right - 4, y_load_top), (x_right - 4, y_top), "9.2", rot=90)
    dim_line((x_right - 4, y_bot), (x_right - 4, y_load_bot), "9.2", rot=90)
    dim_line((x_right - 4, -1.5), (x_right - 4, 1.5), "3", rot=90)

    # Internal Dimensions (A, B, a, 23)
    ext_line((x_notch_tip, 0), (x_notch_tip, -8))
    ext_line((x_mod, y_mod - r_mod), (x_mod, -4))

    dim_line((x_left, -6), (x_notch_tip, -6), "23")
    dim_line((x_mod, -2.5), (x_notch_tip, -2.5), "A")

    # 'a' Dimension: Now measures from the actual crack tip to the load line
    ext_line((x_crack_tip, 0), (x_crack_tip, 4))
    ext_line((x_load, y_load_top - r_load), (x_load, 2.5))
    dim_line((x_crack_tip, 2.5), (x_load, 2.5), "a")

    # B dimension (Vertical)
    ext_line((x_mod + r_mod, y_mod), (x_mod + 6, y_mod))
    ext_line((x_crack_tip + 3, 0), (x_mod + 6, 0))
    dim_line((x_mod + 4, 0), (x_mod + 4, y_mod), "B", rot=90)

    # --- 5. DIAMETER CALLOUTS & CENTERLINES ---
    # Ø9.5 Load hole
    ax.annotate(
        "Ø9.5",
        xy=(
            x_load - r_load * np.cos(np.pi / 4),
            y_load_top + r_load * np.sin(np.pi / 4),
        ),
        xytext=(x_load - 5, y_load_top + 6),
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.2, "mutation_scale": 12},
        fontsize=18,
        ha="right",
        va="bottom",
    )

    # Centerline for top load hole
    ax.plot([x_load - 6, x_load + 6], [y_load_top, y_load_top], "k-.", lw=0.8)
    ax.plot([x_load, x_load], [y_load_top - 6, y_load_top + 6], "k-.", lw=0.8)

    # Centerline for bottom load hole
    ax.plot([x_load - 6, x_load + 6], [y_load_bot, y_load_bot], "k-.", lw=0.8)
    ax.plot([x_load, x_load], [y_load_bot - 6, y_load_bot + 6], "k-.", lw=0.8)

    # Ø7 Modified hole
    ax.annotate(
        "Ø7",
        xy=(x_mod + r_mod * np.cos(np.pi / 4), y_mod + r_mod * np.sin(np.pi / 4)),
        xytext=(x_mod + 5, y_mod + 6),
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.2, "mutation_scale": 12},
        fontsize=18,
        ha="left",
        va="bottom",
    )

    # Centerlines for modified hole
    ax.plot([x_mod - 5, x_mod + 5], [y_mod, y_mod], "k-.", lw=0.8)
    ax.plot([x_mod, x_mod], [y_mod - 5, y_mod + 5], "k-.", lw=0.8)

    # --- 6. LOAD ARROWS (P) & CRACK CENTERLINE ---
    # Top P
    ax.annotate(
        "",
        xy=(x_load, y_load_top + r_load + 5),
        xytext=(x_load, y_load_top),
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 2, "mutation_scale": 15},
    )
    ax.text(
        x_load - 1.5,
        y_load_top + r_load + 2.5,
        "P",
        fontsize=20,
        fontweight="bold",
        ha="right",
    )

    # Bottom P
    ax.annotate(
        "",
        xy=(x_load, y_load_bot - r_load - 5),
        xytext=(x_load, y_load_bot),
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 2, "mutation_scale": 15},
    )
    ax.text(
        x_load - 1.5,
        y_load_bot - r_load - 2.5,
        "P",
        fontsize=20,
        fontweight="bold",
        ha="right",
    )

    # Crack centerline (extends from crack tip into uncut material)
    ax.plot([x_crack_tip, x_crack_tip + 10], [0, 0], "k-.", lw=0.8)

    # --- 7. PLOT FORMATTING ---
    ax.set_aspect("equal")

    # Give a little padding around the edges (adjusted for mirrored coordinates)
    ax.set_xlim(-55, 10)
    ax.set_ylim(-25, 30)

    ax.axis("off")

    plt.tight_layout()
    plt.savefig("CT_Specimen_Dimensions.pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    draw_mirrored_specimen_diagram()
