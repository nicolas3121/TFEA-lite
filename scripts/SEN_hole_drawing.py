import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path


def draw_sen_specimen():
    # --- 1. GEOMETRY DEFINITION ---
    L = 125.0
    W = 30.0

    # Key X coordinates
    x_left = 0.0
    x_right = L
    x_center = L / 2.0  # 62.5

    # Loading coordinates
    s_dist = 50.0
    r_dist = 25.0
    x_bot_load_1 = x_center - s_dist  # 12.5
    x_bot_load_2 = x_center + s_dist  # 112.5
    x_top_load_1 = x_center - r_dist  # 37.5
    x_top_load_2 = x_center + r_dist  # 87.5

    # Notch coordinates
    y_notch = 2.5
    notch_w = 0.6  # Visual width of the notch

    # Hole parameters
    hole_R = 5.2
    hole_x = x_center - 9.3  # 53.2
    hole_y = 14.8

    # 3D tab offset
    _dx, _dy = 3, 3

    # NATIVE SIZING: Increased by 20% from previous size to fix overlap
    fig, ax = plt.subplots(figsize=(3.9, 1.8))

    # --- 2. DRAW MAIN SPECIMEN OUTLINE ---
    boundary_pts = [
        (x_left, W),
        (x_right, W),
        (x_right, 0),
        (x_center + notch_w / 2, 0),
        (x_center + notch_w / 2, y_notch),
        (x_center - notch_w / 2, y_notch),
        (x_center - notch_w / 2, 0),
        (x_left, 0),
        (x_left, W),
    ]

    ax.add_patch(
        PathPatch(
            Path(boundary_pts), facecolor="#dcdcdc", edgecolor="black", lw=1.0, zorder=1
        )
    )

    # Structural Hole
    ax.add_patch(
        Circle(
            (hole_x, hole_y),
            hole_R,
            facecolor="white",
            edgecolor="black",
            lw=1.0,
            zorder=2,
        )
    )

    # --- 3. DRAW 3D THICKNESS INDICATION (Top Right) ---
    # ax.plot([tab_start_x, tab_start_x + dx], [W, W + dy], "k-", lw=1.0)
    # ax.plot([x_right, x_right + dx], [W, W + dy], "k-", lw=1.0)
    # ax.plot([tab_start_x + dx, x_right + dx], [W + dy, W + dy], "k-", lw=1.0)
    # ax.plot([x_right, x_right + dx], [0, dy], "k-", lw=0.6)
    # ax.plot([x_right + dx, x_right + dx], [dy, W + dy], "k-", lw=1.0)

    # --- 4. HELPER FUNCTIONS FOR DRAFTING ---
    def dim_line(p1, p2, text, offset=(0, 0), rot=0, fsize=9):
        # Arrow line
        ax.annotate(
            "",
            xy=p1,
            xytext=p2,
            arrowprops={
                "arrowstyle": "<|-|>", "color": "black", "mutation_scale": 8, "lw": 0.6
            },
        )

        # Calculate text position with automatic offset to avoid line strike-through
        # since the white bbox has been removed for transparency.
        dx_off, dy_off = offset
        if offset == (0, 0):
            if rot == 0:
                dy_off = 1.8  # shift text up off the horizontal line
            elif rot == 90 or rot == -90:
                dx_off = -1.8  # shift text left off the vertical line

        mid_x = (p1[0] + p2[0]) / 2 + dx_off
        mid_y = (p1[1] + p2[1]) / 2 + dy_off

        # Text explicitly drawn without a bbox so it is completely transparent
        ax.text(
            mid_x, mid_y, text, ha="center", va="center", rotation=rot, fontsize=fsize
        )

    def ext_line(p1, p2):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", lw=0.4)

    # --- 5. ADD DIMENSIONS ---

    # Bottom Dimensions
    ext_line((x_left, 0), (x_left, -13))
    ext_line((x_right, 0), (x_right, -13))
    ext_line((x_bot_load_1, 0), (x_bot_load_1, -7))
    ext_line((x_bot_load_2, 0), (x_bot_load_2, -7))
    ext_line((x_center, 0), (x_center, -7))

    dim_line((x_left, -11), (x_right, -11), "125")
    dim_line((x_bot_load_1, -5), (x_center, -5), "s = 50")
    dim_line((x_center, -5), (x_bot_load_2, -5), "s = 50")

    # Top Dimensions
    ext_line((x_top_load_1, W), (x_top_load_1, W + 7))
    ext_line((x_top_load_2, W), (x_top_load_2, W + 7))
    ext_line((x_center, W), (x_center, W + 7))

    dim_line((x_top_load_1, W + 5), (x_center, W + 5), "r = 25")
    dim_line((x_center, W + 5), (x_top_load_2, W + 5), "r = 25")

    # Width Dimension
    ext_line((x_left, 0), (x_left - 13, 0))
    ext_line((x_left, W), (x_left - 13, W))
    dim_line((x_left - 10, 0), (x_left - 10, W), "W = 30", rot=90)

    # Notch depth
    ext_line((x_center, y_notch), (x_center + 10, y_notch))
    dim_line((x_center + 7, 0), (x_center + 7, y_notch), "2.5", offset=(0, 4))

    # Hole Position
    ext_line((hole_x, hole_y), (hole_x, 0))
    ext_line((hole_x, hole_y), (hole_x - 7, hole_y))

    dim_line((hole_x, 3), (x_center, 3), "9.3", offset=(0, 3))
    dim_line((hole_x - 7, 0), (hole_x - 7, hole_y), "14.8", rot=90, offset=(-3, 0))

    # Thickness Dimension
    # ext_line((tab_start_x + dx, W + dy), (tab_start_x + dx + 3, W + dy + 3))
    # ext_line((x_right + dx, W + dy), (x_right + dx + 3, W + dy + 3))

    # ax.annotate(
    #     "",
    #     xy=(tab_start_x + dx + 1, W + dy + 1),
    #     xytext=(x_right + dx + 1, W + dy + 1),
    #     arrowprops=dict(arrowstyle="<|-|>", color="black", mutation_scale=8, lw=0.6),
    # )
    # ax.text(
    #     (tab_start_x + x_right) / 2 + dx,
    #     W + dy + 3.0,
    #     "t = 10",
    #     ha="center",
    #     va="center",
    #     fontsize=9,
    # )

    # --- 6. DIAMETER CALLOUT ---
    ax.annotate(
        "R = 5.2",
        xy=(hole_x - hole_R * np.cos(np.pi / 4), hole_y + hole_R * np.sin(np.pi / 4)),
        xytext=(hole_x - 12, hole_y + 4),
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 0.8, "mutation_scale": 10},
        fontsize=9,
        ha="right",
        va="bottom",
    )

    # Centerlines
    ax.plot([hole_x - 6, hole_x + 6], [hole_y, hole_y], "k-.", lw=0.4)
    ax.plot([hole_x, hole_x], [hole_y - 6, hole_y + 6], "k-.", lw=0.4)
    ax.plot([x_center, x_center], [-2, W + 2], "k-.", lw=0.4)

    # --- 7. LOAD ARROWS (P) ---
    arrow_len = 5
    arrow_props = {"facecolor": "black", "width": 1.0, "headwidth": 4, "headlength": 4, "lw": 0.5}

    # Top Loads
    ax.annotate(
        "",
        xy=(x_top_load_1, W),
        xytext=(x_top_load_1, W + arrow_len),
        arrowprops=arrow_props,
    )
    ax.text(
        x_top_load_1 - 2,
        W + arrow_len - 1,
        "P",
        fontsize=10,
        fontweight="bold",
        ha="right",
    )

    ax.annotate(
        "",
        xy=(x_top_load_2, W),
        xytext=(x_top_load_2, W + arrow_len),
        arrowprops=arrow_props,
    )
    ax.text(
        x_top_load_2 + 2,
        W + arrow_len - 1,
        "P",
        fontsize=10,
        fontweight="bold",
        ha="left",
    )

    # Bottom Loads
    ax.annotate(
        "",
        xy=(x_bot_load_1, 0),
        xytext=(x_bot_load_1, -arrow_len),
        arrowprops=arrow_props,
    )
    ax.text(
        x_bot_load_1 - 2,
        -arrow_len + 1,
        "P",
        fontsize=10,
        fontweight="bold",
        ha="right",
    )

    ax.annotate(
        "",
        xy=(x_bot_load_2, 0),
        xytext=(x_bot_load_2, -arrow_len),
        arrowprops=arrow_props,
    )
    ax.text(
        x_bot_load_2 + 2, -arrow_len + 1, "P", fontsize=10, fontweight="bold", ha="left"
    )

    # --- 8. PLOT FORMATTING ---
    ax.set_aspect("equal")

    # Limits expanded slightly to accommodate the 20% size boost
    ax.set_xlim(-18, 140)
    ax.set_ylim(-16, 48)
    ax.axis("off")

    # Reduced padding so it slots into Typst/LaTeX cleanly
    plt.tight_layout(pad=0.1)
    plt.savefig("SEN_Specimen_Dimensions_A4_Half.pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    draw_sen_specimen()
