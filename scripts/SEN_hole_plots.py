import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from scipy.interpolate import BSpline

# 1. SEN Geometry Configuration
L = 125.0
W = 30.0

notch_x = L / 2.0  # 62.5
# Removed notch_w and notch_height since the notch is gone

hole_r = 5.2
hole_x = notch_x - 9.3  # 53.2
hole_y = 14.8

data_dir = "./"


def plot_specimen_outline(ax, lw=1, color="#d0d0d0"):
    # Define the outer boundary as a simple flat rectangle
    boundary_pts = [
        (0, W),  # Top Left
        (L, W),  # Top Right
        (L, 0),  # Bottom Right
        (0, 0),  # Bottom Left
        (0, W),  # Close path
    ]

    ax.add_patch(
        PathPatch(
            Path(boundary_pts), facecolor="#f9f9f9", edgecolor=color, lw=lw, zorder=0
        )
    )

    # Add the structural hole
    ax.add_patch(
        Circle(
            (hole_x, hole_y),
            hole_r,
            facecolor="white",
            edgecolor=color,
            lw=lw,
            zorder=1,
        )
    )


# 2. Data Loading

# --- Miranda 2018 Data ---
try:
    predicted_2018 = np.genfromtxt(
        os.path.join(data_dir, "Miranda_2018_predicted.csv"), delimiter=","
    )
    predicted_2018[:, :] *= 2

    # 1. Align the first point with the BOTTOM of the SEN specimen (x=62.5, y=0.0)
    predicted_2018 -= predicted_2018[0, :]  # Zero it out
    predicted_2018[:, 0] += notch_x  # Shift X to center
    # (Y is already at 0 after zeroing it out, so we don't truncate it anymore)

    experimental_2003 = np.genfromtxt(
        os.path.join(data_dir, "Miranda_2003_experimental.csv"), delimiter=","
    )

    # 1. Align the first point with the BOTTOM of the SEN specimen (x=62.5, y=0.0)
    experimental_2003 -= experimental_2003[0, :]  # Zero it out
    experimental_2003[:, 0] += notch_x  # Shift X to center
    # (Y is already at 0 after zeroing it out, so we don't truncate it anymore)

except OSError:
    predicted_2018 = None
    experimental_2003 = None
    print("Warning: Miranda reference data not found.")

# --- Simulation Logic ---
try:
    sim_raw = np.genfromtxt("failed_growth_splines.csv", delimiter=",", skip_header=1)
except OSError:
    try:
        sim_raw = np.genfromtxt(
            "successful_growth_splines.csv", delimiter=",", skip_header=1
        )
    except OSError:
        # Fallback dummy data so the script doesn't completely crash if you don't have the files in the directory yet
        sim_raw = np.array(
            [[10, 0, notch_x, 0], [10, 1, notch_x, 5], [10, 2, notch_x - 2, 10]]
        )
        print("Warning: Simulation splines not found. Using fallback data.")

last_iter = np.max(sim_raw[:, 0])
ctrl_pts = sim_raw[sim_raw[:, 0] == last_iter, 2:4]

n_ctrl, k = len(ctrl_pts), min(2, len(ctrl_pts) - 1)  # ensure k is valid
knots = np.concatenate(([0] * k, np.linspace(0, 1, n_ctrl - k + 1), [1] * k))
spline_x = BSpline(knots, ctrl_pts[:, 0], k)
spline_y = BSpline(knots, ctrl_pts[:, 1], k)
t_eval = np.linspace(0, 1, 500)
x_sim, y_sim = spline_x(t_eval), spline_y(t_eval)

# 3. Final Standalone Plotting
fig, ax = plt.subplots(figsize=(10, 4))  # Wider aspect ratio for the full SEN specimen
plot_specimen_outline(ax)

# Plot literature data if it exists
if predicted_2018 is not None:
    ax.plot(
        predicted_2018[:, 0],
        predicted_2018[:, 1],
        "g-.",
        label="Miranda 2018",
        zorder=4,
    )


if experimental_2003 is not None:
    ax.plot(
        experimental_2003[:, 0],
        experimental_2003[:, 1],
        "b-.",
        label="Miranda 2003",
        zorder=4,
    )

# Plot your predicted path
ax.plot(x_sim, y_sim, "r-", linewidth=2.5, label="Present work", zorder=5)

# ==========================================
# VIEW 1: FULL SPECIMEN
# ==========================================
# Set limits to frame the entire 125x30 plate with slight padding
ax.set_xlim(-5, L + 5)
ax.set_ylim(-5, W + 5)
ax.set_aspect("equal")

# Formatting for the FULL view
ax.legend(loc="lower left", fontsize=11)
ax.set_xlabel("x (mm)", fontsize=12)
ax.set_ylabel("y (mm)", fontsize=12)
ax.set_title("MCTS Crack Path Prediction: SEN Specimen (Full View)", fontsize=14)
plt.grid(True, linestyle=":", alpha=0.4)
plt.tight_layout()

# --- Save Full View ---
plt.savefig("MCTS_path_FULL_SEN.pdf", dpi=300, bbox_inches="tight")

# ==========================================
# VIEW 2: ZOOMED IN
# ==========================================
# Calculate dynamic limits to encompass the crack AND the off-center hole
min_x_list = [np.min(x_sim), hole_x - hole_r]
max_x_list = [np.max(x_sim), notch_x + 5.0]
max_y_list = [np.max(y_sim), hole_y + hole_r]

if predicted_2018 is not None:
    min_x_list.append(np.min(predicted_2018[:, 0]))
    max_x_list.append(np.max(predicted_2018[:, 0]))
    max_y_list.append(np.max(predicted_2018[:, 1]))

if experimental_2003 is not None:
    min_x_list.append(np.min(experimental_2003[:, 0]))
    max_x_list.append(np.max(experimental_2003[:, 0]))
    max_y_list.append(np.max(experimental_2003[:, 1]))

x_min = min(min_x_list) - 2.0
x_max = max(max_x_list) + 2.0
y_min = -1.0  # Just below the bottom edge to see where it starts
y_max = max(max_y_list) + 2.0

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Update title and move legend so it doesn't block the crack in the tight view
ax.set_title("Crack Path Prediction: SEN Specimen", fontsize=12)
# Adjust legend to avoid covering the lines (loc=2 is upper left)
ax.legend(loc="lower left", fontsize=10)

# CRITICAL FIX: Force Matplotlib to recalculate layout for the zoomed view
fig.canvas.draw()
plt.tight_layout()

# --- Save Zoomed View (bbox_inches='tight' ensures nothing is cropped) ---
plt.savefig("MCTS_path_ZOOMED_SEN.pdf", dpi=300, bbox_inches="tight")

# Finally, display the plot (it will show the zoomed version on screen)
plt.show()
