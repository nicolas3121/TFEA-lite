import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from scipy.interpolate import BSpline

# 1. Configuration
specimen_params = {
    "CT01": {"K": 8.3, "C": 8.1},
    "CT02": {"K": 8.4, "C": 6.9},
    "CT03": {"K": 8.1, "C": 8.1},
    "CT04": {"K": 7.7, "C": 6.7},
}
cut_off_iter = {
    "CT01": 95,
    "CT02": 75,
    "CT03": 90,
    "CT04": 75,
}

data_dir = "./"
W, H = 40.0, 40.0
dist_opp_edge = 23.0
notch_tip_x = W - dist_opp_edge  # 17.0 mm


def plot_specimen_outline(ax, params, lw=1, color="#d0d0d0"):
    v_depth = 1.5
    notch_shoulder_x = notch_tip_x - v_depth
    hole_x, hole_y, hole_r = notch_tip_x + params["K"], params["C"], 3.5
    load_hole_x, load_hole_y, load_hole_r = 10.5, 20 - 9.2, 9.5 / 2

    boundary_pts = [
        (0, H / 2),
        (W, H / 2),
        (W, -H / 2),
        (0, -H / 2),
        (0, -1.5),
        (notch_shoulder_x, -1.5),
        (notch_tip_x, 0),
        (notch_shoulder_x, 1.5),
        (0, 1.5),
        (0, H / 2),
    ]
    ax.add_patch(
        PathPatch(
            Path(boundary_pts), facecolor="#f9f9f9", edgecolor=color, lw=lw, zorder=0
        )
    )
    for h_x, h_y, h_r in [
        (hole_x, hole_y, hole_r),
        (load_hole_x, load_hole_y, load_hole_r),
        (load_hole_x, -load_hole_y, load_hole_r),
    ]:
        ax.add_patch(
            Circle((h_x, h_y), h_r, facecolor="white", edgecolor=color, lw=lw, zorder=1)
        )


# 3. Data Loading
current_specimen = None
for i in range(1, 5):
    name = f"CT0{i}"
    try:
        experimental = np.genfromtxt(
            f"{data_dir}{name}_experimental.csv", delimiter=","
        )
        predicted_2003 = np.genfromtxt(
            f"{data_dir}{name}_predicted_miranda_2003.csv", delimiter=","
        )
        predicted_2018 = np.genfromtxt(
            f"{data_dir}{name}_predicted_miranda_2018.csv", delimiter=","
        )
        current_specimen = name
        break
    except OSError:
        continue

assert current_specimen is not None

# --- Simulation Logic (Corrected Alignment) ---
try:
    sim_raw = np.genfromtxt("failed_growth_splines.csv", delimiter=",", skip_header=1)
except OSError:
    sim_raw = np.genfromtxt(
        "successful_growth_splines.csv", delimiter=",", skip_header=1
    )

last_iter = np.max(sim_raw[:, 0])
last_iter_data = sim_raw[sim_raw[:, 0] == last_iter, 2:4]
last_iter_data = last_iter_data[: cut_off_iter[current_specimen]]

last_iter_data -= last_iter_data[0, :]
last_iter_data[:, 0] -= 3
last_iter_data = last_iter_data[last_iter_data[:, 0] >= 0]
last_iter_data -= last_iter_data[0, :]  # Re-zero after crop
last_iter_data[:, 0] += notch_tip_x

ctrl_pts = last_iter_data
n_ctrl, k = len(ctrl_pts), 2
knots = np.concatenate(([0] * k, np.linspace(0, 1, n_ctrl - k + 1), [1] * k))
spline_x = BSpline(knots, ctrl_pts[:, 0], k)
spline_y = BSpline(knots, ctrl_pts[:, 1], k)
t_eval = np.linspace(0, 1, 500)
x_sim, y_sim = spline_x(t_eval), spline_y(t_eval)

# --- Paper Data ---
experimental[:, 0] *= -1
predicted_2003[:, 0] *= -1
for arr in [experimental, predicted_2003, predicted_2018]:
    arr[:] = arr[arr[:, 0].argsort()]
    arr -= arr[0, :]
    arr[:, 0] += notch_tip_x

# 4. Final Standalone Plotting
fig, ax = plt.subplots(figsize=(8, 7))  # Adjusted ratio for the zoomed view
plot_specimen_outline(ax, specimen_params[current_specimen])

ax.plot(experimental[:, 0], experimental[:, 1], "k-", label="Experimental")
ax.plot(predicted_2003[:, 0], predicted_2003[:, 1], "b--", label="Miranda 2003")
ax.plot(predicted_2018[:, 0], predicted_2018[:, 1], "g-.", label="Miranda 2018")
ax.plot(x_sim, y_sim, "r-", linewidth=2.5, label="Present work", zorder=5)

# Calculate dynamic limits to encompass the crack AND the upper hole
params = specimen_params[current_specimen]
hole_x = notch_tip_x + params["K"]
hole_y = params["C"]
hole_r = 3.5

# Find the bounding box that captures everything we want to see
x_min = min(np.min(x_sim), notch_tip_x) - 1.5
x_max = max(np.max(x_sim), hole_x + hole_r) + 2.0
y_min = np.min(y_sim) - 2.5
y_max = max(np.max(y_sim), hole_y + hole_r) + 2.0

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect("equal")

# Formatting
ax.legend(loc="upper left", fontsize=11)
ax.set_xlabel("x (mm)", fontsize=12)
ax.set_ylabel("y (mm)", fontsize=12)
ax.set_title(f"MCTS Crack Path Comparison: {current_specimen}", fontsize=14)

plt.grid(True, linestyle=":", alpha=0.4)
plt.tight_layout()
plt.savefig(f"MCTS_path_zoomed_{current_specimen}.pdf", dpi=300)
plt.show()
