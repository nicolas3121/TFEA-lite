import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from scipy.interpolate import BSpline
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 1. Configuration
specimen_params = {
    "CT01": {"K": 8.3, "C": 8.1},
    "CT02": {"K": 8.4, "C": 6.9},
    "CT03": {"K": 8.1, "C": 8.1},
    "CT04": {"K": 7.7, "C": 6.7},
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
sim_raw = np.genfromtxt("failed_growth_splines.csv", delimiter=",", skip_header=1)
last_iter = np.max(sim_raw[:, 0])
last_iter_data = sim_raw[sim_raw[:, 0] == last_iter, 2:4]

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

# 4. Final Plotting
fig, ax = plt.subplots(figsize=(12, 10))
plot_specimen_outline(ax, specimen_params[current_specimen])

ax.plot(experimental[:, 0], experimental[:, 1], "k-", label="Experimental")
ax.plot(predicted_2003[:, 0], predicted_2003[:, 1], "b--", label="Miranda 2003")
ax.plot(predicted_2018[:, 0], predicted_2018[:, 1], "g-.", label="Miranda 2018")
ax.plot(x_sim, y_sim, "r-", linewidth=2.5, label="My Prediction", zorder=5)

# --- 5. ZOOM INSET (Bigger and Tighter) ---
# Width/Height increased to 50% for a larger box
ax_ins = inset_axes(ax, width="50%", height="50%", loc="lower right", borderpad=1.5)
ax_ins.set_aspect("equal")

plot_specimen_outline(
    ax_ins, specimen_params[current_specimen], lw=0.5, color="#e0e0e0"
)
ax_ins.plot(experimental[:, 0], experimental[:, 1], "k-", lw=1.5)
ax_ins.plot(predicted_2003[:, 0], predicted_2003[:, 1], "b--", lw=1.2)
ax_ins.plot(predicted_2018[:, 0], predicted_2018[:, 1], "g-.", lw=1.2)
ax_ins.plot(x_sim, y_sim, "r-", linewidth=2.5, zorder=5)

# ZOOM SETTINGS
# Reduced padding to 0.75mm to "zoom in more" on the crack path
padding = 0.75
ax_ins.set_xlim(np.min(x_sim) - padding, np.max(x_sim) + padding)
ax_ins.set_ylim(np.min(y_sim) - padding, np.max(y_sim) + padding)

# Hide ticks for a clean look
ax_ins.tick_params(labelleft=False, labelbottom=False)

# Connectors
mark_inset(
    ax, ax_ins, loc1=2, loc2=4, fc="none", ec="0.5", lw=1, linestyle="--", alpha=0.6
)

# Main Axis Formatting
ax.set_aspect("equal")
ax.set_xlim(0, 40)
ax.set_ylim(-20, 20)
ax.legend(loc="upper left", fontsize=12)
ax.set_xlabel("x (mm)", fontsize=12)
ax.set_ylabel("y (mm)", fontsize=12)
ax.set_title(f"MCTS Crack Path Comparison: {current_specimen}", fontsize=14)

plt.grid(True, linestyle=":", alpha=0.4)
plt.show()
