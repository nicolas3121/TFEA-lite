import matplotlib.pyplot as plt
import numpy as np


def get_analytical_displacements(
    nu=0.30, plane_strain=True, xc=5.0, yc=5.0, crack_angle=0.0
):
    kappa = 3.0 - 4.0 * nu if plane_strain else (3.0 - nu) / (1.0 + nu)
    c, s = np.cos(crack_angle), np.sin(crack_angle)

    def eval_displacements(x, y):
        dx, dy = x - xc, y - yc

        # Rotate global points into local crack tip coordinate system
        x_loc = c * dx + s * dy
        y_loc = -s * dx + c * dy

        r = np.sqrt(x_loc**2 + y_loc**2)
        theta = np.arctan2(y_loc, x_loc)

        u_x_loc = np.sqrt(r) * (
            (kappa - 0.5) * np.cos(theta / 2) - 0.5 * np.cos(3 * theta / 2)
        )
        u_y_loc = np.sqrt(r) * (
            (kappa + 0.5) * np.sin(theta / 2) - 0.5 * np.sin(3 * theta / 2)
        )

        # Rotate displacements back to global system
        u_x = c * u_x_loc - s * u_y_loc
        u_y = s * u_x_loc + c * u_y_loc
        return u_x, u_y

    return eval_displacements


# --- Plotting Script ---
L = 10.0
xc, yc = 5.0, 5.0

# Generate coordinates along the four boundaries
n_pts = 15  # Number of arrows per edge
s = np.linspace(0, L, n_pts)

# Bottom, Right, Top, Left edges
xb = s
yb = np.zeros_like(s)
xr = np.full_like(s, L)
yr = s
xt = s
yt = np.full_like(s, L)
xl = np.zeros_like(s)
yl = s

# Combine boundary arrays
x_bounds = np.concatenate([xb, xr, xt, xl])
y_bounds = np.concatenate([yb, yr, yt, yl])

# Initialize the displacement function and calculate vectors
eval_disp = get_analytical_displacements(xc=xc, yc=yc)
u_x, u_y = eval_disp(x_bounds, y_bounds)

# --- Create the Figure ---
fig, ax = plt.subplots(figsize=(3.5, 3.5))

# Draw the square domain boundary
ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], "k-", lw=1.5)


# Plot the displacement vectors using quiver
# Note: 'scale' controls the arrow length. Decrease scale to make arrows longer.
ax.quiver(
    x_bounds,
    y_bounds,
    u_x,
    u_y,
    color="red",
    angles="xy",
    scale_units="xy",
    scale=2.5,
    width=0.004,
    zorder=3,
    label="Displacement Vectors",
)

# Draw the crack
ax.plot([0, xc], [yc, yc], color="cyan", lw=2)
ax.text(xc, yc + 0.2, "(5, 5)", ha="left", va="bottom", fontsize=12, zorder=4)

# Add corner labels
offset = 0.3
ax.text(0, 0 - offset, "(0, 0)", ha="center", va="top", fontsize=12, zorder=4)
ax.text(L, 0 - offset, "(10, 0)", ha="center", va="top", fontsize=12, zorder=4)
ax.text(L, L + offset, "(10, 10)", ha="center", va="bottom", fontsize=12, zorder=4)
ax.text(0, L + offset, "(0, 10)", ha="center", va="bottom", fontsize=12, zorder=4)

# Draw custom coordinate axes at the bottom left (matching your image)
ax.annotate(
    "", xy=(2, -1.5), xytext=(-1.5, -1.5), arrowprops={"arrowstyle": "->", "lw": 1.5}
)
ax.annotate(
    "", xy=(-1.5, 2), xytext=(-1.5, -1.5), arrowprops={"arrowstyle": "->", "lw": 1.5}
)
ax.text(2, -1.5, " x", ha="left", va="center", fontsize=14)
ax.text(-1.5, 2, " y", ha="center", va="bottom", fontsize=14)

# Formatting
ax.set_aspect("equal")
ax.set_xlim(-2.5, L + 2.5)
ax.set_ylim(-2.5, L + 2.5)
ax.axis("off")  # Hide the default matplotlib axes box
# plt.legend(loc="lower right")
plt.savefig("convergence_test_model.pdf")

plt.show()
