import numpy as np
import matplotlib.pyplot as plt

# Setup the data
theta = np.linspace(np.pi / 2, 3 * np.pi / 2, 500)
f1 = np.sin(theta / 2)
pi = np.pi
epsilon = 0.5

# Calculate points
theta_mirrored = pi - epsilon
theta_analytic = pi + epsilon

# Calculate slopes for tangent lines
slope_mirrored = 0.5 * np.cos(theta_mirrored / 2)
slope_analytic = 0.5 * np.cos(theta_analytic / 2)


def get_tangent(x, x_point, slope, y_point):
    return slope * (x - x_point) + y_point


plt.figure(figsize=(9, 6))

# Highlight regions
plt.axvspan(
    np.pi / 2, pi, facecolor="blue", alpha=0.05, label="Valid Sub-element Domain"
)
plt.axvspan(
    pi,
    3 * np.pi / 2,
    facecolor="red",
    alpha=0.05,
    label="Analytic Continuation (Unphysical)",
)

# Plot the main branch function
plt.plot(theta, f1, "k-", linewidth=2.5, label=r"$F_1 = \sin(\theta/2)$")

# Plot the crack interface
plt.axvline(pi, color="gray", linestyle="--", linewidth=2)
plt.text(
    pi - 0.02,
    0.98,
    "Crack Interface",
    rotation=90,
    va="top",
    ha="right",
    color="gray",
    fontsize=10,
)

# Plot the Mirrored Point (Code Evaluation)
y_m = np.sin(theta_mirrored / 2)
plt.plot(theta_mirrored, y_m, "bo", markersize=8)
x_m_tan = np.linspace(theta_mirrored - 0.25, theta_mirrored + 0.25, 10)
plt.plot(
    x_m_tan,
    get_tangent(x_m_tan, theta_mirrored, slope_mirrored, y_m),
    "b-",
    linewidth=2,
)

# Anchor text far away to prevent overlap
plt.annotate(
    "Mirrored Point\n(Evaluated Slope is Positive)",
    xy=(theta_mirrored, y_m),
    xytext=(0.2, 0.5),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", color="blue", connectionstyle="arc3,rad=0.2"),
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.9),
)

# Plot the Analytic Continuation Point (True Physics)
y_a = np.sin(theta_analytic / 2)
plt.plot(theta_analytic, y_a, "ro", markersize=8)
x_a_tan = np.linspace(theta_analytic - 0.25, theta_analytic + 0.25, 10)
plt.plot(
    x_a_tan,
    get_tangent(x_a_tan, theta_analytic, slope_analytic, y_a),
    "r-",
    linewidth=2,
)

# Anchor text far away to prevent overlap
plt.annotate(
    "Analytic Continuation\n(True Slope is Negative)",
    xy=(theta_analytic, y_a),
    xytext=(0.55, 0.5),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", color="red", connectionstyle="arc3,rad=-0.2"),
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9),
)

# --- CUSTOM AXIS FLIP ---
plt.xticks(
    [np.pi / 2, 3 * np.pi / 4, pi, 5 * np.pi / 4, 3 * np.pi / 2],
    [r"$+\pi/2$", r"$+3\pi/4$", r"$+\pi \,\,/\,\, -\pi$", r"$-3\pi/4$", r"$-\pi/2$"],
    fontsize=12,
)
# ------------------------

plt.xlabel("Physical Polar Angle", fontsize=12)
plt.ylabel(r"Branch Function Value", fontsize=12)
# plt.title("Slope Inversion Across the Crack Face Peak", fontsize=14, pad=15)
plt.xlim(np.pi / 2, 3 * np.pi / 2)
plt.legend(loc="lower right", framealpha=1.0)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sign_extension.pdf")
plt.show()
