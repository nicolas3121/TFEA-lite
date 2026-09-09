import matplotlib.pyplot as plt
import numpy as np

percentage = {
    "CTS01": 85,
    "CTS02": 75,
    "CTS03": 85,
    "CTS04": 75,
    "SEN": 100,
}
data_dir = "./"

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
        current_specimen = f"CTS0{i}"
        break
    except OSError:
        continue

if current_specimen is None:
    current_specimen = "SEN"
# assert current_specimen is not None

try:
    sim_raw = np.genfromtxt("failed_growth_sifs.csv", delimiter=",", skip_header=1)
except OSError:
    sim_raw = np.genfromtxt("successful_growth_sifs.csv", delimiter=",", skip_header=1)

# 1. Load the data
# (Replace io.StringIO(csv_data) with 'your_file.csv' in your actual code)
data = sim_raw[: percentage[current_specimen]]

# Read the CSV into a 2D NumPy array (skip the header row)
# data = np.genfromtxt(io.StringIO(csv_data), delimiter=",", skip_header=1)
# data = np.genfromtxt('crack_data.csv', delimiter=',', skip_header=1) # Use this for a file

# Extract columns into 1D arrays for easier math
iteration = data[:, 0]
KI = data[:, 1]
KII = data[:, 2]

# 2. Calculate Crack Length
initial_length = 2.5  # mm
increment = 0.15  # mm
crack_length = initial_length + (iteration * increment)

# 3. Calculate Direction Change (Angle theta) using the MTS criterion
theta_rad = 2 * np.arctan((-2 * KII) / (KI + np.sqrt(KI**2 + 8 * KII**2)))
theta_deg = np.degrees(theta_rad)

# Display the calculated data as a clean text table
# print(
#     f"{'Iteration':<10} | {'Crack Length':<12} | {'K_I':<18} | {'K_II':<18} | {'Theta (deg)'}"
# )
# print("-" * 78)
# for i in range(len(iteration)):
#     print(
#         f"{int(iteration[i]):<10} | {crack_length[i]:<12.2f} | {KI[i]:<18.4f} | {KII[i]:<18.4f} | {theta_deg[i]:.4f}"
#     )

# 4. Create the Plot
plt.figure(figsize=(8, 5))
plt.plot(
    crack_length, theta_deg, marker="o", linestyle="-", color="#D55E00", linewidth=2
)

# Formatting
plt.axhline(
    0, color="black", linewidth=1, linestyle="--"
)  # Reference line at 0 degrees
plt.title("Crack Propagation Angle vs. Crack Length", fontsize=20)
plt.tick_params(axis="both", which="major", labelsize=18)
plt.xlabel("Crack Length $a$ (mm)", fontsize=18)
plt.ylabel(r"Direction Change $\theta$ (Degrees)", fontsize=18)
plt.grid(True, linestyle=":", alpha=0.7)

plt.tight_layout()
plt.savefig(f"{current_specimen}_angle_plot.pdf")
plt.show()
