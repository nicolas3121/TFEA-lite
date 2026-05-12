import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 300)
N1 = (1 - x) / 2
N2 = (1 + x) / 2
h = (1 + np.sign(x)) / 2

theta = np.zeros_like(x)
theta[:150] = np.linspace(0, -np.pi, 150)
theta[150:] = np.linspace(np.pi, 0, 150)

interpolant = N2

plt.figure(figsize=(5, 4))
plt.tight_layout()
plt.plot(x, h, label="H")
plt.plot(x, interpolant, label="interpolant")
plt.plot(x, h - interpolant, label="modified H")
plt.xlabel("$\\phi_n$")
plt.legend()
plt.title("Interpolant modification")
plt.savefig("interpolant.png")


plt.figure(figsize=(5, 4))
plt.tight_layout()
plt.plot(x, h, label="H")
plt.plot(x, N1 * (h - 0), label="shifted H node 1")
plt.plot(x, N2 * (h - 1), label="shifted H node 2")
plt.xlabel("$\\phi_n$")
plt.title("Shifting modification")
plt.legend()
plt.savefig("shifting.png")

plt.figure(figsize=(5, 4))
plt.tight_layout()
plt.plot(x, N1, label="N1")
plt.plot(x, N2, label="N2")
plt.title("Hat functions")
plt.legend()
plt.savefig("hat.png")
Q0 = (x - 1) ** 2 * (x + 2) / 4
Q1 = (2 - x) * (x + 1) ** 2 / 4

plt.figure(figsize=(5, 4))
plt.tight_layout()
plt.plot(x, Q0, label="Q0")
plt.plot(x, Q1, label="Q1")
plt.title("Cubic functions")
plt.legend()
plt.savefig("cubic.png")
