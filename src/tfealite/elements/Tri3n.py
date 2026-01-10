import numpy as np


class Tri3n:
    def __init__(self, node_coords, material, real):
        self.node_coords = node_coords
        self.E = material["E"]
        self.nu = material["nu"]
        self.rho = material["rho"] if ("rho" in material) else 0.0
        self.t = real["t"]

        c1 = self.E / (1.0 - self.nu**2)
        self.C = c1 * np.array(
            [
                [1.0, self.nu, 0.0],
                [self.nu, 1.0, 0.0],
                [0.0, 0.0, (1.0 - self.nu) / 2.0],
            ]
        )

    def cal_element_matrices(self, eval_mass=False):
        xi, eta = 1 / 3, 1 / 3
        weight = 1 / 2

        Ke = np.zeros((6, 6))
        Me = np.zeros((6, 6)) if eval_mass else None
        x_e = self.node_coords

        N, dN_dxi = self.shape_functions(xi, eta)
        J = dN_dxi @ x_e
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)

        B = np.zeros((3, 6))
        dN_dxy = invJ @ dN_dxi
        for i in range(3):
            ix = 2 * i
            iy = 2 * i + 1
            B[0, ix] = dN_dxy[0, i]
            B[1, iy] = dN_dxy[1, i]
            B[2, ix] = dN_dxy[1, i]
            B[2, iy] = dN_dxy[0, i]
        Ke += (B.T @ self.C @ B) * detJ * weight * self.t
        if eval_mass:
            rho_t = self.rho * self.t
            N_2d = np.zeros((2, 6))
            for i in range(3):
                N_2d[0, 2 * i] = N[i]
                N_2d[1, 2 * i + 1] = N[i]
            Me += rho_t * (N_2d.T @ N_2d) * detJ * weight
            return Me, Ke
        else:
            return Ke

    def shape_functions(self, xi, eta):
        N = np.array([1 - xi - eta, xi, eta])
        dN_dxi = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
        return N, dN_dxi

    def stresses_at_nodes(self, Ue):
        Ue = np.asanyarray(Ue, dtype=float).ravel()
        # same at all points in triangle
        xi, eta = 1 / 3, 1 / 3

        x_e = self.node_coords

        _, dN_dxi = self.shape_functions(xi, eta)
        J = dN_dxi @ x_e
        invJ = np.linalg.inv(J)

        B = np.zeros((3, 6))
        dN_dxy = invJ @ dN_dxi
        for i in range(3):
            ix = 2 * i
            iy = 2 * i + 1
            B[0, ix] = dN_dxy[0, i]
            B[1, iy] = dN_dxy[1, i]
            B[2, ix] = dN_dxy[1, i]
            B[2, iy] = dN_dxy[0, i]

        eps = B @ Ue
        sig = self.C @ eps

        return np.tile(sig, (3, 1))
