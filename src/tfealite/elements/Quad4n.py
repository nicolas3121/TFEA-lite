import numpy as np
from .utils import cal_B_2d_vec
from ..core.quadratures import QUAD_RULES


class Quad4n:
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
        rule, correction = QUAD_RULES[3]
        Me = np.zeros((8, 8)) if eval_mass else None
        x_e = self.node_coords
        xi = rule[:, 0]
        eta = rule[:, 1]
        N, dN_dxi = Quad4n.shape_functions2(self, xi, eta)
        J = dN_dxi @ x_e
        detJ = np.linalg.det(J)
        dN_dxy = np.linalg.solve(J, dN_dxi)
        B = cal_B_2d_vec(dN_dxy)
        w_eff = rule[:, 2] * correction * detJ
        Ke = np.sum((B.transpose(0, 2, 1) @ self.C @ B) * w_eff[:, None, None], axis=0)
        if eval_mass:
            rho_t = self.rho
            N_2d = np.empty((xi.shape[0], 2, 8))
            N_2d[:, 0, ::2] = N[:, :]
            N_2d[:, 1, 1::2] = N[:, :]
            Me = np.sum(
                rho_t * (N_2d.transpose(0, 2, 1) @ N_2d) * w_eff[:, None, None], axis=0
            )
            return Me, Ke
        return Ke

    def cal_element_matrices2(self, eval_mass=False):
        gauss_pts = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        Ke = np.zeros((8, 8))
        Me = np.zeros((8, 8)) if eval_mass else None
        x_e = self.node_coords
        for xi in gauss_pts:
            for eta in gauss_pts:
                N, dN_dxi = Quad4n.shape_functions(self, xi, eta)
                J = dN_dxi @ x_e
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                B = np.zeros((3, 8))
                dN_dxy = invJ @ dN_dxi
                for i in range(4):
                    ix = 2 * i
                    iy = 2 * i + 1
                    B[0, ix] = dN_dxy[0, i]
                    B[1, iy] = dN_dxy[1, i]
                    B[2, ix] = dN_dxy[1, i]
                    B[2, iy] = dN_dxy[0, i]
                Ke += (B.T @ self.C @ B) * detJ
                if eval_mass:
                    rho_t = self.rho
                    N_2d = np.zeros((2, 8))
                    for i in range(4):
                        N_2d[0, 2 * i] = N[i]
                        N_2d[1, 2 * i + 1] = N[i]
                    Me += rho_t * (N_2d.T @ N_2d) * detJ
        if eval_mass:
            return Me, Ke
        else:
            return Ke

    def shape_functions(self, xi, eta):
        N = 0.25 * np.array(
            [
                (1 - xi) * (1 - eta),
                (1 + xi) * (1 - eta),
                (1 + xi) * (1 + eta),
                (1 - xi) * (1 + eta),
            ]
        )
        dN_dxi = 0.25 * np.array(
            [
                [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
            ]
        )
        return N, dN_dxi

    def shape_functions2(self, xi, eta):
        xi_min = 1 - xi
        xi_plus = 1 + xi
        eta_min = 1 - eta
        eta_plus = 1 + eta
        N = (
            0.25
            * np.array(
                [
                    xi_min * eta_min,
                    xi_plus * eta_min,
                    xi_plus * eta_plus,
                    xi_min * eta_plus,
                ]
            ).T
        )
        row1 = [-eta_min, eta_min, eta_plus, -eta_plus]
        row2 = [-xi_min, -xi_plus, xi_plus, xi_min]
        dN_dxi = 0.25 * np.stack([row1, row2]).transpose(2, 0, 1)
        return N, dN_dxi

    def cal_stresses(self, xi, eta, Ue):
        Ue = np.asarray(Ue, dtype=float).ravel()
        _, dN_dxi = self.shape_functions2(xi, eta)
        J = dN_dxi @ self.node_coords
        dN_dxy = np.linalg.solve(J, dN_dxi)
        B = cal_B_2d_vec(dN_dxy)
        eps = B @ Ue
        sig = self.C @ eps[:, :, None]
        return sig.reshape(-1, 3)

    def stresses_at_nodes(self, Ue):
        xi = np.array([-1.0, 1.0, 1.0, -1.0])
        eta = np.array([-1.0, -1.0, 1.0, 1.0])
        return self.cal_stresses(xi, eta, Ue)
