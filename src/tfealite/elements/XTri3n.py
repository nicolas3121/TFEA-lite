from .Tri3n import Tri3n
import numpy as np


class XTri3n(Tri3n):
    def __init__(
        self, node_coords, phi_n, phi_t, h_enrich: bool, t_enrich: bool, material, real
    ):
        super().__init__(node_coords, material, real)
        self.phi_n = phi_n
        self.phi_t = phi_t
        self.h_enrich = h_enrich
        self.t_enrich = t_enrich

    def cal_element_matrices(self, eval_mass=False):
        if not self.h_enrich and not self.t_enrich:
            return super().cal_element_matrices(eval_mass)
        n = 6 + int(self.h_enrich) * 6 + int(self.t_enrich) * 24
        Ke = np.zeros((n, n))
        x_e = self.node_coords

        xi, eta = 1 / 3, 1 / 3
        weight = 1 / 2

        N, dN_dxi = super().shape_functions(xi, eta)
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

        Ke[0:6, 0:6] = B.T @ self.C @ B

        Nc = np.zeros((3, 3))
        # 4 exceptions
        # snijdt een zijde niet --> projecteren op zijde --> 1, 0
        # valt samen met een zijde --> oneindig veel oplossingen --> kan willekeurig punt kiezen eg 1 van de vertices
        # parallel aan zijde --> geen oplossing --> kies 1 van de vertices als 1, andere 0
        for i in range(3):
            j = (i + 1) % 3
            phi_i = self.phi_n[i]
            phi_j = self.phi_n[j]
            if np.isclose(phi_i, 0) or np.isclose(phi_i, phi_j):
                Nc[j, i] = 0
                Nc[i, i] = 1
            else:
                Nc[j, i] = np.clip(1 / (1 - phi_j / phi_i), 0, 1)
                Nc[i, i] = np.clip(1 - Nc[j, i], 0, 1)
        for i in range(4):
            if i != 3:
                Ni = np.eye(3)
                Ni[:, (i + 1) % 3] = Ni[:, i]
                Ni[:, (i + 2) % 3] = Ni[:, (i + 2) % 3]
            else:
                Ni = Nc.copy()
            detJi = np.linalg.det(Ni)
            if np.isclose(detJi, 0):
                continue
            xi_sub, eta_sub = invJ.T @ (x_e.T @ Ni @ N - x_e[0, :])
            N_sub, _ = super().shape_functions(xi_sub, eta_sub)
            h_shifted = np.sign(np.dot(self.phi_n, N_sub)) - np.sign(self.phi_n)
            HB = np.zeros((3, 6))
            for i in range(3):
                ix = 2 * i
                iy = 2 * i + 1
                HB[0, ix] = h_shifted[i] * dN_dxy[0, i]
                HB[1, iy] = h_shifted[i] * dN_dxy[1, i]
                HB[2, ix] = h_shifted[i] * dN_dxy[1, i]
                HB[2, iy] = h_shifted[i] * dN_dxy[0, i]
            Ke[6:12, 6:12] += (HB.T @ self.C @ HB) * detJi
            Ke[0:6, 6:12] += (B.T @ self.C @ HB) * detJi
            Ke[6:12, 0:6] += Ke[0:6, 6:12].T
        Ke *= detJ * weight * self.t

        if eval_mass:
            raise NotImplementedError
        else:
            return Ke

    def shape_functions(self, xi, eta):
        N = np.zeros(3 + int(self.h_enrich) * 3 + int(self.t_enrich) * 12)
        dN_dxi = np.zeros_like(
            (2, 3 + int(self.h_enrich) * 3 + int(self.t_enrich) * 12)
        )
        (N[0:3], dN_dxi[:, 0:3]) = super().shape_functions(xi, eta)
        if self.h_enrich:
            h_shifted = np.sign(np.dot(self.phi_n, N[0:3])) - np.sign(self.phi_n)
            N[3:6] = h_shifted * N[0:3]
            dN_dxi[:, 3:6] = h_shifted * dN_dxi[:, 0:3]
        if self.t_enrich:
            raise NotImplementedError
        return N, dN_dxi

    def stresses_at_nodes(self, Ue):
        raise NotImplementedError
