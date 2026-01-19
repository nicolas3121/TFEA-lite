from .Tri3n import Tri3n
from typing import Final
import numpy as np

NODES: Final = 3
DOFS: Final = 2
BRANCH_FN: Final = 4  # branch functions
N_FN: Final = NODES
H_FN: Final = NODES
TIP_FN: Final = NODES * BRANCH_FN
N_DOFS: Final = DOFS * N_FN
H_DOFS: Final = DOFS * H_FN
TIP_DOFS: Final = DOFS * TIP_FN


class XTri3n(Tri3n):
    def __init__(
        self,
        node_coords,
        phi_n,
        phi_t,
        h_enrich: bool,
        t_enrich: bool,
        partial_cut: bool,
        material,
        real,
    ):
        super().__init__(node_coords, material, real)
        self.phi_n = phi_n
        self.phi_t = phi_t
        self.h_enrich = h_enrich
        self.t_enrich = t_enrich
        self.partial_cut = partial_cut

    def cal_element_matrices(self, eval_mass=False):
        if not self.h_enrich and not self.t_enrich:
            return super().cal_element_matrices(eval_mass)
        n = N_DOFS + int(self.h_enrich) * H_DOFS + int(self.t_enrich) * TIP_DOFS
        Ke = np.zeros((n, n))
        x_e = self.node_coords

        xi, eta = 1 / 3, 1 / 3
        weight = 1 / 2

        N, dN_dxi = super().shape_functions(xi, eta)
        J = dN_dxi @ x_e
        detJ = np.linalg.det(J)

        B = np.zeros((3, 6))
        dN_dxy = np.linalg.solve(J, dN_dxi)
        B[0, ::2] = dN_dxy[0, :]
        B[1, 1::2] = dN_dxy[1, :]
        B[2, ::2] = dN_dxy[1, :]
        B[2, 1::2] = dN_dxy[0, :]
        # for i in range(3):
        #     ix = 2 * i
        #     iy = 2 * i + 1
        #     B[0, ix] = dN_dxy[0, i]
        #     B[1, iy] = dN_dxy[1, i]
        #     B[2, ix] = dN_dxy[1, i]
        #     B[2, iy] = dN_dxy[0, i]

        Ke[0:6, 0:6] = B.T @ self.C @ B

        Nc = np.zeros((NODES, 3))
        # 4 exceptions
        # snijdt een zijde niet --> projecteren op zijde --> 1, 0
        # valt samen met een zijde --> oneindig veel oplossingen --> kan willekeurig punt kiezen eg 1 van de vertices
        # parallel aan zijde --> geen oplossing --> kies 1 van de vertices als 1, andere 0
        # gaat door een node
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
        if self.partial_cut:
            x, w = np.polynomial.legendre.leggauss(40)
            print(x.shape)
            x = (1 + x) / 2
            w /= 2
            Ni_template = np.zeros((3, 3))
            Ni_template[:, 0] = np.linalg.solve(
                np.array([self.phi_t, self.phi_n, [1, 1, 1]]), np.array([0, 0, 1])
            )
            for i in range(6):  # singularity verplaatst naar 1ste vertex
                Ni = Ni_template.copy()
                Ni[int((i % 5 + 1) / 2), 1 + i % 2] = 1
                Ni[:, 2 - i % 2] = Nc[:, int(i / 2)]
                detJi = np.linalg.det(Ni)
                if np.isclose(detJi, 0):
                    continue
                for u0, w1 in zip(x, w):
                    for v0, w2 in zip(x, w):
                        beta = 1
                        w_eff = w1 * w2 * beta * u0 ** (2 * beta - 1)
                        u = u0**beta
                        n = np.array([2 * u, u * (1 - v0), u * v0])
                        xi_sub, eta_sub = np.linalg.solve(
                            J.T, x_e.T @ Ni @ n - x_e[0, :]
                        )
                        N_sub, dN_dxi_sub = self.shape_functions(xi_sub, eta_sub)
                        dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
                        TIP_B = np.zeros((3, TIP_DOFS))
                        begin_tip = N_FN + int(self.h_enrich) * H_FN
                        TIP_B[0, ::DOFS] = dN_dxy_sub[0, begin_tip:]
                        TIP_B[1, 1::DOFS] = dN_dxy_sub[1, begin_tip:]
                        TIP_B[2, ::DOFS] = dN_dxy_sub[1, begin_tip:]
                        TIP_B[2, 1::DOFS] = dN_dxy_sub[0, begin_tip:]
                        # print(TIP_B.shape, self.C.shape)
                        begin_tip *= DOFS
                        Ke[begin_tip:, begin_tip:] += (
                            (TIP_B.T @ self.C @ TIP_B) * w_eff * detJi
                        )
                        beta = 2
                        w_eff = w1 * w2 * beta * u0 ** (2 * beta - 1)
                        u = u0**beta
                        n = np.array([2 * u, u * (1 - v0), u * v0])
                        xi_sub, eta_sub = np.linalg.solve(
                            J.T, x_e.T @ Ni @ n - x_e[0, :]
                        )
                        N_sub, dN_dxi_sub = self.shape_functions(xi_sub, eta_sub)
                        dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
                        TIP_B = np.zeros((3, TIP_DOFS))
                        begin_tip = N_FN + int(self.h_enrich) * H_FN
                        TIP_B[0, ::DOFS] = dN_dxy_sub[0, begin_tip:]
                        TIP_B[1, 1::DOFS] = dN_dxy_sub[1, begin_tip:]
                        TIP_B[2, ::DOFS] = dN_dxy_sub[1, begin_tip:]
                        TIP_B[2, 1::DOFS] = dN_dxy_sub[0, begin_tip:]
                        # print(TIP_B.shape, self.C.shape)
                        begin_tip *= DOFS
                        Ke[0:begin_tip, begin_tip:] += (
                            (B.T @ self.C @ TIP_B) * w_eff * detJi
                        )
                        Ke[begin_tip:, 0:begin_tip] += Ke[0:begin_tip, begin_tip:].T

                # zorgen dat alle punten voor sub triangle 1 keer berekent worden en dan voor alle nodige integraties gebruikt worden
                # momenteel is crack tip 3de node moet eerste worden --> cyclisch doorschuiven
        else:
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
                xi_sub, eta_sub = np.linalg.solve(J.T, x_e.T @ Ni @ N - x_e[0, :])
                N_sub, _ = super().shape_functions(xi_sub, eta_sub)
                h_shifted = (
                    np.sign(np.dot(self.phi_n, N_sub)) - np.sign(self.phi_n)
                ) / 2
                HB = B.copy()
                HB[:, ::DOFS] *= h_shifted
                HB[:, 1::DOFS] *= h_shifted
                Ke[6:12, 6:12] += (HB.T @ self.C @ HB) * detJi
                Ke[0:6, 6:12] += (B.T @ self.C @ HB) * detJi
                Ke[6:12, 0:6] += Ke[0:6, 6:12].T
            Ke *= detJ * weight * self.t

        if eval_mass:
            raise NotImplementedError
        else:
            return Ke

    # TODO: compatibel maken met xi, eta als arrays
    def shape_functions(self, xi, eta):
        N = np.zeros(N_FN + int(self.h_enrich) * H_FN + int(self.t_enrich) * TIP_FN)
        dN_dxi = np.zeros(
            (
                DOFS,
                N_FN + int(self.h_enrich) * H_FN + int(self.t_enrich) * TIP_FN,
            )
        )
        (N[:N_FN], dN_dxi[:, :N_FN]) = super().shape_functions(xi, eta)
        phi_n = np.dot(self.phi_n, N[:N_FN])
        phi_t = np.dot(self.phi_t, N[:N_FN])
        if self.h_enrich:
            h_shifted = (np.sign(phi_n) - np.sign(self.phi_n)) / 2
            begin_h, end_h = N_FN, N_FN + H_FN
            N[begin_h:end_h] = h_shifted * N[:N_FN]
            dN_dxi[:, begin_h:end_h] = h_shifted * dN_dxi[:, :N_FN]
        if self.t_enrich:
            # TODO: handle divide by 0 if phi_n or phi_t 0
            r = np.sqrt(phi_n**2 + phi_t**2)
            sqrt_r = np.sqrt(r)
            sqrt_r_i = (self.phi_n**2 + self.phi_t**2) ** (1 / 4)
            theta = np.atan2(phi_n, -phi_t)
            theta_i = np.atan2(self.phi_n, -self.phi_t)
            dphi_n_dxi = np.sum(self.phi_n * dN_dxi[:, :N_FN], axis=1)
            dphi_t_dxi = np.sum(self.phi_t * dN_dxi[:, :N_FN], axis=1)
            # sin(theta) = phi_n / r, cos(theta) = -phi_t / r
            dr_dxi = (
                1 / r * (phi_n * dphi_n_dxi + phi_t * dphi_t_dxi)
            )  # = np.sin(theta) * dphi_n_dxi - np.cos(theta) * dphi_t_dxi
            dtheta_dxi = (
                -1 * (dphi_n_dxi * phi_t - phi_n * dphi_t_dxi) / (phi_t**2 + phi_n**2)
            )  # = (dphi_n_dxi * np.cos(theta) + np.sin(theta) dphi_t_dxi) / r
            bf = branch_functions(sqrt_r, theta)
            bf_i = branch_functions(sqrt_r_i, theta_i)
            dbf_dxi = (
                1 / (2 * sqrt_r) * dr_dxi.reshape((2, 1)) * (bf / sqrt_r)
                + sqrt_r
                * np.array(
                    [
                        -np.sin(theta / 2) * dtheta_dxi / 2,
                        np.cos(theta / 2) * dtheta_dxi / 2,
                        np.cos(theta / 2) * dtheta_dxi / 2 * np.sin(theta)
                        + np.sin(theta / 2) * np.cos(theta) * dtheta_dxi,
                        -np.sin(theta / 2) * dtheta_dxi / 2 * np.sin(theta)
                        + np.cos(theta / 2) * np.cos(theta) * dtheta_dxi,
                    ]
                ).T
            )
            if self.h_enrich:
                begin_tip, end_tip = N_FN + H_FN, N_FN + H_FN + TIP_FN
                N[begin_tip:end_tip] = (
                    (
                        bf
                        - np.sign(phi_n) * np.sign(self.phi_n).reshape((-1, 1)) * bf_i.T
                    )
                    * N[:N_FN].reshape((-1, 1))
                ).reshape((1, TIP_FN))
            else:
                begin_tip, end_tip = N_FN, N_FN + TIP_FN
                N[begin_tip:end_tip] = (
                    (bf - bf_i.T) * N[:N_FN].reshape((-1, 1))
                ).reshape((1, TIP_FN))
            term1 = dbf_dxi * N[:N_FN, None, None]  # (3, 2, 4)
            term2 = (
                N[begin_tip:end_tip].reshape((-1, BRANCH_FN))[:, :, None]
                * dN_dxi[:, :N_FN].T[:, None, :]
            )  # (3, 4, 2)
            dN_dxi[0, begin_tip:end_tip] = (term1[:, 0, :] + term2[:, :, 0]).flatten()
            dN_dxi[1, begin_tip:end_tip] = (term1[:, 1, :] + term2[:, :, 1]).flatten()
            # for i in range(0, TIP_FN, BRANCH_FN):
            #     node = int(i / BRANCH_FN)
            #     term1 = dbf_dxi * N[node]
            #     term2 = N[begin_tip + i : begin_tip + i + BRANCH_FN].reshape(
            #         (1, BRANCH_FN)
            #     ) * dN_dxi[:, node].reshape((-1, 1))
            #     dN_dxi[:, begin_tip + i : begin_tip + i + BRANCH_FN] = term1 + term2
        return N, dN_dxi

    def stresses_at_nodes(self, Ue):
        raise NotImplementedError


def branch_functions(sqrt_r, theta):
    return sqrt_r * np.array(
        [
            np.cos(theta / 2),
            np.sin(theta / 2),
            np.sin(theta / 2) * np.sin(theta),
            np.cos(theta / 2) * np.sin(theta),
        ]
    )
