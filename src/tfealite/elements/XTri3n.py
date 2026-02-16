from .Tri3n import Tri3n
from ..core import quadratures as qd
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

        dN_dxy = np.linalg.solve(J, dN_dxi)
        B = cal_B(dN_dxy)

        Ke[0:6, 0:6] = B.T @ self.C @ B * weight * detJ * self.t

        Nc = np.zeros((NODES, 3))
        # 4 exceptions
        # snijdt een zijde niet --> projecteren op zijde --> 1, 0
        # valt samen met een zijde --> oneindig veel oplossingen --> kan willekeurig punt kiezen eg 1 van de vertices
        # parallel aan zijde --> geen oplossing --> kies 1 van de vertices als 1, andere 0
        # gaat door een node
        if self.h_enrich or self.partial_cut:
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
            self._integrate_partial_cut(Ke, Nc, J, detJ, B)
        elif self.t_enrich and not self.h_enrich:
            (rule, correction) = qd.TRI_RULES[10]
            begin_tip_fn = N_FN
            begin_tip_dofs = N_DOFS
            for [xi, eta, w] in rule:
                N, dN_dxi = self.shape_functions(xi, eta)
                dN_dxy = np.linalg.solve(J, dN_dxi)
                TIP_B = cal_B(dN_dxy[:, begin_tip_fn:])
                w_eff = w * correction * detJ * self.t
                Ke[begin_tip_dofs:, begin_tip_dofs:] += TIP_B.T @ self.C @ TIP_B * w_eff
                res = B.T @ self.C @ TIP_B * w_eff
                Ke[0:begin_tip_dofs, begin_tip_dofs:] += res
                Ke[begin_tip_dofs:, 0:begin_tip_dofs] += res.T
        else:
            begin_h_fn = N_FN
            begin_tip_fn = N_FN + H_FN
            begin_h_dofs = N_DOFS
            begin_tip_dofs = N_DOFS + H_DOFS
            for i in range(4):
                if i != 3:
                    Ni = np.eye(3)
                    Ni[:, (i + 1) % 3] = Nc[:, i]
                    Ni[:, (i + 2) % 3] = Nc[:, (i + 2) % 3]
                else:
                    Ni = Nc.copy()
                detJi = np.linalg.det(Ni)
                if np.isclose(detJi, 0):
                    continue
                xi_sub, eta_sub = np.linalg.solve(J.T, x_e.T @ Ni @ N - x_e[0, :])
                _, dN_dxi_sub = self.shape_functions(xi_sub, eta_sub)
                dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
                HB = cal_B(dN_dxy_sub[:, begin_h_fn:begin_tip_fn])
                w_eff = detJi * detJ * weight * self.t
                Ke[begin_h_dofs:begin_tip_dofs, begin_h_dofs:begin_tip_dofs] += (
                    HB.T @ self.C @ HB
                ) * w_eff
                res = (B.T @ self.C @ HB) * w_eff
                Ke[0:begin_h_dofs, begin_h_dofs:begin_tip_dofs] += res
                Ke[begin_h_dofs:begin_tip_dofs, 0:begin_h_dofs] += res.T
                if self.t_enrich:
                    (rule, correction) = qd.TRI_RULES[10]
                    for [xi, eta, w] in rule:
                        _, dN_dxi_sub = self.shape_functions(xi, eta)
                        dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
                        TIP_B = cal_B(dN_dxy_sub[:, begin_tip_fn:])
                        w_eff = w * correction * detJ * detJi * self.t
                        Ke[begin_tip_dofs:, begin_tip_dofs:] += (
                            TIP_B.T @ self.C @ TIP_B * w_eff
                        )
                        res = B.T @ self.C @ TIP_B * w_eff
                        Ke[0:begin_h_dofs, begin_tip_dofs:] += res
                        Ke[begin_tip_dofs:, 0:begin_h_dofs] += res.T
                        res = HB.T @ self.C @ TIP_B * w_eff
                        Ke[begin_h_dofs:begin_tip_dofs, begin_tip_dofs:] += res
                        Ke[begin_tip_dofs:, begin_h_dofs:begin_tip_dofs] += res.T
        if eval_mass:
            raise NotImplementedError
        else:
            return Ke

    def _integrate_partial_cut(self, Ke, Nc, J, detJ, B):
        x_e = self.node_coords
        (rule, correction) = qd.QUAD_RULES[10]
        rule = rule.copy()
        # rescale from standard quad to unit quad
        rule[:, 0:2] = (1 + rule[:, 0:2]) / 2
        rule[:, 2] /= 4
        Ni_template = np.zeros((3, 3))
        Ni_template[:, 0] = np.linalg.solve(
            np.array([self.phi_t, self.phi_n, [1, 1, 1]]), np.array([0, 0, 1])
        )
        for i in range(6):  # singularity verplaatst naar 1ste vertex
            Ni = Ni_template.copy()
            Ni[int((i % 5 + 1) / 2), 1 + i % 2] = 1
            Ni[:, 2 - i % 2] = Nc[:, int(i / 2)]
            detJi = np.linalg.det(Ni)
            if detJi < 0:
                print("DetJi smaller than 0")
            if np.isclose(detJi, 0):
                continue
            x_e_i = (x_e.T @ Ni).T
            duffy = DuffyDistance(x_e_i)
            for [u, v, w] in rule:
                [xi_d, eta_d, w_d] = duffy.transform(u, v, beta=1)
                w_eff = w * correction * w_d * self.t * detJi * detJ
                n, _ = super().shape_functions(xi_d, eta_d)
                xi_sub, eta_sub = np.linalg.solve(J.T, x_e.T @ Ni @ n - x_e[0, :])
                _, dN_dxi_sub = self.shape_functions(xi_sub, eta_sub)
                dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
                begin_tip = N_FN + int(self.h_enrich) * H_FN
                TIP_B = cal_B(dN_dxy_sub[:, begin_tip:])
                begin_tip *= DOFS
                Ke[begin_tip:, begin_tip:] += (TIP_B.T @ self.C @ TIP_B) * w_eff

                [xi_d, eta_d, w_d] = duffy.transform(u, v, beta=1)
                w_eff = w * correction * w_d * self.t * detJi * detJ
                n, _ = super().shape_functions(xi_d, eta_d)
                xi_sub, eta_sub = np.linalg.solve(J.T, x_e.T @ Ni @ n - x_e[0, :])
                _, dN_dxi_sub = self.shape_functions(xi_sub, eta_sub)
                dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
                begin_tip = N_FN + int(self.h_enrich) * H_FN
                TIP_B = cal_B(dN_dxy_sub[:, begin_tip:])
                # print(TIP_B.shape, self.C.shape)
                begin_tip *= DOFS
                Ke[0:begin_tip, begin_tip:] += (B.T @ self.C @ TIP_B) * w_eff
                Ke[begin_tip:, 0:begin_tip] += (TIP_B.T @ self.C @ B) * w_eff

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
        # print("phi_n", self.phi_n)
        # print("phi_t", self.phi_t)
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
        Ue = np.asanyarray(Ue, dtype=float).ravel()
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


def cal_B(dN_dxy):
    B = np.zeros((3, DOFS * dN_dxy.shape[1]))
    B[0, ::DOFS] = dN_dxy[0, :]
    B[1, 1::DOFS] = dN_dxy[1, :]
    B[2, ::DOFS] = dN_dxy[1, :]
    B[2, 1::DOFS] = dN_dxy[0, :]
    return B


class GeneralizedDuffy:
    def __init__(self, _x_e):
        pass

    def transform(self, u, v, beta):
        u_d = u**beta
        v_d = v
        j_d = beta * u ** (2 * beta - 1)
        xi_ddt = u_d * (1 - v_d)
        eta_ddt = u_d * v_d
        return np.asarray([xi_ddt, eta_ddt, j_d])


class DuffyDistance:
    def __init__(self, x_e):
        r21 = x_e[0] - x_e[1]
        r23 = x_e[2] - x_e[1]
        self.vp = np.dot(r21, r23) / np.dot(r23, r23)
        r2p = r23 * self.vp
        # self.vp = np.linalg.norm(r2p) / np.linalg.norm(r23)
        r1p = x_e[1] + r2p - x_e[0]
        self.d = np.linalg.norm(r1p) / np.linalg.norm(r23)
        self.s_min = np.log(np.sqrt(self.vp**2 + self.d**2) - self.vp)
        self.s_max = np.log(np.sqrt((1 - self.vp) ** 2 + self.d**2) + (1 - self.vp))

    def transform(self, u, v, beta):
        u_ddt = u**beta
        s = self.s_min + v * (self.s_max - self.s_min)
        v_ddt = (np.exp(s) - self.d**2 * np.exp(-s)) / 2 + self.vp
        j_ddt = (
            beta
            * u ** (2 * beta - 1)
            * np.sqrt((v_ddt - self.vp) ** 2 + self.d**2)
            * (self.s_max - self.s_min)
        )
        xi_ddt = u_ddt * (1 - v_ddt)
        eta_ddt = u_ddt * v_ddt
        return np.asarray([xi_ddt, eta_ddt, j_ddt])


class DuffySinh:
    def __init__(self, x_e):
        r21 = x_e[0] - x_e[1]
        r23 = x_e[2] - x_e[1]
        self.vp = np.dot(r21, r23) / np.dot(r23, r23)
        r2p = r23 * self.vp
        # self.vp = np.linalg.norm(r2p) / np.linalg.norm(r23)
        r1p = x_e[1] + r2p - x_e[0]
        self.d = np.linalg.norm(r1p) / np.linalg.norm(r23)
        self.s_min = np.arcsinh(-self.vp / self.d)
        self.s_max = np.arcsinh((1 - self.vp) / self.d)

    def transform(self, u, v, beta):
        u_sinh = u**beta
        s = self.s_min + v * (self.s_max - self.s_min)
        v_sinh = self.vp + self.d * np.sinh(s)
        j_sinh = self.d * np.cosh(s) * (self.s_max - self.s_min)
        j_sinh = (
            beta * u ** (2 * beta - 1) * self.d * np.cosh(s) * (self.s_max - self.s_min)
        )

        xi = u_sinh * (1 - v_sinh)
        eta = u_sinh * v_sinh
        return np.asarray([xi, eta, j_sinh])
