from .Quad4n import Quad4n
import numpy as np
from typing import Final
from .utils import branch_functions, cal_B_2d, cal_B_2d_vec
from ..core import quadratures as qd
from ..core.quadratures import DuffyDistance

NODES: Final = 4
DOFS: Final = 2
BRANCH_FN: Final = 4  # branch functions
N_FN: Final = NODES
H_FN: Final = NODES
TIP_FN: Final = NODES * BRANCH_FN
N_DOFS: Final = DOFS * N_FN
H_DOFS: Final = DOFS * H_FN
TIP_DOFS: Final = DOFS * TIP_FN


NAT_1: Final = np.array([[-1, -1], [1, -1], [1, 1]])
NAT_2: Final = np.array([[-1, -1], [1, 1], [-1, 1]])


class XQuad4n(Quad4n):
    def __new__(
        cls,
        node_coords,
        material,
        real,
        phi_n=None,
        phi_t=None,
        h_enrich: bool = False,
        t_enrich: bool = False,
        partial_cut: bool = False,
        h_enrich_per_node=None,
    ):
        if not h_enrich and not t_enrich:
            # print("creating basic element instead")
            return Quad4n(node_coords, material, real)
        assert h_enrich is not None
        assert t_enrich is not None
        assert phi_n is not None
        assert phi_t is not None
        assert partial_cut is not None
        return super().__new__(cls)

    def __init__(
        self,
        node_coords,
        material,
        real,
        phi_n,
        phi_t,
        h_enrich: bool,
        t_enrich: bool,
        partial_cut: bool,
        h_enrich_per_node=np.zeros(4),
    ):
        super().__init__(node_coords, material, real)
        self.phi_n = phi_n
        self.phi_t = phi_t
        self.h_enrich = h_enrich
        self.t_enrich = t_enrich
        self.partial_cut = partial_cut
        self.h_enrich_per_node = h_enrich_per_node

    def cal_element_matrices2(self, eval_mass=False):
        n = N_DOFS + int(self.h_enrich) * H_DOFS + int(self.t_enrich) * TIP_DOFS
        Ke = np.zeros((n, n))
        x_e = self.node_coords
        rule, correction = qd.QUAD_RULES[3]
        for [xi, eta, w] in rule:
            _, dN_dxi = super().shape_functions(xi, eta)
            J = dN_dxi @ x_e
            detJ = np.linalg.det(J)
            dN_dxy = np.linalg.solve(J, dN_dxi)
            B = cal_B_2d(dN_dxy)
            Ke[0:N_DOFS, 0:N_DOFS] += (
                (B.T @ self.C @ B) * detJ * self.t * w * correction
            )
        Nc1 = None
        Nc2 = None
        if self.h_enrich or self.partial_cut:
            Nc1, Nc2 = self._cal_intersections()
        else:
            (rule, correction) = qd.QUAD_RULES[10]
            begin_tip_fn = N_FN
            begin_tip_dofs = N_DOFS
            for [xi, eta, w] in rule:
                _, dN_dxi = self.shape_functions(xi, eta)
                J = dN_dxi[:, :N_FN] @ x_e
                detJ = np.linalg.det(J)
                dN_dxy = np.linalg.solve(J, dN_dxi)
                B = cal_B_2d(dN_dxy[:, :begin_tip_fn])
                TIP_B = cal_B_2d(dN_dxy[:, begin_tip_fn:])
                w_eff = w * correction * detJ * self.t
                Ke[begin_tip_dofs:, begin_tip_dofs:] += TIP_B.T @ self.C @ TIP_B * w_eff
                res = B.T @ self.C @ TIP_B * w_eff
                Ke[0:begin_tip_dofs, begin_tip_dofs:] += res
                Ke[begin_tip_dofs:, 0:begin_tip_dofs] += res.T
        if self.partial_cut:
            # print("partial cut")
            assert Nc1 is not None and Nc2 is not None
            assert not self.h_enrich
            (rule, correction) = qd.QUAD_RULES[10]
            rule = rule.copy()
            rule[:, 0:2] = (1 + rule[:, 0:2]) / 2
            rule[:, 2] /= 4
            tip1 = np.linalg.solve(
                np.array([self.phi_t[:-1], self.phi_n[:-1], [1, 1, 1]]),
                np.array([0, 0, 1]),
            )
            # print("tip1", tip1)
            tip2 = np.linalg.solve(
                np.array([self.phi_t[[0, 2, 3]], self.phi_n[[0, 2, 3]], [1, 1, 1]]),
                np.array([0, 0, 1]),
            )
            self._integrate_partial_cut(
                Ke,
                tip1,
                Nc1,
                range(4),
                NAT_1,
                rule,
                correction,
            )
            self._integrate_partial_cut(
                Ke,
                tip2,
                Nc2,
                range(2, 6),
                NAT_2,
                rule,
                correction,
            )
        elif self.h_enrich:
            assert Nc1 is not None and Nc2 is not None
            self._integrate_sub_tri(Ke, Nc1, NAT_1)
            self._integrate_sub_tri(Ke, Nc2, NAT_2)

        if eval_mass:
            raise NotImplementedError
        return Ke

    def cal_element_matrices(self, eval_mass=False):
        n = N_DOFS + int(self.h_enrich) * H_DOFS + int(self.t_enrich) * TIP_DOFS
        Ke = np.zeros((n, n))
        Nc1 = None
        Nc2 = None
        if self.h_enrich or self.partial_cut:
            Nc1, Nc2 = self._cal_intersections()
        else:
            x_e = self.node_coords
            (rule, correction) = qd.QUAD_RULES[10]
            xi = rule[:, 0]
            eta = rule[:, 1]
            _, dN_dxi = self.shape_functions2(xi, eta)
            J = dN_dxi[:, :, :N_FN] @ x_e
            detJ = np.linalg.det(J)
            dN_dxy = np.linalg.solve(J, dN_dxi)
            B = cal_B_2d_vec(dN_dxy)
            w_eff = rule[:, 2] * correction * detJ
            Ke[:, :] = np.sum(
                (B.transpose(0, 2, 1) @ self.C @ B) * w_eff[:, None, None], axis=0
            )
        if self.partial_cut:
            Ke[:N_DOFS, :N_DOFS] = super().cal_element_matrices(eval_mass=False)
            assert Nc1 is not None and Nc2 is not None
            assert not self.h_enrich
            (rule, correction) = qd.QUAD_RULES[10]
            rule = rule.copy()
            rule[:, 0:2] = (1 + rule[:, 0:2]) / 2
            rule[:, 2] /= 4
            tip1 = np.linalg.solve(
                np.array([self.phi_t[:-1], self.phi_n[:-1], [1, 1, 1]]),
                np.array([0, 0, 1]),
            )
            tip2 = np.linalg.solve(
                np.array([self.phi_t[[0, 2, 3]], self.phi_n[[0, 2, 3]], [1, 1, 1]]),
                np.array([0, 0, 1]),
            )
            self._integrate_partial_cut2(
                Ke,
                tip1,
                Nc1,
                range(4),
                NAT_1,
                rule,
                correction,
            )
            self._integrate_partial_cut2(
                Ke,
                tip2,
                Nc2,
                range(2, 6),
                NAT_2,
                rule,
                correction,
            )
        elif self.h_enrich:
            assert Nc1 is not None and Nc2 is not None
            self._integrate_sub_tri2(Ke, Nc1, NAT_1)
            self._integrate_sub_tri2(Ke, Nc2, NAT_2)

        if eval_mass:
            raise NotImplementedError
        return Ke

    def _cal_intersections(self):
        num = np.empty_like(self.phi_n)
        num[:-1] = self.phi_n[1:]
        num[-1] = self.phi_n[0]
        denom = num - self.phi_n
        unsolvable = np.isclose(denom, 0)
        denom += unsolvable
        N1 = np.clip(num / denom * ~unsolvable * ~np.isclose(self.phi_n, 0), 0, 1)
        N1 += unsolvable | np.isclose(self.phi_n, 0)
        num_diag = self.phi_n[0]
        denom_diag = num_diag - self.phi_n[2]
        unsolvable_diag = np.isclose(denom_diag, 0)
        N1_diag = np.clip(
            num_diag / denom_diag * ~unsolvable_diag * ~np.isclose(self.phi_n[2], 0),
            0,
            1,
        )
        N1_diag += unsolvable_diag | np.isclose(self.phi_n[2], 0)
        Nc1 = np.array(
            [
                [N1[0], 0, 1 - N1_diag],
                [1 - N1[0], N1[1], 0],
                [0, 1 - N1[1], N1_diag],
            ]
        )
        Nc2 = np.array(
            [
                [1 - N1_diag, 0, 1 - N1[3]],
                [N1_diag, N1[2], 0],
                [0, 1 - N1[2], N1[3]],
            ]
        )
        return Nc1, Nc2

    @staticmethod
    def _cut_embedding_iter(Nc, range=range(4)):
        for i in range:
            if i != 3:
                Ni = np.eye(3)
                Ni[:, (i + 1) % 3] = Nc[:, i]
                Ni[:, (i + 2) % 3] = Nc[:, (i + 2) % 3]
            else:
                Ni = Nc.copy()
            detJi = np.linalg.det(Ni)
            if not np.isclose(detJi, 0.0):
                yield Ni, detJi

    @staticmethod
    def _partial_cut_embedding_iter(Nc, tip, range):
        Ni_template = np.zeros((3, 3))
        Ni_template[:, 0] = tip
        for i in range:
            Ni = Ni_template.copy()
            Ni[int((i % 5 + 1) / 2), 1 + i % 2] = 1
            Ni[:, 2 - i % 2] = Nc[:, int(i / 2)]
            detJi = np.linalg.det(Ni)
            if not np.isclose(detJi, 0):
                yield Ni, detJi

    def _integrate_sub_tri(self, Ke, Nc, nat_x_e):
        begin_h_fn = N_FN
        begin_tip_fn = N_FN + H_FN
        begin_h_dofs = N_DOFS
        begin_tip_dofs = N_DOFS + H_DOFS
        if self.t_enrich:
            rule, correction = qd.TRI_RULES[10]
        else:
            rule, correction = qd.TRI_RULES[3]
        x_e = self.node_coords
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
            for [xi, eta, w] in rule:
                n = np.array([1 - xi - eta, xi, eta])
                xi_sub, eta_sub = nat_x_e.T @ Ni @ n
                _, dN_dxi_sub = self.shape_functions(xi_sub, eta_sub)
                J = dN_dxi_sub[:, 0:begin_h_fn] @ x_e
                detJ = np.linalg.det(J)
                dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
                B = cal_B_2d(dN_dxy_sub[:, 0:begin_h_fn])
                HB = cal_B_2d(dN_dxy_sub[:, begin_h_fn:begin_tip_fn])
                w_eff = w * correction * detJ * detJi * self.t * 4
                Ke[begin_h_dofs:begin_tip_dofs, begin_h_dofs:begin_tip_dofs] += (
                    HB.T @ self.C @ HB
                ) * w_eff
                res = (B.T @ self.C @ HB) * w_eff
                Ke[0:begin_h_dofs, begin_h_dofs:begin_tip_dofs] += res
                Ke[begin_h_dofs:begin_tip_dofs, 0:begin_h_dofs] += res.T
                if self.t_enrich:
                    TIP_B = cal_B_2d(dN_dxy_sub[:, begin_tip_fn:])
                    Ke[begin_tip_dofs:, begin_tip_dofs:] += (
                        TIP_B.T @ self.C @ TIP_B * w_eff
                    )
                    res = B.T @ self.C @ TIP_B * w_eff
                    Ke[0:begin_h_dofs, begin_tip_dofs:] += res
                    Ke[begin_tip_dofs:, 0:begin_h_dofs] += res.T
                    res = HB.T @ self.C @ TIP_B * w_eff
                    Ke[begin_h_dofs:begin_tip_dofs, begin_tip_dofs:] += res
                    Ke[begin_tip_dofs:, begin_h_dofs:begin_tip_dofs] += res.T

    def _integrate_sub_tri2(self, Ke, Nc, nat_x_e):
        if self.t_enrich:
            rule, correction = qd.TRI_RULES[10]
        else:
            rule, correction = qd.TRI_RULES[3]
        x_e = self.node_coords
        for Ni, detJi in self._cut_embedding_iter(Nc):
            xi = rule[:, 0]
            eta = rule[:, 1]
            w = rule[:, 2]
            n = np.array([1 - xi - eta, xi, eta])
            xi_sub, eta_sub = nat_x_e.T @ Ni @ n
            _, dN_dxi_sub = self.shape_functions2(xi_sub, eta_sub)
            J = dN_dxi_sub[:, :, 0:N_FN] @ x_e
            detJ = np.linalg.det(J)
            dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
            B = cal_B_2d_vec(dN_dxy_sub)
            w_eff = w * correction * detJ * detJi * 4
            Ke += np.sum(
                B.transpose(0, 2, 1) @ self.C @ B * w_eff[:, None, None], axis=0
            )

    def _integrate_partial_cut(self, Ke, tip, Nc, range, nat_x_e, rule, correction):
        x_e = self.node_coords
        Ni_template = np.zeros((3, 3))
        Ni_template[:, 0] = tip

        for i in range:
            Ni = Ni_template.copy()
            Ni[int((i % 5 + 1) / 2), 1 + i % 2] = 1
            Ni[:, 2 - i % 2] = Nc[:, int(i / 2)]
            detJi = np.linalg.det(Ni)
            if np.isclose(detJi, 0):
                continue
            if detJi < 0:
                print("DetJi smaller than 0", detJi)
            # print("Ni", Ni)
            nat_sub_x_e = nat_x_e.T @ Ni
            x_e_i = np.array(
                [
                    super().shape_functions(xi, eta)[0] @ x_e
                    for [xi, eta] in nat_sub_x_e.T
                ]
            )
            duffy = DuffyDistance(x_e_i)
            for [u, v, w] in rule:
                [xi_d, eta_d, w_d] = duffy.transform(u, v, beta=1)
                n = np.array([1 - xi_d - eta_d, xi_d, eta_d])
                xi_sub, eta_sub = nat_sub_x_e @ n
                _, dN_dxi_sub = self.shape_functions(xi_sub, eta_sub)
                J = dN_dxi_sub[:, 0:N_FN] @ x_e
                detJ = np.linalg.det(J)
                w_eff = w * correction * w_d * self.t * detJi * detJ * 4
                dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
                B = cal_B_2d(dN_dxy_sub[:, :N_FN])
                TIP_B = cal_B_2d(dN_dxy_sub[:, N_FN:])
                begin_tip = N_DOFS
                Ke[begin_tip:, begin_tip:] += (TIP_B.T @ self.C @ TIP_B) * w_eff

                [xi_d, eta_d, w_d] = duffy.transform(u, v, beta=2)
                n = np.array([1 - xi_d - eta_d, xi_d, eta_d])
                xi_sub, eta_sub = nat_x_e.T @ Ni @ n
                _, dN_dxi_sub = self.shape_functions(xi_sub, eta_sub)
                J = dN_dxi_sub[:, 0:N_FN] @ x_e
                detJ = np.linalg.det(J)
                w_eff = w * correction * w_d * self.t * detJi * detJ * 4
                dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
                B = cal_B_2d(dN_dxy_sub[:, :N_FN])
                TIP_B = cal_B_2d(dN_dxy_sub[:, N_FN:])
                begin_tip = N_DOFS
                Ke[0:begin_tip, begin_tip:] += (B.T @ self.C @ TIP_B) * w_eff
                Ke[begin_tip:, 0:begin_tip] += (TIP_B.T @ self.C @ B) * w_eff
        # print("tot det Ji", tot_detJi)

    def _integrate_partial_cut2(self, Ke, tip, Nc, range, nat_x_e, rule, correction):
        x_e = self.node_coords
        Ni_template = np.zeros((3, 3))
        Ni_template[:, 0] = tip

        for Ni, detJi in self._partial_cut_embedding_iter(Nc, tip, range):
            if detJi < 0:
                print("DetJi smaller than 0", detJi)
            nat_sub_x_e = nat_x_e.T @ Ni
            x_e_i = super().shape_functions2(nat_sub_x_e[0], nat_sub_x_e[1])[0] @ x_e
            duffy = DuffyDistance(x_e_i)
            u, v = rule[:, 0], rule[:, 1]
            xi_d, eta_d, w_d = duffy.transform(u, v, beta=1)
            n = np.array([1 - xi_d - eta_d, xi_d, eta_d])
            xi_sub, eta_sub = nat_sub_x_e @ n
            _, dN_dxi_sub = self.shape_functions2(xi_sub, eta_sub)
            J = dN_dxi_sub[:, :, 0:N_FN] @ x_e
            detJ = np.linalg.det(J)
            dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
            TIP_B = cal_B_2d_vec(dN_dxy_sub[:, :, N_FN:])
            w_eff = rule[:, 2] * correction * w_d * detJi * detJ * 4
            begin_tip = N_DOFS
            Ke[begin_tip:, begin_tip:] += np.sum(
                (TIP_B.transpose(0, 2, 1) @ self.C @ TIP_B) * w_eff[:, None, None],
                axis=0,
            )

            xi_d, eta_d, w_d = duffy.transform(u, v, beta=2)
            n = np.array([1 - xi_d - eta_d, xi_d, eta_d])
            xi_sub, eta_sub = nat_sub_x_e @ n
            _, dN_dxi_sub = self.shape_functions2(xi_sub, eta_sub)
            J = dN_dxi_sub[:, :, 0:N_FN] @ x_e
            detJ = np.linalg.det(J)
            dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
            B = cal_B_2d_vec(dN_dxy_sub[:, :, :N_FN])
            TIP_B = cal_B_2d_vec(dN_dxy_sub[:, :, N_FN:])
            w_eff = rule[:, 2] * correction * w_d * detJi * detJ * 4
            begin_tip = N_DOFS
            res = np.sum(
                B.transpose(0, 2, 1) @ self.C @ TIP_B * w_eff[:, None, None], axis=0
            )
            Ke[0:begin_tip, begin_tip:] += res
            Ke[begin_tip:, 0:begin_tip] += res.T

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
            r = np.sqrt(phi_n**2 + phi_t**2)
            r = np.maximum(r, 1e-14)  # avoid divide by zero
            sqrt_r = np.sqrt(r)
            sqrt_r_i = (self.phi_n**2 + self.phi_t**2) ** (1 / 4)
            theta = np.atan2(phi_n, phi_t)
            theta_i = np.atan2(self.phi_n, self.phi_t)
            dphi_n_dxi = np.sum(self.phi_n * dN_dxi[:, :N_FN], axis=1)
            dphi_t_dxi = np.sum(self.phi_t * dN_dxi[:, :N_FN], axis=1)
            # sin(theta) = phi_n / r, cos(theta) = phi_t / r
            dr_dxi = (
                1 / r * (phi_n * dphi_n_dxi + phi_t * dphi_t_dxi)
            )  # = np.sin(theta) * dphi_n_dxi - np.cos(theta) * dphi_t_dxi
            dtheta_dxi = (dphi_n_dxi * phi_t - phi_n * dphi_t_dxi) / (
                phi_t**2 + phi_n**2
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
                # print(bf_i.T)
                print((1 - self.h_enrich_per_node)[:, None])
                bf_shifted = (
                    bf
                    - (1 - self.h_enrich_per_node[:, None]) * bf_i.T
                    - self.h_enrich_per_node[:, None]
                    * np.sign(phi_n)
                    * np.sign(self.phi_n).reshape((-1, 1))
                    * bf_i.T
                )
                print("bf_shifted")
                print(bf_shifted)
            else:
                bf_shifted = bf - bf_i.T
            begin_tip = N_FN + int(self.h_enrich) * H_FN
            end_tip = begin_tip + TIP_FN
            N[begin_tip:end_tip] = (bf_shifted * N[:N_FN].reshape((-1, 1))).flatten()
            term1 = dbf_dxi * N[:N_FN, None, None]  # (3, 2, 4)
            term2 = bf_shifted[:, :, None] * dN_dxi[:, :N_FN].T[:, None, :]  # (3, 4, 2)
            dN_dxi[0, begin_tip:end_tip] = (term1[:, 0, :] + term2[:, :, 0]).flatten()
            dN_dxi[1, begin_tip:end_tip] = (term1[:, 1, :] + term2[:, :, 1]).flatten()
        return N, dN_dxi

    def shape_functions2(self, xi, eta):
        n_points = xi.shape[0]
        N = np.empty(
            (n_points, N_FN + int(self.h_enrich) * H_FN + int(self.t_enrich) * TIP_FN)
        )
        dN_dxi = np.empty(
            (
                n_points,
                DOFS,
                N_FN + int(self.h_enrich) * H_FN + int(self.t_enrich) * TIP_FN,
            )
        )
        (N[:, :N_FN], dN_dxi[:, :, :N_FN]) = super().shape_functions2(xi, eta)
        phi_n = np.sum(self.phi_n * N[:, :N_FN], axis=1)
        phi_t = np.sum(self.phi_t * N[:, :N_FN], axis=1)
        # print(phi_n)
        if self.h_enrich:
            h_shifted = (np.sign(phi_n)[:, None] - np.sign(self.phi_n)) / 2
            # print(h_shifted)
            begin_h, end_h = N_FN, N_FN + H_FN
            N[:, begin_h:end_h] = h_shifted * N[:, :N_FN]
            dN_dxi[:, :, begin_h:end_h] = h_shifted[:, None, :] * dN_dxi[:, :, :N_FN]
        if self.t_enrich:
            r = np.sqrt(phi_n**2 + phi_t**2)
            r = np.maximum(r, 1e-14)  # avoid divide by zero
            sqrt_r = np.sqrt(r)
            sqrt_r_i = (self.phi_n**2 + self.phi_t**2) ** (1 / 4)
            theta = np.atan2(phi_n, phi_t)
            theta_i = np.atan2(self.phi_n, self.phi_t)
            dphi_n_dxi = np.sum(self.phi_n * dN_dxi[:, :, :N_FN], axis=2)
            dphi_t_dxi = np.sum(self.phi_t * dN_dxi[:, :, :N_FN], axis=2)
            # sin(theta) = phi_n / r, cos(theta) = phi_t / r
            dr_dxi = (
                1
                / r[:, None]
                * (phi_n[:, None] * dphi_n_dxi + phi_t[:, None] * dphi_t_dxi)
            )  # = np.sin(theta) * dphi_n_dxi - np.cos(theta) * dphi_t_dxi
            dtheta_dxi = (dphi_n_dxi * phi_t[:, None] - phi_n[:, None] * dphi_t_dxi) / (
                phi_t**2 + phi_n**2
            )[:, None]  # = (dphi_n_dxi * np.cos(theta) + np.sin(theta) dphi_t_dxi) / r
            bf = branch_functions(sqrt_r, theta).T
            bf_i = branch_functions(sqrt_r_i, theta_i).T
            # print(sqrt_r)
            # print(dr_dxi.reshape((-1, 2, 1)))
            # print((bf / sqrt_r[:, None]))

            dbf_dxi = 1 / (2 * sqrt_r[:, None, None]) * dr_dxi.reshape((-1, 2, 1)) * (
                bf / sqrt_r[:, None]
            )[:, None, :] + sqrt_r[:, None, None] * np.array(
                [
                    -np.sin(theta[:, None] / 2) * dtheta_dxi / 2,
                    np.cos(theta[:, None] / 2) * dtheta_dxi / 2,
                    np.cos(theta[:, None] / 2) * dtheta_dxi / 2 * np.sin(theta[:, None])
                    + np.sin(theta[:, None] / 2) * np.cos(theta[:, None]) * dtheta_dxi,
                    -np.sin(theta[:, None] / 2)
                    * dtheta_dxi
                    / 2
                    * np.sin(theta[:, None])
                    + np.cos(theta[:, None] / 2) * np.cos(theta[:, None]) * dtheta_dxi,
                ]
            ).transpose(1, 2, 0)
            if self.h_enrich:
                print(bf_i)
                print(self.h_enrich_per_node[None, :, None])
                bf_shifted = (
                    bf[:, None, :]
                    - (1 - self.h_enrich_per_node)[None, :, None] * bf_i[None, :, :]
                    - self.h_enrich_per_node[None, :, None]
                    * np.sign(phi_n[:, None, None])
                    * np.sign(self.phi_n[None, :, None])
                    * bf_i[None, :, :]
                )
                print("bf_shifted")
                print(bf_shifted)
            else:
                bf_shifted = bf[:, None, :] - bf_i[None, :, :]
            begin_tip = N_FN + int(self.h_enrich) * H_FN
            end_tip = begin_tip + TIP_FN

            N[:, begin_tip:end_tip] = (bf_shifted * N[:, :N_FN, None]).reshape(
                -1, TIP_FN
            )
            term1 = dbf_dxi[:, None, :, :] * N[:, :N_FN, None, None]  # (n, 4, 2, 4)
            term2 = (
                bf_shifted[:, :, :, None]
                * dN_dxi[:, :, :N_FN].transpose(0, 2, 1)[:, :, None, :]
            )  # (n, 4, 4, 2)
            dN_dxi[:, 0, begin_tip:end_tip] = (
                term1[:, :, 0, :] + term2[:, :, :, 0]
            ).reshape(-1, TIP_FN)
            dN_dxi[:, 1, begin_tip:end_tip] = (
                term1[:, :, 1, :] + term2[:, :, :, 1]
            ).reshape(-1, TIP_FN)
        return N, dN_dxi

    def cal_stresses(self, xi, eta, Ue):
        Ue = np.asarray(Ue, dtype=float).ravel()
        _, dN_dxi = self.shape_functions2(xi, eta)
        J = dN_dxi @ self.node_coords
        dN_dxy = np.linalg.solve(J, dN_dxi)
        B = cal_B_2d_vec(dN_dxy)
        eps = B @ Ue
        sig = self.C @ eps
        return sig
