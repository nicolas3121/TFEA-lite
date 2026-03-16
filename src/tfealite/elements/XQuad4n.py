from .Quad4n import Quad4n
import numpy as np
from typing import Final
from .utils import branch_functions, cal_B_2d_vec
from ..core import quadratures as qd
from ..core.quadratures import DuffyDistance

NODES: Final = 4
DOFS: Final = 2
BRANCH_FN: Final = 4  # branch functions
N_FN: Final = NODES
H_FN: Final = NODES
LH_FN: Final = 2 * NODES
TIP_FN: Final = NODES * BRANCH_FN
N_DOFS: Final = DOFS * N_FN
H_DOFS: Final = DOFS * H_FN
LH_DOFS: Final = DOFS * LH_FN
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
        in_range=None,
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
        in_range=np.ones(4, dtype=bool),
    ):
        super().__init__(node_coords, material, real)
        self.phi_n = phi_n
        self.phi_t = phi_t
        self.h_enrich = h_enrich
        self.t_enrich = t_enrich
        self.partial_cut = partial_cut
        self.in_range = in_range

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
            _, dN_dxi = self.shape_functions(xi, eta)
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
            self._cal_tip_nat_coords()
            xi_tip, eta_tip = self._cal_tip_nat_coords()
            tri1_coords = np.array([[-1, 1, 1], [-1, -1, 1], [1, 1, 1]])
            tip1 = np.linalg.solve(tri1_coords, [xi_tip, eta_tip, 1.0])

            tri2_coords = np.array([[-1, 1, -1], [-1, 1, 1], [1, 1, 1]])
            tip2 = np.linalg.solve(tri2_coords, [xi_tip, eta_tip, 1.0])

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

    def _cal_intersections(self):
        num = np.empty_like(self.phi_n)
        num[:-1] = self.phi_n[1:]
        num[-1] = self.phi_n[0]
        denom = num - self.phi_n
        unsolvable = np.isclose(denom, 0)
        on_crack = np.isclose(self.phi_n, 0)
        N1 = np.clip(
            np.divide(
                num,
                denom,
                out=np.ones_like(num, dtype=float),
                where=~unsolvable & ~on_crack,
            ),
            0,
            1,
        )
        [phi1, phi2, phi3, phi4] = self.phi_n
        A = phi1 - phi2 + phi3 - phi4
        B = -2.0 * phi1 + phi2 + phi4
        C = phi1
        num_diag = self.phi_n[0]
        denom_diag = num_diag - self.phi_n[2]
        unsolvable_diag = np.isclose(denom_diag, 0)
        s_linear = np.divide(
            num_diag,
            denom_diag,
            out=np.ones_like(phi1, dtype=float),
            where=~unsolvable_diag,
        )
        discriminant = B**2 - 4 * A * C
        use_linear = np.isclose(A, 0) | (discriminant < 0)
        safe_A = np.where(use_linear, 1.0, A)
        sqrt_disc = np.sqrt(np.maximum(discriminant, 0))
        root1 = (-B + sqrt_disc) / (2.0 * safe_A)
        root2 = (-B - sqrt_disc) / (2.0 * safe_A)

        valid_root1 = (root1 >= 0.0) & (root1 <= 1.0)
        valid_root2 = (root2 >= 0.0) & (root2 <= 1.0)

        s_quad = np.where(valid_root1, root1, np.where(valid_root2, root2, s_linear))

        N1_diag = np.where(use_linear, s_linear, s_quad)

        N1_diag = np.where(unsolvable_diag | np.isclose(phi3, 0), 1.0, N1_diag)
        N1_diag = np.clip(N1_diag, 0, 1)

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

    def _cal_tip_nat_coords(self):
        xi, eta = 0.0, 0.0
        for _ in range(50):
            N, dN_dxi = super().shape_functions(xi, eta)
            N = N[0]
            dN_dxi = dN_dxi[0]
            val_t = np.dot(N, self.phi_t)
            val_n = np.dot(N, self.phi_n)
            if np.isclose(val_t, 0) and np.isclose(val_n, 0):
                return xi, eta
            J = np.array(
                [
                    [
                        np.dot(dN_dxi[0, :], self.phi_t),
                        np.dot(dN_dxi[1, :], self.phi_t),
                    ],
                    [
                        np.dot(dN_dxi[0, :], self.phi_n),
                        np.dot(dN_dxi[1, :], self.phi_n),
                    ],
                ]
            )
            step = np.linalg.solve(J, [-val_t, -val_n])
            xi += step[0]
            eta += step[1]
        raise ValueError("newton iterations didn't converge")

    def _cal_nat_coords(self, points):
        points = np.atleast_2d(points)
        xi = np.zeros(points.shape[0])
        eta = np.zeros_like(xi)
        x_e = self.node_coords
        for _ in range(50):
            N, dN_dxi = super().shape_functions(xi, eta)
            dx = N @ x_e - points
            if np.all(np.isclose(np.sum(dx**2, axis=1), 0.0)):
                return xi, eta
            J = dN_dxi @ x_e

            step = np.linalg.solve(J.T, dx)
            xi += step[:, 0]
            eta += step[:, 1]
        raise ValueError("newton iterations didn't converge")

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
        w_tot = 0
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
            _, dN_dxi_sub = self.shape_functions(xi_sub, eta_sub)
            J = dN_dxi_sub[:, :, 0:N_FN] @ x_e
            detJ = np.linalg.det(J)
            dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
            B = cal_B_2d_vec(dN_dxy_sub)
            w_eff = w * correction * detJ * detJi * 4
            w_tot += w_eff
            Ke += np.sum(
                B.transpose(0, 2, 1) @ self.C @ B * w_eff[:, None, None], axis=0
            )

    def _integrate_partial_cut(self, Ke, tip, Nc, range, nat_x_e, rule, correction):
        x_e = self.node_coords
        Ni_template = np.zeros((3, 3))
        Ni_template[:, 0] = tip

        for Ni, detJi in self._partial_cut_embedding_iter(Nc, tip, range):
            if detJi < 0:
                print("DetJi smaller than 0", detJi)
            nat_sub_x_e = nat_x_e.T @ Ni
            x_e_i = super().shape_functions(nat_sub_x_e[0], nat_sub_x_e[1])[0] @ x_e
            duffy = DuffyDistance(x_e_i)
            u, v = rule[:, 0], rule[:, 1]
            xi_d, eta_d, w_d = duffy.transform(u, v, beta=1)
            n = np.array([1 - xi_d - eta_d, xi_d, eta_d])
            xi_sub, eta_sub = nat_sub_x_e @ n
            _, dN_dxi_sub = self.shape_functions(xi_sub, eta_sub)
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
            _, dN_dxi_sub = self.shape_functions(xi_sub, eta_sub)
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

    def _cubic_shape_functions(self, xi, eta):
        Q0_xi = (xi - 1) ** 2 * (xi + 2) / 4
        Q1_xi = (2 - xi) * (xi + 1) ** 2 / 4
        Q0_eta = (eta - 1) ** 2 * (eta + 2) / 4
        Q1_eta = (2 - eta) * (eta + 1) ** 2 / 4
        Q = np.array(
            [
                Q0_xi * Q0_eta,
                Q1_xi * Q0_eta,
                Q1_xi * Q1_eta,
                Q0_xi * Q1_eta,
            ]
        ).T
        dQ0_xi = 3 / 4 * (xi**2 - 1)
        dQ1_xi = 3 / 4 * (1 - xi**2)
        dQ0_eta = 3 / 4 * (eta**2 - 1)
        dQ1_eta = 3 / 4 * (1 - eta**2)
        row1 = [dQ0_xi * Q0_eta, dQ1_xi * Q0_eta, dQ1_xi * Q1_eta, dQ0_xi * Q1_eta]
        row2 = [Q0_xi * dQ0_eta, Q1_xi * dQ0_eta, Q1_xi * dQ1_eta, Q0_xi * dQ1_eta]
        dQ_dxi = np.stack([row1, row2]).transpose(2, 0, 1)
        return Q, dQ_dxi

    def nearest_point_on_crack(self, coords):
        Nc1, Nc2 = self._cal_intersections()

        def project_on_crack(Nc, x_e, coords):
            cols = np.where(np.sum(np.isclose(Nc, 0.0, 1e-12), axis=0) == 1)[0]
            if len(cols) == 2:
                intersections = x_e.T @ Nc[:, cols]
                v = intersections[:, 1] - intersections[:, 0]
                w = coords - intersections[:, 0]
                t = np.clip(np.dot(v, w) / np.linalg.norm(v), 0.0, 1.0)
                p = intersections[:, 0] + t * v
                d = np.linalg.norm(p - coords)
                return p, d
            return None, np.inf

        touching = np.where(np.isclose(self.phi_n, 0.0, 1e-12))[0]
        p1, d1 = project_on_crack(Nc1, NAT_1, coords)
        p2, d2 = project_on_crack(Nc2, NAT_2, coords)
        if len(touching) == 1:
            p = self.node_coords[touching[0], :]
            np.linalg.norm(p - coords)
        elif len(touching) == 2:
            nodes = self.node_coords[touching, :]
            v = nodes[1, :] - nodes[0, :]
            w = coords - nodes[0, :]
            t = np.clip(np.dot(v, w) / np.linalg.norm(v), 0.0, 1.0)
            p = nodes[0, :] + t * v
            np.linalg.norm(p - coords)
        elif p1 is not None or p2 is not None:
            if d1 < d2:
                p = p1
            else:
                p = p2
        else:
            return None
        return p

    def contains_points(self, points):
        points = np.atleast_2d(points)
        x_e_1 = self.node_coords
        x_e_2 = np.empty_like(x_e_1)
        x_e_2[:-1, :] = x_e_1[1:, :]
        x_e_2[-1, :] = x_e_1[0, :]

        edges = x_e_2 - x_e_1
        vec = points[:, None, :] - x_e_1[None, :, :]
        cross = edges[None, :, 0] * vec[:, :, 1] - edges[None, :, 1] * vec[:, :, 0]

        is_inside = np.all(cross >= -1e-12, axis=1)
        return is_inside

    def jump_shape_functions(self, xi, eta, tip_coords):
        n_points = xi.shape[0]
        N, _ = super().shape_functions(xi, eta)
        Q, _ = self._cubic_shape_functions(xi, eta)
        N_jump = np.empty(
            (n_points, N_FN + int(self.h_enrich) * H_FN + int(self.t_enrich) * TIP_FN)
        )
        N_jump[:, :N_FN] = 0.0
        r = np.linalg.norm(N @ self.node_coords - tip_coords)
        if self.h_enrich:
            begin_h, end_h = N_FN, N_FN + H_FN
            N_jump[:, begin_h:end_h] = N[:, :N_FN]
        if self.t_enrich:
            sqrt_r = np.sqrt(r)
            begin_tip = N_FN + int(self.h_enrich) * H_FN
            end_tip = begin_tip + TIP_FN
            N_jump[:, begin_tip:end_tip] = 0.0
            N_jump[:, begin_tip::4] = 2 * sqrt_r[:, None] * Q
        return N_jump, r

    def shape_functions(self, xi, eta):
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
        (N[:, :N_FN], dN_dxi[:, :, :N_FN]) = super().shape_functions(xi, eta)
        Q, dQ_dxi = self._cubic_shape_functions(xi, eta)
        phi_n = np.sum(self.phi_n * N[:, :N_FN], axis=1)
        phi_t = np.sum(self.phi_t * N[:, :N_FN], axis=1)
        if self.h_enrich:
            h_shifted = (np.sign(phi_n)[:, None] - np.sign(self.phi_n)) / 2
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

            dbf_dxi = 1 / (2 * sqrt_r[:, None, None]) * dr_dxi.reshape((-1, 2, 1)) * (
                bf / sqrt_r[:, None]
            )[:, None, :] + sqrt_r[:, None, None] * np.array(
                [
                    np.cos(theta[:, None] / 2) * dtheta_dxi / 2,
                    -np.sin(theta[:, None] / 2) * dtheta_dxi / 2,
                    np.cos(theta[:, None] / 2) * dtheta_dxi / 2 * np.sin(theta[:, None])
                    + np.sin(theta[:, None] / 2) * np.cos(theta[:, None]) * dtheta_dxi,
                    -np.sin(theta[:, None] / 2)
                    * dtheta_dxi
                    / 2
                    * np.sin(theta[:, None])
                    + np.cos(theta[:, None] / 2) * np.cos(theta[:, None]) * dtheta_dxi,
                ]
            ).transpose(1, 2, 0)
            shifter = bf_i[None, :, :]
            interpolant = np.sum(shifter * N[:, :N_FN, None], axis=1)
            begin_tip = N_FN + int(self.h_enrich) * H_FN
            end_tip = begin_tip + TIP_FN
            bf_shifted = bf - interpolant
            ramp = np.sum(N[:, np.where(self.in_range)[0]], axis=1)
            dramp_dxi = np.sum(dN_dxi[:, :, np.where(self.in_range)[0]], axis=2)

            N[:, begin_tip:end_tip] = (
                bf_shifted[:, None, :] * ramp[:, None, None] * Q[:, :, None]
            ).reshape(-1, TIP_FN)

            term1 = (
                (
                    dbf_dxi
                    - np.sum(shifter[:, None, :, :] * dN_dxi[:, :, :N_FN, None], axis=2)
                )[:, None, :, :]  # (n, 1, 2, 4)
                * ramp[:, None, None, None]  # (n)
                * Q[:, :, None, None]  # (n, 4, 1, 1)
            )  # (n, 4, 2, 4)
            term2 = (
                bf_shifted[:, None, None, :]
                * dramp_dxi[:, :, None, None]
                * Q[:, None, :, None]
            )  # (n, 2, 4, 4)

            term3 = (
                bf_shifted[:, None, None, :]  # (n, 1, 1, 4)
                * ramp[:, None, None, None]
                * dQ_dxi[:, :, :, None]  # (n, 2, 4, 1)
            )  # (n, 2, 4, 4)
            dN_dxi[:, 0, begin_tip:end_tip] = (
                term1[:, :, 0, :] + term2[:, 0, :, :] + term3[:, 0, :, :]
            ).reshape(-1, TIP_FN)
            dN_dxi[:, 1, begin_tip:end_tip] = (
                term1[:, :, 1, :] + term2[:, 1, :, :] + term3[:, 1, :, :]
            ).reshape(-1, TIP_FN)
        return N, dN_dxi

    def cal_stresses(self, xi, eta, Ue):
        Ue = np.asarray(Ue, dtype=float).ravel()
        _, dN_dxi = self.shape_functions(xi, eta)
        J = dN_dxi[:, :, :N_FN] @ self.node_coords
        dN_dxy = np.linalg.solve(J, dN_dxi)
        B = cal_B_2d_vec(dN_dxy)
        eps = B @ Ue
        sig = self.C @ eps
        return sig

    def stresses_at_nodes(self, Ue):
        xi = np.array([-1.0, 1.0, 1.0, -1.0])
        eta = np.array([-1.0, -1.0, 1.0, 1.0])
        return self.cal_stresses(xi, eta, Ue)
