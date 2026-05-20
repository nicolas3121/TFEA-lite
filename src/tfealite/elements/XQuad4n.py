from typing import Final

import numpy as np

from ..core import quadratures as qd
from ..core.quadratures import DuffyDistance
from .Quad4n import Quad4n
from .utils import (
    cal_B_2d_vec,
    cal_N_2d_vec,
    enriched_shape_functions,
    cut_embedding_tri_iter,
    partial_cut_embedding_tri_iter,
    jump_shape_functions,
)


class XQuad4n(Quad4n):
    NODES: Final = 4
    DOFS: Final = 2
    BRANCH_FN: Final = 4
    N_FN: Final = NODES
    H_FN: Final = NODES
    LH_FN: Final = 2 * NODES
    TIP_FN: Final = NODES * BRANCH_FN
    N_DOFS: Final = DOFS * N_FN
    H_DOFS: Final = DOFS * H_FN
    TIP_DOFS: Final = DOFS * TIP_FN

    NAT_13_1: Final = np.array([[-1, -1], [1, -1], [1, 1]])
    NAT_13_2: Final = np.array([[-1, -1], [1, 1], [-1, 1]])
    NAT_24_1: Final = np.array([[1, -1], [1, 1], [-1, 1]])
    NAT_24_2: Final = np.array([[1, -1], [-1, 1], [-1, -1]])

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
        n = (
            self.N_DOFS
            + int(self.h_enrich) * self.H_DOFS
            + int(self.t_enrich) * self.TIP_DOFS
        )
        Ke = np.zeros((n, n))
        Me = np.zeros((n, n))
        Nc1 = None
        Nc2 = None
        if self.h_enrich or self.partial_cut:
            Nc1, Nc2 = self._cal_intersections()
        else:
            x_e = self.node_coords
            (rule, correction) = qd.QUAD_RULES[20]
            nat_coords = rule[:, :2].T
            N, dN_dxi = self.shape_functions(nat_coords)
            J = dN_dxi[:, :, : self.N_FN] @ x_e
            detJ = np.linalg.det(J)
            if np.any(np.isnan(detJ)):
                print("nan occured at first one")
                print("nodes", self.node_coords)
                print("phi_n", self.phi_n)
                print("phi_t", self.phi_t)
            dN_dxy = np.linalg.solve(J, dN_dxi)
            B = cal_B_2d_vec(dN_dxy)
            w_eff = rule[:, 2] * correction * detJ
            Ke[:, :] = np.sum(
                (B.transpose(0, 2, 1) @ self.C @ B) * w_eff[:, None, None], axis=0
            )
            if eval_mass:
                N_2d = cal_N_2d_vec(N)
                Me[:, :] = np.sum(
                    (N_2d.transpose(0, 2, 1) @ N_2d) * w_eff[:, None, None],
                    axis=0,
                )
        if self.partial_cut:
            Ke_temp, _ = super().cal_element_matrices(eval_mass=False)
            Ke[: self.N_DOFS, : self.N_DOFS] = Ke_temp
            assert Nc1 is not None and Nc2 is not None
            assert not self.h_enrich
            (rule, correction) = qd.QUAD_RULES[20]
            rule = rule.copy()
            rule[:, 0:2] = (1 + rule[:, 0:2]) / 2
            rule[:, 2] /= 4
            xi_tip, eta_tip = self._cal_tip_nat_coords()
            # tri1_coords = np.array([[-1, 1, 1], [-1, -1, 1], [1, 1, 1]])
            tri1_coords = np.vstack([self.NAT_1.T, np.ones(3)])
            tip1 = np.linalg.solve(tri1_coords, [xi_tip, eta_tip, 1.0])

            # tri2_coords = np.array([[-1, 1, -1], [-1, 1, 1], [1, 1, 1]])
            tri2_coords = np.vstack([self.NAT_2.T, np.ones(3)])
            tip2 = np.linalg.solve(tri2_coords, [xi_tip, eta_tip, 1.0])

            self._integrate_partial_cut(
                Ke,
                Me,
                tip1,
                Nc1,
                range(4),
                self.NAT_1,
                rule,
                correction,
                eval_mass,
            )
            self._integrate_partial_cut(
                Ke,
                Me,
                tip2,
                Nc2,
                range(2, 6),
                self.NAT_2,
                rule,
                correction,
                eval_mass,
            )
        elif self.h_enrich:
            assert Nc1 is not None and Nc2 is not None
            self._integrate_sub_tri(Ke, Me, Nc1, self.NAT_1, eval_mass)
            self._integrate_sub_tri(Ke, Me, Nc2, self.NAT_2, eval_mass)

        if eval_mass:
            return Ke, Me
        return Ke, None

    def _cal_intersections(self):
        ZERO_TOL = 1e-10
        ROOT_TOL = 1e-10
        DENOM_TOL = 1e-12
        num = np.empty_like(self.phi_n)
        num[:-1] = self.phi_n[1:]
        num[-1] = self.phi_n[0]
        denom = num - self.phi_n
        unsolvable = np.abs(denom) < DENOM_TOL
        # print("unsolvable", unsolvable)
        on_crack = np.abs(self.phi_n) < ZERO_TOL
        # print("on_crack", on_crack)
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

        prod_13 = phi1 * phi3
        prod_24 = phi2 * phi4
        diag_13_bad = prod_24 < prod_13

        if diag_13_bad:
            self.NAT_1 = self.NAT_24_1
            self.NAT_2 = self.NAT_24_2
            A = -phi1 + phi2 - phi3 + phi4
            B = phi1 - 2.0 * phi2 + phi3
            C = phi2
            num_diag = phi2
            denom_diag = phi2 - phi4
            n11, n12 = 1, 2
            n22, n23 = 3, 0
        else:
            self.NAT_1 = self.NAT_13_1
            self.NAT_2 = self.NAT_13_2
            A = phi1 - phi2 + phi3 - phi4
            B = -2.0 * phi1 + phi2 + phi4
            C = phi1
            num_diag = phi1
            denom_diag = phi1 - phi3
            n11, n12 = 0, 1
            n22, n23 = 2, 3

        unsolvable_diag = np.abs(denom_diag) < DENOM_TOL
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

        sign_B = np.sign(B)
        sign_B = np.where(sign_B == 0, 1.0, sign_B)
        q = -0.5 * (B + sign_B * sqrt_disc)

        safe_q = np.where(np.abs(q) < 1e-15, 1.0, q)

        root1 = q / safe_A
        root2 = C / safe_q

        valid_root1 = (root1 >= -ROOT_TOL) & (root1 <= 1.0 + ROOT_TOL)
        valid_root2 = (root2 >= -ROOT_TOL) & (root2 <= 1.0 + ROOT_TOL)

        s_quad = np.where(valid_root2, root2, np.where(valid_root1, root1, s_linear))

        N1_diag = np.where(use_linear, s_linear, s_quad)

        N1_diag = np.where(
            unsolvable_diag | (np.abs(num_diag - denom_diag) < ZERO_TOL), 1.0, N1_diag
        )
        N1_diag = np.clip(N1_diag, 0, 1)

        Nc1 = np.array(
            [
                [N1[n11], 0, 1 - N1_diag],
                [1 - N1[n11], N1[n12], 0],
                [0, 1 - N1[n12], N1_diag],
            ]
        )
        Nc2 = np.array(
            [
                [1 - N1_diag, 0, 1 - N1[n23]],
                [N1_diag, N1[n22], 0],
                [0, 1 - N1[n22], N1[n23]],
            ]
        )
        return Nc1, Nc2

    def _cal_tip_nat_coords(self):
        xi, eta = 0.0, 0.0
        ZERO_TOL = 1e-10
        for _ in range(50):
            N, dN_dxi = self._base_shape_functions(np.stack([xi, eta], axis=0))
            N = N[0]
            dN_dxi = dN_dxi[0]
            val_t = np.dot(N, self.phi_t)
            val_n = np.dot(N, self.phi_n)
            if np.abs(val_t) < ZERO_TOL and np.abs(val_n) < ZERO_TOL:
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
        nat_coords = np.zeros((2, points.shape[0]))
        x_e = self.node_coords
        for _ in range(100):
            N, dN_dxi = self._base_shape_functions(nat_coords)
            dx = N @ x_e - points
            if np.all(np.isclose(np.sum(dx**2, axis=1), 0.0, atol=1e-12)):
                return nat_coords
            J = dN_dxi @ x_e

            step = np.linalg.solve(J.transpose(0, 2, 1), dx[:, :, None])
            nat_coords[:, :] -= step[:, :, 0].T
        raise ValueError("newton iterations didn't converge")

    def _cal_curved_edge_node(self, p1, p2):
        ZERO_TOL = 1e-15
        p_mid = (p1 + p2) / 2
        t_vec = p2 - p1

        p4 = p_mid.copy()
        phi_n = self.phi_n

        for _ in range(50):
            N, dN_dxi = self._base_shape_functions(p4)
            N, dN_dxi = N[0], dN_dxi[0]

            phi_n4 = np.dot(N, phi_n)
            constraint = np.dot(t_vec, p4 - p_mid)

            if np.abs(phi_n4) < ZERO_TOL and np.abs(constraint) < ZERO_TOL:
                break

            grad_phi = dN_dxi @ phi_n

            J = np.array([[grad_phi[0], grad_phi[1]], [t_vec[0], t_vec[1]]])

            step = np.linalg.solve(J, [-phi_n4, -constraint])
            p4 += step
        else:
            raise ValueError("newton iterations didn't converge")

        return p4

    def _integrate_sub_tri(self, Ke, Me, Nc, nat_x_e, eval_mass):
        DET_TOL = -1e-14
        if self.t_enrich:
            rule, correction = qd.TRI_RULES[13]
        else:
            rule, correction = qd.TRI_RULES[5]
        x_e = self.node_coords

        force_linear = False

        while True:
            Ke_temp = np.zeros_like(Ke)
            Me_temp = np.zeros_like(Me)
            is_valid = True

            for Ni, detJi in cut_embedding_tri_iter(Nc):
                xi = rule[:, 0]
                eta = rule[:, 1]
                w = rule[:, 2]
                nat_sub_x_e = nat_x_e.T @ Ni
                N, _ = self._base_shape_functions(nat_sub_x_e)
                sub_phi_n = N @ self.phi_n

                nat_coords_sub, detJi_mod, sign = self._get_mapped_coords(
                    xi,
                    eta,
                    sub_phi_n,
                    True,
                    nat_sub_x_e,
                    detJi,
                    force_linear,
                )
                if np.any(detJi_mod < DET_TOL):
                    is_valid = False
                    break

                N, dN_dxi_sub = self.shape_functions(nat_coords_sub, enforce_sign=sign)
                J = dN_dxi_sub[:, :, 0 : self.N_FN] @ x_e
                detJ = np.linalg.det(J)
                if np.any(np.isnan(detJi)):
                    print("nan occured at third one")
                    print("nodes", self.node_coords)
                    print("phi_n", self.phi_n)
                    print("phi_t", self.phi_t)
                dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
                B = cal_B_2d_vec(dN_dxy_sub)
                w_eff = w * correction * detJ * detJi_mod

                Ke_temp += np.sum(
                    B.transpose(0, 2, 1) @ self.C @ B * w_eff[:, None, None], axis=0
                )
                if eval_mass:
                    N_2d = cal_N_2d_vec(N)
                    Me_temp[:, :] += np.sum(
                        (N_2d.transpose(0, 2, 1) @ N_2d) * w_eff[:, None, None],
                        axis=0,
                    )

            if is_valid:
                Ke += Ke_temp
                if eval_mass:
                    Me += Me_temp
                break
            else:
                print("warning had to fall back to linear triangles (full cut)")
                print(self.phi_n)
                force_linear = True

    def _get_mapped_coords(
        self,
        xi,
        eta,
        sub_phi_n,
        behind_tip,
        nat_sub_x_e,
        detJi,
        force_linear,
    ):
        ZERO_TOL = 1e-10
        on_crack = np.abs(sub_phi_n) < ZERO_TOL
        is_on_crack = np.sum(on_crack & behind_tip) == 2
        p1_idx = np.where(~on_crack)[0][0]
        sign = np.sign(sub_phi_n[p1_idx])

        if not force_linear and is_on_crack:
            c1, c2 = np.where(on_crack)[0]
            p4 = self._cal_curved_edge_node(nat_sub_x_e[:, c1], nat_sub_x_e[:, c2])
            nat_sub_x_e_ext = np.column_stack([nat_sub_x_e, p4])

            L = np.array([1.0 - xi - eta, xi, eta])
            dL_dxi = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])

            N4 = 4.0 * L[c1] * L[c2]
            dN4_dxi = np.array(
                [
                    4.0 * (dL_dxi[c1, 0] * L[c2] + L[c1] * dL_dxi[c2, 0]),
                    4.0 * (dL_dxi[c1, 1] * L[c2] + L[c1] * dL_dxi[c2, 1]),
                ]
            )

            N = np.zeros((4, len(xi)))
            N[:3] = L
            N[c1] -= 0.5 * N4
            N[c2] -= 0.5 * N4
            N[3] = N4

            dn_dxi = np.zeros((2, 4, len(xi)))
            dn_dxi[:, :3, :] = dL_dxi.T[:, :, None]
            dn_dxi[:, [c1, c2], :] -= 0.5 * dN4_dxi[:, None, :]
            dn_dxi[:, 3, :] = dN4_dxi

            dn_dxi = dn_dxi.transpose(2, 0, 1)

            Ji = dn_dxi @ nat_sub_x_e_ext.T
            detJi = np.linalg.det(Ji)
            if np.any(np.isnan(detJi)):
                print("nan occured at fourth one")
                print("nodes", self.node_coords)
                print("phi_n", self.phi_n)
                print("phi_t", self.phi_t)
            nat_coords_sub = nat_sub_x_e_ext @ N
            return nat_coords_sub, detJi, sign
        else:
            N = np.array([1.0 - xi - eta, xi, eta])
            nat_coords_sub = nat_sub_x_e @ N
            return nat_coords_sub, 4 * detJi, sign

    def _integrate_partial_cut(
        self, Ke, Me, tip, Nc, range, nat_x_e, rule, correction, eval_mass
    ):
        DET_TOL = -1e-14
        x_e = self.node_coords
        Ni_template = np.zeros((3, 3))
        Ni_template[:, 0] = tip

        force_linear = False
        while True:
            Ke_temp = np.zeros_like(Ke)
            Me_temp = np.zeros_like(Me)
            is_valid = True

            for Ni, detJi in partial_cut_embedding_tri_iter(Nc, tip, range):
                if detJi < DET_TOL:
                    print("DetJi smaller than 0", detJi)
                nat_sub_x_e = nat_x_e.T @ Ni
                N, _ = self._base_shape_functions(nat_sub_x_e)
                sub_phi_n = N @ self.phi_n
                sub_phi_t = N @ self.phi_t
                behind_tip = sub_phi_t < 1e-10

                x_e_i = self._base_shape_functions(nat_sub_x_e)[0] @ x_e
                duffy = DuffyDistance(x_e_i)
                u, v = rule[:, 0], rule[:, 1]
                N_gp = len(u)

                xi_d_1, eta_d_1, w_d_1 = duffy.transform(u, v, beta=1)
                xi_d_2, eta_d_2, w_d_2 = duffy.transform(u, v, beta=2)

                xi_d_all = np.concatenate([xi_d_1, xi_d_2])
                eta_d_all = np.concatenate([eta_d_1, eta_d_2])
                w_d_all = np.concatenate([w_d_1, w_d_2])
                rule_w_all = np.tile(rule[:, 2], 2)  # Repeat the Gauss weights

                nat_coords_sub, detJi_mod, sign = self._get_mapped_coords(
                    xi_d_all,
                    eta_d_all,
                    sub_phi_n,
                    behind_tip,
                    nat_sub_x_e,
                    detJi,
                    force_linear,
                )

                if np.any(detJi_mod < DET_TOL):
                    is_valid = False
                    break

                N, dN_dxi_sub = self.shape_functions(nat_coords_sub, enforce_sign=sign)
                J = dN_dxi_sub[:, :, 0 : self.N_FN] @ x_e
                detJ = np.linalg.det(J)
                if np.any(np.isnan(detJ)):
                    print("nan occured at fifth one")
                    print("nodes", self.node_coords)
                    print("phi_n", self.phi_n)
                    print("phi_t", self.phi_t)
                dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)

                w_eff_all = rule_w_all * correction * w_d_all * detJi_mod * detJ

                B_all = cal_B_2d_vec(dN_dxy_sub[:, :, : self.N_FN])
                TIP_B_all = cal_B_2d_vec(dN_dxy_sub[:, :, self.N_FN :])

                begin_tip = self.N_DOFS

                Ke_temp[begin_tip:, begin_tip:] += np.sum(
                    TIP_B_all[:N_gp].transpose(0, 2, 1)
                    @ self.C
                    @ TIP_B_all[:N_gp]
                    * w_eff_all[:N_gp, None, None],
                    axis=0,
                )

                res = np.sum(
                    B_all[N_gp:].transpose(0, 2, 1)
                    @ self.C
                    @ TIP_B_all[N_gp:]
                    * w_eff_all[N_gp:, None, None],
                    axis=0,
                )
                Ke_temp[0:begin_tip, begin_tip:] += res
                Ke_temp[begin_tip:, 0:begin_tip] += res.T

                if eval_mass:
                    N_2d = cal_N_2d_vec(N[:N_gp])
                    Me_temp[:, :] += np.sum(
                        (N_2d.transpose(0, 2, 1) @ N_2d) * w_eff_all[:N_gp, None, None],
                        axis=0,
                    )

            if is_valid:
                # All sub-triangles mapped successfully without folding.
                Ke += Ke_temp
                if eval_mass:
                    Me += Me_temp
                break
            else:
                # A sub-triangle folded over.
                print("warning had to fall back to linear triangles (partial cut)")
                print("phi_n", self.phi_n)
                force_linear = True

    def _cubic_shape_functions(self, nat_coords):
        xi = np.atleast_1d(nat_coords[0])
        eta = np.atleast_1d(nat_coords[1])
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

    def _base_shape_functions(self, nat_coords):
        xi = np.atleast_1d(nat_coords[0])
        eta = np.atleast_1d(nat_coords[1])
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

    def nearest_point_on_crack(self, coords):
        coords_2d = np.atleast_2d(coords)
        N_pts = coords_2d.shape[0]
        print(coords_2d)

        try:
            nat_coords = self._cal_nat_coords(coords_2d)
        except ValueError:
            nat_coords = np.zeros((2, N_pts))

        d2N_dxideta = np.array([0.25, -0.25, 0.25, -0.25])
        dX_dxideta = d2N_dxideta @ self.node_coords  # Shape: (2,)
        dphi_dxideta = np.dot(d2N_dxideta, self.phi_n)  # Scalar

        X_proj = np.zeros_like(coords_2d)
        converged = np.zeros(N_pts, dtype=bool)

        for _ in range(50):
            if np.all(converged):
                break
            N, dN_dxi = self._base_shape_functions(nat_coords)
            X = N @ self.node_coords
            dx = X - coords_2d
            phi_n_val = N @ self.phi_n
            grad_phi_nat = np.einsum("nij,j->ni", dN_dxi, self.phi_n)
            J_map = dN_dxi @ self.node_coords  # (N_pts, 2, 2)
            v = np.stack([-grad_phi_nat[:, 1], grad_phi_nat[:, 0]], axis=1)
            t = np.einsum("nij,ni->nj", J_map, v)
            orth_val = np.sum(dx * t, axis=1)
            current_converged = np.isclose(phi_n_val, 0.0, atol=1e-12) & np.isclose(
                orth_val, 0.0, atol=1e-12
            )
            X_proj[current_converged] = X[current_converged]
            converged |= current_converged
            active = ~converged
            if not np.any(active):
                break
            # Derivative of the tangent vector w.r.t xi and eta
            dt_dxi = (J_map[active, 0, :] * -dphi_dxideta) + (
                dX_dxideta * grad_phi_nat[active, 0, np.newaxis]
            )
            dt_deta = (dX_dxideta * -grad_phi_nat[active, 1, np.newaxis]) + (
                J_map[active, 1, :] * dphi_dxideta
            )
            J_NR = np.zeros((np.sum(active), 2, 2))
            J_NR[:, 0, 0] = grad_phi_nat[active, 0]
            J_NR[:, 0, 1] = grad_phi_nat[active, 1]
            J_NR[:, 1, 0] = np.sum(J_map[active, 0, :] * t[active], axis=1) + np.sum(
                dx[active] * dt_dxi, axis=1
            )
            J_NR[:, 1, 1] = np.sum(J_map[active, 1, :] * t[active], axis=1) + np.sum(
                dx[active] * dt_deta, axis=1
            )
            Residual = np.stack([-phi_n_val[active], -orth_val[active]], axis=1)
            try:
                step = np.linalg.solve(J_NR, Residual[..., np.newaxis]).squeeze(-1)
            except np.linalg.LinAlgError:
                break
            nat_coords[0, active] += step[:, 0]
            nat_coords[1, active] += step[:, 1]
        inside = (np.abs(nat_coords[0, :]) <= 1.0 + 1e-6) & (
            np.abs(nat_coords[1, :]) <= 1.0 + 1e-6
        )
        valid = converged & inside

        if not np.all(valid):
            edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
            all_intersections = []
            # Find the exact zero-crossings on the 4 physical edges
            for i, j in edges:
                if self.phi_n[i] * self.phi_n[j] <= 0:
                    denom = self.phi_n[i] - self.phi_n[j]
                    if np.isclose(denom, 0.0):
                        all_intersections.extend(
                            [self.node_coords[i], self.node_coords[j]]
                        )
                    else:
                        t_intersect = self.phi_n[i] / denom
                        pt = self.node_coords[i] + t_intersect * (
                            self.node_coords[j] - self.node_coords[i]
                        )
                        all_intersections.append(pt)

            all_intersections = np.unique(np.round(all_intersections, 8), axis=0)
            outside_idx = np.where(~valid)[0]

            for idx in outside_idx:
                dists = np.linalg.norm(all_intersections - coords_2d[idx], axis=1)
                best_pt = all_intersections[np.argmin(dists)]
                X_proj[idx] = best_pt

                try:
                    nat_coords[:, idx] = self._cal_nat_coords(best_pt).flatten()
                except ValueError:
                    pass

        if np.asarray(coords).ndim == 1:
            return X_proj[0], nat_coords[:, 0]

        return X_proj, nat_coords

    def jump_shape_functions(self, nat_coords, tip_coords):
        return jump_shape_functions(
            self,
            self._base_shape_functions,
            self._base_shape_functions,
            # self._cubic_shape_functions,
            nat_coords,
            tip_coords,
        )

    def shape_functions(
        self,
        nat_coords,
        phi_n=None,
        phi_t=None,
        dphi_n_dxi=None,
        dphi_t_dxi=None,
        enforce_sign=None,
    ):
        return enriched_shape_functions(
            self,
            self._base_shape_functions,
            self._base_shape_functions,
            # self._cubic_shape_functions,
            nat_coords,
            phi_n,
            phi_t,
            dphi_n_dxi,
            dphi_t_dxi,
            enforce_sign,
        )

    def cal_stresses(self, nat_coords, Ue):
        Ue = np.asarray(Ue, dtype=float).ravel()
        _, dN_dxi = self.shape_functions(nat_coords)
        J = dN_dxi[:, :, : self.N_FN] @ self.node_coords
        dN_dxy = np.linalg.solve(J, dN_dxi)
        B = cal_B_2d_vec(dN_dxy)
        eps = B @ Ue
        sig = self.C @ eps[:, :, None]
        return sig

    def stresses_at_nodes(self, Ue):
        xi = np.array([-1.0, 1.0, 1.0, -1.0])
        eta = np.array([-1.0, -1.0, 1.0, 1.0])
        return self.cal_stresses(np.stack([xi, eta], axis=0), Ue)
