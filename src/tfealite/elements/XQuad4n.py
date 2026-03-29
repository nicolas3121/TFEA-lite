from typing import Final

import numpy as np

from ..core import quadratures as qd
from ..core.quadratures import DuffyDistance
from .Quad4n import Quad4n
from .utils import (
    cal_B_2d_vec,
    enriched_shape_functions,
    cut_embedding_tri_iter,
    partial_cut_embedding_tri_iter,
    jump_shape_functions,
)


# in crack tip element locaal coordinaten stelsel definieren op basis van level set op tip
# in plaats van level set te interpoleren rechtstreeks r en theta daarmee berekenen dan niet meer mogelijk dat sign change tegen gekomen wordt door enrichment
# en perfecte orthogonaliteit van 2 level sets daar binnen element
# kan in theorie mogelijk iets gelijkaardig doen voor geometrical enrichment elementen maar daar nog niet helemaal zeker
# voor de intersecties in elke sub triangle de waarde van de level sets berekenen
# 2de punt als referentie om crack tip coordinate system te definieren is die met phi_n = 0, phi_t < 0
# indien scheur exact op punt / edge ligt en er is geen ander punt in element dat doorsneden wordt is waarschijnlijk gewoon afgeleide in dat punt --> zoals normaal bepalen
# voor elementen met geometrical enrichment achter scheur punt die ook heaviside enrichment hebben lineair interpolleren binnen sub driehoek om sign change te vermijden


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

    NAT_1: Final = np.array([[-1, -1], [1, -1], [1, 1]])
    NAT_2: Final = np.array([[-1, -1], [1, 1], [-1, 1]])

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
        self.tip_coords = None
        self.tip_n = None
        self.tip_t = None

    def cal_element_matrices(self, eval_mass=False):
        n = (
            self.N_DOFS
            + int(self.h_enrich) * self.H_DOFS
            + int(self.t_enrich) * self.TIP_DOFS
        )
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
            J = dN_dxi[:, :, : self.N_FN] @ x_e
            detJ = np.linalg.det(J)
            dN_dxy = np.linalg.solve(J, dN_dxi)
            B = cal_B_2d_vec(dN_dxy)
            w_eff = rule[:, 2] * correction * detJ
            Ke[:, :] = np.sum(
                (B.transpose(0, 2, 1) @ self.C @ B) * w_eff[:, None, None], axis=0
            )
        if self.partial_cut:
            Ke[: self.N_DOFS, : self.N_DOFS] = super().cal_element_matrices(
                eval_mass=False
            )
            assert Nc1 is not None and Nc2 is not None
            assert not self.h_enrich
            (rule, correction) = qd.QUAD_RULES[10]
            rule = rule.copy()
            rule[:, 0:2] = (1 + rule[:, 0:2]) / 2
            rule[:, 2] /= 4
            xi_tip, eta_tip = self._cal_tip_nat_coords()
            tri1_coords = np.array([[-1, 1, 1], [-1, -1, 1], [1, 1, 1]])
            tip1 = np.linalg.solve(tri1_coords, [xi_tip, eta_tip, 1.0])

            tri2_coords = np.array([[-1, 1, -1], [-1, 1, 1], [1, 1, 1]])
            tip2 = np.linalg.solve(tri2_coords, [xi_tip, eta_tip, 1.0])

            self._set_tip_var(xi_tip, eta_tip, Nc1, Nc2)

            self._integrate_partial_cut(
                Ke,
                tip1,
                Nc1,
                range(4),
                self.NAT_1,
                rule,
                correction,
            )
            self._integrate_partial_cut(
                Ke,
                tip2,
                Nc2,
                range(2, 6),
                self.NAT_2,
                rule,
                correction,
            )
        elif self.h_enrich:
            assert Nc1 is not None and Nc2 is not None
            self._integrate_sub_tri(
                Ke, Nc1, self.NAT_1, self.phi_n[:-1], self.phi_t[:-1]
            )
            self._integrate_sub_tri(
                Ke, Nc2, self.NAT_2, self.phi_n[[0, 2, 3]], self.phi_t[[0, 2, 3]]
            )

        if eval_mass:
            raise NotImplementedError
        return Ke

    def _set_tip_var(self, xi_tip, eta_tip, Nc1, Nc2):
        pass
        phi_n1 = self.phi_n[:-1] @ Nc1[:, :-1]
        phi_t1 = self.phi_t[:-1] @ Nc1[:, :-1]
        phi_n2 = self.phi_n[[0, 2, 3]] @ Nc2[:, 1:]
        phi_t2 = self.phi_t[[0, 2, 3]] @ Nc2[:, 1:]
        in_sub_1 = np.where(np.isclose(phi_n1, 0.0, atol=1e-12) & (phi_t1 <= 0))[0]
        in_sub_2 = np.where(np.isclose(phi_n2, 0.0, atol=1e-12) & (phi_t2 <= 0))[0]

        N, dN_dxi = self._base_shape_functions(xi_tip, eta_tip)
        N, dN_dxi = N[0], dN_dxi[0]
        tip = self.node_coords.T @ N

        if len(in_sub_1) != 0:
            origin = self.node_coords[:-1, :].T @ Nc1[:, in_sub_1[0]]
        else:
            origin = self.node_coords[[0, 2, 3], :].T @ Nc2[:, 1 + in_sub_2[0]]
        if not np.allclose(origin, tip, atol=1e-12):
            t = tip - origin
        else:
            in_sub_1_fwd = np.where(
                np.isclose(phi_n1, 0.0, atol=1e-12) & (phi_t1 >= 0)
            )[0]
            in_sub_2_fwd = np.where(
                np.isclose(phi_n2, 0.0, atol=1e-12) & (phi_t2 >= 0)
            )[0]
            if len(in_sub_1_fwd) != 0:
                origin = self.node_coords[:-1, :].T @ Nc1[:, in_sub_1_fwd[0]]
            else:
                origin = self.node_coords[[0, 2, 3], :].T @ Nc2[:, 1 + in_sub_2_fwd[0]]
            t = origin - tip
        t = t / np.linalg.norm(t)
        J = dN_dxi @ self.node_coords
        dN_dxy = np.linalg.solve(J, dN_dxi)
        grad_phi_n = dN_dxy @ self.phi_n

        n_candidate_1 = np.array([-t[1], t[0]])
        n_candidate_2 = np.array([t[1], -t[0]])
        if np.dot(n_candidate_1, grad_phi_n) > 0:
            n = n_candidate_1
        else:
            n = n_candidate_2
        self.tip_coords = tip
        self.tip_n = n
        self.tip_t = t

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
            N, dN_dxi = self._base_shape_functions(xi, eta)
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
        for _ in range(100):
            N, dN_dxi = self._base_shape_functions(xi, eta)
            dx = N @ x_e - points
            if np.all(np.isclose(np.sum(dx**2, axis=1), 0.0, atol=1e-15)):
                return xi, eta
            J = dN_dxi @ x_e

            step = np.linalg.solve(J.transpose(0, 2, 1), dx[:, :, None])
            xi -= step[:, 0, 0]
            eta -= step[:, 1, 0]
        raise ValueError("newton iterations didn't converge")

    def _integrate_sub_tri(self, Ke, Nc, nat_x_e, phi_n, phi_t):
        w_tot = 0
        if self.t_enrich:
            rule, correction = qd.TRI_RULES[10]
        else:
            rule, correction = qd.TRI_RULES[3]
        x_e = self.node_coords
        for Ni, detJi in cut_embedding_tri_iter(Nc):
            xi = rule[:, 0]
            eta = rule[:, 1]
            w = rule[:, 2]
            nat_sub_x_e = nat_x_e.T @ Ni
            n = np.array([1 - xi - eta, xi, eta])
            dn_dxi = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
            xi_sub, eta_sub = nat_x_e.T @ Ni @ n

            sub_J = dn_dxi @ nat_sub_x_e.T
            N, _ = self._base_shape_functions(nat_sub_x_e[0, :], nat_sub_x_e[1, :])
            sub_phi_n = N @ self.phi_n
            sub_phi_t = N @ self.phi_t
            phi_n_sub = sub_phi_n @ n
            phi_t_sub = sub_phi_t @ n
            dphi_n_sub_dxi = np.linalg.solve(sub_J, dn_dxi @ sub_phi_n)
            dphi_t_sub_dxi = np.linalg.solve(sub_J, dn_dxi @ sub_phi_t)

            _, dN_dxi_sub = self.shape_functions(
                xi_sub, eta_sub, phi_n_sub, phi_t_sub, dphi_n_sub_dxi, dphi_t_sub_dxi
            )
            J = dN_dxi_sub[:, :, 0 : self.N_FN] @ x_e
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

        for Ni, detJi in partial_cut_embedding_tri_iter(Nc, tip, range):
            if detJi < 0:
                print("DetJi smaller than 0", detJi)
            nat_sub_x_e = nat_x_e.T @ Ni
            x_e_i = self._base_shape_functions(nat_sub_x_e[0], nat_sub_x_e[1])[0] @ x_e
            duffy = DuffyDistance(x_e_i)
            u, v = rule[:, 0], rule[:, 1]
            xi_d, eta_d, w_d = duffy.transform(u, v, beta=1)
            n = np.array([1 - xi_d - eta_d, xi_d, eta_d])
            xi_sub, eta_sub = nat_sub_x_e @ n
            _, dN_dxi_sub = self.shape_functions(xi_sub, eta_sub)
            J = dN_dxi_sub[:, :, 0 : self.N_FN] @ x_e
            detJ = np.linalg.det(J)
            dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
            TIP_B = cal_B_2d_vec(dN_dxy_sub[:, :, self.N_FN :])
            w_eff = rule[:, 2] * correction * w_d * detJi * detJ * 4
            begin_tip = self.N_DOFS
            Ke[begin_tip:, begin_tip:] += np.sum(
                (TIP_B.transpose(0, 2, 1) @ self.C @ TIP_B) * w_eff[:, None, None],
                axis=0,
            )

            xi_d, eta_d, w_d = duffy.transform(u, v, beta=2)
            n = np.array([1 - xi_d - eta_d, xi_d, eta_d])
            xi_sub, eta_sub = nat_sub_x_e @ n
            _, dN_dxi_sub = self.shape_functions(xi_sub, eta_sub)
            J = dN_dxi_sub[:, :, 0 : self.N_FN] @ x_e
            detJ = np.linalg.det(J)
            dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
            B = cal_B_2d_vec(dN_dxy_sub[:, :, : self.N_FN])
            TIP_B = cal_B_2d_vec(dN_dxy_sub[:, :, self.N_FN :])
            w_eff = rule[:, 2] * correction * w_d * detJi * detJ * 4
            begin_tip = self.N_DOFS
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

    def _base_shape_functions(self, xi, eta):
        xi = np.atleast_1d(np.asarray(xi))
        eta = np.atleast_1d(np.asarray(eta))
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
        Nc1, Nc2 = self._cal_intersections()

        def project_on_crack(Nc, x_e, coords):
            cols = np.where(np.sum(np.isclose(Nc, 0.0, 1e-12), axis=0) == 1)[0]
            if len(cols) == 2:
                intersections = x_e.T @ Nc[:, cols]
                v = intersections[:, 1] - intersections[:, 0]
                w = coords - intersections[:, 0]
                t = np.clip(np.dot(v, w) / np.dot(v, v), 0.0, 1.0)
                p = intersections[:, 0] + t * v
                d = np.linalg.norm(p - coords)
                return p, d
            return None, np.inf

        touching = np.where(np.isclose(self.phi_n, 0.0, 1e-12))[0]
        p1, d1 = project_on_crack(Nc1, self.node_coords[[0, 1, 2], :], coords)
        p2, d2 = project_on_crack(Nc2, self.node_coords[[0, 2, 3], :], coords)
        if len(touching) == 1:
            print("warning: touching")
            print("coords", coords)
            p = self.node_coords[touching[0], :]

        elif len(touching) == 2:
            nodes = self.node_coords[touching, :]
            v = nodes[1, :] - nodes[0, :]
            w = coords - nodes[0, :]
            t = np.clip(np.dot(v, w) / np.dot(v, v), 0.0, 1.0)
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

    def jump_shape_functions(self, xi, eta, tip_coords):
        return jump_shape_functions(
            self,
            self._base_shape_functions,
            self._base_shape_functions,
            xi,
            eta,
            tip_coords,
        )

    def shape_functions(
        self, xi, eta, phi_n=None, phi_t=None, dphi_n_dxi=None, dphi_t_dxi=None
    ):
        # return enriched_shape_functions(
        #     self, self._base_shape_functions, self._cubic_shape_functions, xi, eta
        # )
        return enriched_shape_functions(
            self,
            self._base_shape_functions,
            self._base_shape_functions,
            xi,
            eta,
            phi_n,
            phi_t,
            dphi_n_dxi,
            dphi_t_dxi,
        )

    def cal_stresses(self, xi, eta, Ue):
        Ue = np.asarray(Ue, dtype=float).ravel()
        _, dN_dxi = self.shape_functions(xi, eta)
        J = dN_dxi[:, :, : self.N_FN] @ self.node_coords
        dN_dxy = np.linalg.solve(J, dN_dxi)
        B = cal_B_2d_vec(dN_dxy)
        eps = B @ Ue
        sig = self.C @ eps[:, :, None]
        return sig

    def stresses_at_nodes(self, Ue):
        xi = np.array([-1.0, 1.0, 1.0, -1.0])
        eta = np.array([-1.0, -1.0, 1.0, 1.0])
        return self.cal_stresses(xi, eta, Ue)
