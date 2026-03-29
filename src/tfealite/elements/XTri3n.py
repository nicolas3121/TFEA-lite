from typing import Final

import numpy as np

from ..core import quadratures as qd
from ..core.quadratures import DuffyDistance
from .Tri3n import Tri3n
from .utils import (
    cal_B_2d_vec,
    cut_embedding_tri_iter,
    enriched_shape_functions,
    partial_cut_embedding_tri_iter,
)


class XTri3n(Tri3n):
    NODES: Final = 3
    DOFS: Final = 2
    BRANCH_FN: Final = 4  # branch functions
    N_FN: Final = NODES
    H_FN: Final = NODES
    TIP_FN: Final = NODES * BRANCH_FN
    N_DOFS: Final = DOFS * N_FN
    H_DOFS: Final = DOFS * H_FN
    TIP_DOFS: Final = DOFS * TIP_FN

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
            return Tri3n(node_coords, material, real)
        assert h_enrich is not None
        assert t_enrich is not None
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
        in_range=np.ones(3, dtype=bool),
    ):
        super().__init__(node_coords, material, real)
        self.phi_n = phi_n
        self.phi_t = phi_t
        self.h_enrich = h_enrich
        self.t_enrich = t_enrich
        self.partial_cut = partial_cut
        self.in_range = in_range

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
        Nc = np.array(
            [
                [N1[0], 0, 1 - N1[2]],
                [1 - N1[0], N1[1], 0],
                [0, 1 - N1[1], N1[2]],
            ]
        )
        return Nc

    def cal_element_matrices(self, eval_mass=False):
        n = (
            self.N_DOFS
            + int(self.h_enrich) * self.H_DOFS
            + int(self.t_enrich) * self.TIP_DOFS
        )
        Ke = np.zeros((n, n))
        x_e = self.node_coords

        xi, eta = 1 / 3, 1 / 3
        weight = 1 / 2

        _, dN_dxi = self._base_shape_functions(xi, eta)
        J = dN_dxi[0, :, :] @ x_e
        detJ = np.linalg.det(J)
        Nc = None
        if self.h_enrich or self.partial_cut:
            Nc = self._cal_intersections()
        else:
            (rule, correction) = qd.TRI_RULES[10]
            xi = rule[:, 0]
            eta = rule[:, 1]
            _, dN_dxi = self.shape_functions(xi, eta)
            dN_dxy = np.linalg.solve(J, dN_dxi)
            B = cal_B_2d_vec(dN_dxy)
            w_eff = rule[:, 2] * correction * detJ
            Ke[:, :] = np.sum(
                (B.transpose(0, 2, 1) @ self.C @ B) * w_eff[:, None, None], axis=0
            )
        if self.partial_cut:
            dN_dxy = np.linalg.solve(J, dN_dxi)
            B = cal_B_2d_vec(dN_dxy)
            w_eff = weight * detJ
            Ke[: self.N_DOFS, : self.N_DOFS] = np.sum(
                (B.transpose(0, 2, 1) @ self.C @ B) * w_eff, axis=0
            )
            assert Nc is not None
            self._integrate_partial_cut(Ke, Nc, J, detJ, B)
        elif self.h_enrich:
            assert Nc is not None
            if self.t_enrich:
                rule, correction = qd.TRI_RULES[10]
            else:
                rule, correction = qd.TRI_RULES[1]
            for Ni, detJi in cut_embedding_tri_iter(Nc):
                xi = rule[:, 0]
                eta = rule[:, 1]
                n, _ = self._base_shape_functions(xi, eta)
                print(J.T)
                xi_sub, eta_sub = np.linalg.solve(
                    J.T, x_e.T @ Ni @ n.T - x_e[0, :, None]
                )
                _, dN_dxi_sub = self.shape_functions(xi_sub, eta_sub)
                dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
                B = cal_B_2d_vec(dN_dxy_sub)
                w_eff = rule[:, 2] * detJi * correction * detJ
                Ke[:, :] += np.sum(
                    (B.transpose(0, 2, 1) @ self.C @ B) * w_eff[:, None, None], axis=0
                )

        if eval_mass:
            raise NotImplementedError
        else:
            return Ke

    def _integrate_partial_cut(self, Ke, Nc, J, detJ, B):
        x_e = self.node_coords
        (rule, correction) = qd.QUAD_RULES[10]
        rule = rule.copy()
        rule[:, 0:2] = (1 + rule[:, 0:2]) / 2
        rule[:, 2] /= 4
        tip = np.linalg.solve(
            np.array([self.phi_t, self.phi_n, [1, 1, 1]]), np.array([0, 0, 1])
        )
        for Ni, detJi in partial_cut_embedding_tri_iter(Nc, tip, range(6)):
            if detJi < 0:
                print("DetJi smaller than 0")
            x_e_i = (x_e.T @ Ni).T
            duffy = DuffyDistance(x_e_i)
            u, v = rule[:, 0], rule[:, 1]

            xi_d, eta_d, w_d = duffy.transform(u, v, beta=1)
            n, _ = self._base_shape_functions(xi_d, eta_d)
            xi_sub, eta_sub = np.linalg.solve(J.T, x_e.T @ Ni @ n.T - x_e[0, :, None])
            _, dN_dxi_sub = self.shape_functions(xi_sub, eta_sub)
            dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
            TIP_B = cal_B_2d_vec(dN_dxy_sub[:, :, self.N_FN :])
            w_eff = rule[:, 2] * correction * w_d * detJi * detJ
            begin_tip = self.N_DOFS
            Ke[begin_tip:, begin_tip:] += np.sum(
                (TIP_B.transpose(0, 2, 1) @ self.C @ TIP_B) * w_eff[:, None, None],
                axis=0,
            )

            xi_d, eta_d, w_d = duffy.transform(u, v, beta=2)
            n, _ = self._base_shape_functions(xi_d, eta_d)
            xi_sub, eta_sub = np.linalg.solve(J.T, x_e.T @ Ni @ n.T - x_e[0, :, None])
            _, dN_dxi_sub = self.shape_functions(xi_sub, eta_sub)
            dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
            TIP_B = cal_B_2d_vec(dN_dxy_sub[:, :, self.N_FN :])
            w_eff = rule[:, 2] * correction * w_d * detJi * detJ
            begin_tip = self.N_DOFS
            res = np.sum(
                B.transpose(0, 2, 1) @ self.C @ TIP_B * w_eff[:, None, None], axis=0
            )
            Ke[0:begin_tip, begin_tip:] += res
            Ke[begin_tip:, 0:begin_tip] += res.T

    def _base_shape_functions(self, xi, eta):
        xi = np.atleast_1d(xi)
        eta = np.atleast_1d(eta)
        N = np.array([1 - xi - eta, xi, eta]).T
        dN_dxi = np.tile(
            np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]), (xi.shape[0], 1, 1)
        )
        return N, dN_dxi

    def _quadratic_shape_functions(self, xi, eta):
        N, dN_dxi = self._base_shape_functions(xi, eta)
        N2 = np.empty_like(N)
        N2[:-1] = N[:-1] * N[1:]
        N2[-1] = N[0] * N[-1]
        dN2_dxi = np.empty_like(dN_dxi)
        dN2_dxi[:, :-1] = N[None, :-1] * dN_dxi[:, 1:] + dN_dxi[:, :-1] * N[None, 1:]
        dN2_dxi[:, -1] = N[None, -1] * dN_dxi[:, 0] + dN_dxi[:, -1] * N[None, 0]
        return N2, dN2_dxi

    def _cubic_shape_functions(self, xi, eta):
        L1 = 1.0 - xi - eta
        L2 = xi
        L3 = eta
        Q1 = 3 * L1**2 - 2 * L1**3
        Q2 = 3 * L2**2 - 2 * L2**3
        Q3 = 3 * L3**2 - 2 * L3**3
        Q = np.array([Q1, Q2, Q3]).T
        dQ_dL1 = 6 * L1 * (1.0 - L1)
        dQ_dL2 = 6 * L2 * (1.0 - L2)
        dQ_dL3 = 6 * L3 * (1.0 - L3)
        dQ1_dxi = -dQ_dL1
        dQ2_dxi = dQ_dL2
        dQ3_dxi = np.zeros_like(xi)
        dQ1_deta = -dQ_dL1
        dQ2_deta = np.zeros_like(eta)
        dQ3_deta = dQ_dL3
        row1 = [dQ1_dxi, dQ2_dxi, dQ3_dxi]
        row2 = [dQ1_deta, dQ2_deta, dQ3_deta]
        dQ_dxi = np.stack([row1, row2]).transpose(2, 0, 1)
        return Q, dQ_dxi

    def shape_functions(self, xi, eta):
        return enriched_shape_functions(
            self, self._base_shape_functions, self._cubic_shape_functions, xi, eta
        )

    def stresses_at_nodes(self, Ue):
        Ue = np.asanyarray(Ue, dtype=float).ravel()
        raise NotImplementedError
