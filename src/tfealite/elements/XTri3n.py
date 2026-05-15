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
    jump_shape_functions,
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
    NAT_COORDS = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)

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

        _, dN_dxi = self._base_shape_functions(np.array([xi, eta]))
        J = dN_dxi[0, :, :] @ x_e
        detJ = np.linalg.det(J)
        Nc = None
        if self.h_enrich or self.partial_cut:
            Nc = self._cal_intersections()
        else:
            (rule, correction) = qd.TRI_RULES[19]
            nat_coords = rule[:, :2].T
            _, dN_dxi = self.shape_functions(nat_coords)
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
                rule, correction = qd.TRI_RULES[19]
            else:
                rule, correction = qd.TRI_RULES[5]
            for Ni, detJi in cut_embedding_tri_iter(Nc):
                nat_coords = rule[:, :2].T
                n, _ = self._base_shape_functions(nat_coords)
                sub_nat_coords = self.NAT_COORDS.T @ Ni @ n.T
                _, dN_dxi_sub = self.shape_functions(sub_nat_coords)
                dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
                B = cal_B_2d_vec(dN_dxy_sub)
                w_eff = rule[:, 2] * detJi * correction * detJ
                Ke[:, :] += np.sum(
                    (B.transpose(0, 2, 1) @ self.C @ B) * w_eff[:, None, None], axis=0
                )

        if eval_mass:
            raise NotImplementedError
        else:
            return Ke, None

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

            N_gp = len(u)

            xi_d_1, eta_d_1, w_d_1 = duffy.transform(u, v, beta=1)
            xi_d_2, eta_d_2, w_d_2 = duffy.transform(u, v, beta=2)

            xi_d_all = np.concatenate([xi_d_1, xi_d_2])
            eta_d_all = np.concatenate([eta_d_1, eta_d_2])
            w_d_all = np.concatenate([w_d_1, w_d_2])
            rule_w_all = np.tile(rule[:, 2], 2)  # Repeat the Gauss weights

            n = np.array([1 - xi_d_all - eta_d_all, xi_d_all, eta_d_all]).T
            sub_nat_coords = self.NAT_COORDS.T @ Ni @ n.T
            _, dN_dxi_sub = self.shape_functions(sub_nat_coords)
            dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
            TIP_B_all = cal_B_2d_vec(dN_dxy_sub[:, :, self.N_FN :])
            w_eff_all = rule_w_all * correction * w_d_all * detJi * detJ
            begin_tip = self.N_DOFS
            Ke[begin_tip:, begin_tip:] += np.sum(
                (TIP_B_all[:N_gp].transpose(0, 2, 1) @ self.C @ TIP_B_all[:N_gp])
                * w_eff_all[:N_gp, None, None],
                axis=0,
            )

            begin_tip = self.N_DOFS
            res = np.sum(
                B.transpose(0, 2, 1)
                @ self.C
                @ TIP_B_all[N_gp:]
                * w_eff_all[N_gp:, None, None],
                axis=0,
            )
            Ke[0:begin_tip, begin_tip:] += res
            Ke[begin_tip:, 0:begin_tip] += res.T

    def _base_shape_functions(self, nat_coords):
        xi = np.atleast_1d(nat_coords[0])
        eta = np.atleast_1d(nat_coords[1])
        N = np.array([1 - xi - eta, xi, eta]).T
        dN_dxi = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
        return N, dN_dxi[None, :, :]

    def _quadratic_shape_functions(self, nat_coords):
        N, dN_dxi = self._base_shape_functions(nat_coords)
        N2 = np.empty_like(N)
        N2[:-1] = N[:-1] * N[1:]
        N2[-1] = N[0] * N[-1]
        dN2_dxi = np.empty_like(dN_dxi)
        dN2_dxi[:, :-1] = N[None, :-1] * dN_dxi[:, 1:] + dN_dxi[:, :-1] * N[None, 1:]
        dN2_dxi[:, -1] = N[None, -1] * dN_dxi[:, 0] + dN_dxi[:, -1] * N[None, 0]
        return N2, dN2_dxi

    def shape_functions(self, nat_coords):
        return enriched_shape_functions(
            self, self._base_shape_functions, self._base_shape_functions, nat_coords
        )

    def stresses_at_nodes(self, Ue):
        Ue = np.asanyarray(Ue, dtype=float).ravel()
        raise NotImplementedError

    def nearest_point_on_crack(self, coords):
        coords_2d = np.atleast_2d(coords)
        N_pts = coords_2d.shape[0]

        Nc = self._cal_intersections()
        phi_n = self.phi_n @ Nc
        on_crack = np.isclose(phi_n, 0.0, atol=1e-12)
        crack_pts = self.node_coords.T @ Nc
        crack_pts = crack_pts[:, on_crack].T

        if np.all(np.isclose(crack_pts[1:], crack_pts[0], 1e-12)):  # touching node
            crack_indices = np.where(on_crack)[0]
            return np.tile(crack_pts[0], (N_pts, 1)), np.tile(
                self.NAT_COORDS @ Nc[:, crack_indices[0]], (N_pts, 1)
            )

        P_A = crack_pts[0]
        P_B = crack_pts[1]
        v = P_B - P_A
        v_norm_sq = np.dot(v, v)
        w = coords_2d - P_A
        t = np.dot(w, v) / v_norm_sq
        t = np.clip(t, 0.0, 1.0)
        X_proj = P_A + t[:, np.newaxis] * v
        X0, X1, X2 = self.node_coords[0], self.node_coords[1], self.node_coords[2]
        J = np.column_stack([X1 - X0, X2 - X0])
        dx = X_proj - X0
        nat_coords = np.linalg.solve(J, dx.T)
        if np.asarray(coords).ndim == 1:
            return X_proj[0], nat_coords[:, 0]

        return X_proj, nat_coords

    def jump_shape_functions(self, nat_coords, tip_coords):
        return jump_shape_functions(
            self,
            self._base_shape_functions,
            self._base_shape_functions,
            nat_coords,
            tip_coords,
        )
