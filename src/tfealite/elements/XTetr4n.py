from .Tetr4n import Tetr4n
import numpy as np
from typing import Final
from ..core import quadratures as qd
from ..core.quadratures import DuffySinh3D
from .utils import (
    cal_B_3d_vec,
    cut_embedding_tetr_iter,
    enriched_shape_functions,
    partial_cut_embedding_tetr_iter,
    ELEM_EDGES,
)


class XTetr4n(Tetr4n):
    NODES: Final = 4
    DOFS: Final = 3
    BRANCH_FN: Final = 4
    N_FN: Final = NODES
    H_FN: Final = NODES
    LH_FN: Final = 2 * NODES
    TIP_FN: Final = NODES * BRANCH_FN
    N_DOFS: Final = DOFS * N_FN
    H_DOFS: Final = DOFS * H_FN
    TIP_DOFS: Final = DOFS * TIP_FN
    NAT_COORDS = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=float,
    )
    num_edges, denom_edges = ELEM_EDGES["Tetr4n"]

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
            return Tetr4n(node_coords, material, real)
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

    def _base_shape_functions(self, natural_coordinate):
        natural_coordinate = np.atleast_2d(natural_coordinate)
        # print(natural_coordinate)
        xi = natural_coordinate[0, :]
        eta = natural_coordinate[1, :]
        zeta = natural_coordinate[2, :]
        N = np.array([1 - xi - eta - zeta, xi, eta, zeta]).T
        dN_dxi = np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ).T
        return N, dN_dxi[None, :, :]

    def _cal_intersections(self):
        phi_n1 = self.phi_n[self.num_edges]
        phi_n2 = self.phi_n[self.denom_edges]
        num = phi_n1
        denom = num - phi_n2
        unsolvable = np.isclose(denom, 0)
        on_crack = np.isclose(denom, 0)
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
                [N1[0], 0, 0, N1[3], N1[4], 0],
                [1 - N1[0], N1[1], 0, 0, 0, N1[5]],
                [0, 1 - N1[1], N1[2], 1 - N1[3], 0, 0],
                [0, 0, 1 - N1[2], 0, 1 - N1[4], 1 - N1[5]],
            ]
        )
        phi_n12 = N1 * phi_n2 + (1 - N1) * phi_n1
        on_interface = np.isclose(phi_n12, 0.0, atol=1e-12)
        kappa = np.sum(on_interface[None, :] * Nc, axis=1) / np.sum(on_interface)
        return Nc, kappa

    def _cal_front_intersections(self):
        tip = np.empty((4, 4))
        tip_on_interface = [True, True, True, True]
        B = np.array([0, 0, 1])
        for j in range(4):
            tip[j, j] = 0
            i, r, g = (j + 1) % 4, (j + 2) % 4, (j + 3) % 4
            phi_n_face = self.phi_n[[i, r, g]]
            phi_t_face = self.phi_t[[i, r, g]]

            A = np.array([phi_n_face, phi_t_face, [1, 1, 1]])
            cj = None
            try:
                cj = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                pass
            if cj is None or not np.all(cj >= -1e-10):
                num = np.empty_like(phi_n_face)
                num[:-1] = phi_n_face[1:]
                num[-1] = phi_n_face[0]
                denom = num - phi_n_face
                unsolvable = np.isclose(denom, 0.0)
                on_crack = np.isclose(phi_n_face, 0.0)
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
                phi_n12 = N1 * phi_n_face + (1 - N1) * num
                on_interface = np.isclose(phi_n12, 0.0, 1e-12)
                if not np.any(on_interface):
                    cj = np.full(3, 1 / 3)
                    tip_on_interface[j] = False
                else:
                    cj = np.sum(on_interface[None, :] * Nc, axis=1) / np.sum(
                        on_interface
                    )
            tip[[i, r, g], j] = cj
        return tip, tip_on_interface

    def cal_element_matrices(self, eval_mass=False):
        n = (
            self.N_DOFS
            + int(self.h_enrich) * self.H_DOFS
            + int(self.t_enrich) * self.TIP_DOFS
        )
        Ke = np.zeros((n, n))
        x_e = self.node_coords

        (rule, correction) = qd.TETR_RULES[1]
        D = self.cal_D()

        _, dN_dxi = self._base_shape_functions(rule[:, :-1].T)
        J = dN_dxi[0, :, :] @ x_e
        J_inv = np.linalg.inv(J)
        detJ = np.linalg.det(J)
        Nc = None
        if self.partial_cut:
            dN_dxy = J_inv @ dN_dxi
            B = cal_B_3d_vec(dN_dxy)
            w_eff = rule[:, 3] * correction * detJ
            Ke[: self.N_DOFS, : self.N_DOFS] = np.sum(
                (B.transpose(0, 2, 1) @ D @ B) * w_eff[:, None, None], axis=0
            )
            self._integrate_partial_cut(Ke, D, J_inv, detJ, B)
        elif self.h_enrich:
            Nc, kappa = self._cal_intersections()
            if self.t_enrich:
                rule, correction = qd.TRI_RULES[13]
            else:
                rule, correction = qd.TETR_RULES[2]
            for Ni, detJi in cut_embedding_tetr_iter(Nc, kappa):
                n, _ = self._base_shape_functions(rule[:, :3].T)
                sub_nat_coords = self.NAT_COORDS.T @ Ni @ n.T
                _, dN_dxi_sub = self.shape_functions(sub_nat_coords)
                # dN_dxy_sub = np.linalg.solve(J, dN_dxi_sub)
                dN_dxy_sub = J_inv @ dN_dxi_sub
                B = cal_B_3d_vec(dN_dxy_sub)
                w_eff = rule[:, 3] * detJi * correction * detJ
                Ke[:, :] += np.sum(
                    (B.transpose(0, 2, 1) @ D @ B) * w_eff[:, None, None], axis=0
                )
        else:
            (rule, correction) = qd.TETR_RULES[13]
            nat_coords = rule[:, :3]

            _, dN_dxi = self.shape_functions(nat_coords)
            # dN_dxy = np.linalg.solve(J, dN_dxi)
            dN_dxy = J_inv @ dN_dxi
            B = cal_B_3d_vec(dN_dxy)
            w_eff = rule[:, 3] * correction * detJ
            Ke[:, :] = np.sum(
                (B.transpose(0, 2, 1) @ D @ B) * w_eff[:, None, None], axis=0
            )

        if eval_mass:
            raise NotImplementedError
        else:
            return Ke

    def _integrate_partial_cut(self, Ke, D, J_inv, detJ, B):
        x_e = self.node_coords
        Nc, _ = self._cal_intersections()
        tip, tip_on_interface = self._cal_front_intersections()
        (rule, correction) = qd.UNIT_HEX_RULES[10]
        for Ni, detJi, n_on_interface in partial_cut_embedding_tetr_iter(
            Nc, tip, tip_on_interface
        ):
            if detJi < 0:
                print("DetJi smaller than 0")
            x_e_i = (x_e.T @ Ni).T
            duffy = DuffySinh3D(x_e_i)
            rule_d = duffy.transform(rule[:, :3].T, beta1=1, beta2=1)
            nat_coords_d = rule_d[:3]
            w_d = rule_d[3]
            n, _ = self._base_shape_functions(nat_coords_d)
            sub_nat_coords = self.NAT_COORDS.T @ Ni @ n.T
            _, dN_dxi_sub = self.shape_functions(sub_nat_coords)
            dN_dxy_sub = J_inv @ dN_dxi_sub[:, :, self.N_FN :]
            TIP_B = cal_B_3d_vec(dN_dxy_sub)
            w_eff = rule[:, 3] * w_d * detJi * correction * detJ
            begin_tip = self.N_DOFS
            Ke[begin_tip:, begin_tip:] += np.sum(
                (TIP_B.transpose(0, 2, 1) @ D @ TIP_B) * w_eff[:, None, None],
                axis=0,
            )

            rule_d = duffy.transform(rule[:, :3].T, beta1=2, beta2=n_on_interface)
            nat_coords_d = rule_d[:3]
            w_d = rule_d[3]
            n, _ = self._base_shape_functions(nat_coords_d)
            sub_nat_coords = self.NAT_COORDS.T @ Ni @ n.T
            _, dN_dxi_sub = self.shape_functions(sub_nat_coords)
            dN_dxy_sub = J_inv @ dN_dxi_sub[:, :, self.N_FN :]
            TIP_B = cal_B_3d_vec(dN_dxy_sub)
            w_eff = rule[:, 3] * w_d * detJi * correction * detJ
            begin_tip = self.N_DOFS
            res = np.sum(
                B.transpose(0, 2, 1) @ D @ TIP_B * w_eff[:, None, None], axis=0
            )
            Ke[0:begin_tip, begin_tip:] += res
            Ke[begin_tip:, 0:begin_tip] += res.T

    def shape_functions(self, natural_coordinate):
        return enriched_shape_functions(
            self, self._base_shape_functions, None, natural_coordinate
        )
