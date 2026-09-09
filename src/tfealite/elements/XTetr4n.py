from typing import Final

import numpy as np

from ..core import quadratures as qd
from ..core.quadratures import DuffySinh3D
from .Tetr4n import Tetr4n
from .utils import (
    ELEM_EDGES,
    cal_B_3d_vec,
    cut_embedding_tetr_iter,
    enriched_shape_functions,
    jump_shape_functions,
    partial_cut_embedding_tetr_iter,
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
        in_range=None,
    ):
        if in_range is None:
            in_range = (np.ones(4, dtype=bool),)
        super().__init__(node_coords, material, real)
        self.phi_n = phi_n
        self.phi_t = phi_t
        self.h_enrich = h_enrich
        self.t_enrich = t_enrich
        self.partial_cut = partial_cut
        self.in_range = in_range

    def _base_shape_functions(self, natural_coordinate):
        # natural_coordinate = np.atleast_2d(natural_coordinate)
        # print(natural_coordinate)
        xi = np.atleast_1d(natural_coordinate[0])
        eta = np.atleast_1d(natural_coordinate[1])
        zeta = np.atleast_1d(natural_coordinate[2])
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
        tol = 1e-10
        phi_num = self.phi_n[self.num_edges]
        phi_denom = self.phi_n[self.denom_edges]
        num = phi_num
        denom = phi_num - phi_denom
        unsolvable = np.isclose(denom, 0, atol=tol)
        on_crack = np.isclose(phi_denom, 0, atol=tol)
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
        phi_n12 = N1 * phi_denom + (1 - N1) * phi_num
        on_interface = np.isclose(phi_n12, 0.0, atol=tol)
        if not np.any(on_interface):
            kappa = None
        else:
            kappa = np.sum(on_interface[None, :] * Nc, axis=1) / np.sum(on_interface)
        return Nc, kappa

    def _cal_front_intersections(self):
        # print("nodes", self.node_coords)
        tol = 1e-10
        tip = np.empty((4, 4))
        tip_on_interface = [True, True, True, True]
        B = np.array([0, 0, 1])
        for j in range(4):
            tip[j, j] = 0
            i, r, g = (j + 1) % 4, (j + 2) % 4, (j + 3) % 4
            # print("face", i, r, g)
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
                unsolvable = np.isclose(denom, 0.0, atol=tol)
                on_crack = np.isclose(phi_n_face, 0.0, atol=tol)
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
                on_interface = np.isclose(phi_n12, 0.0, atol=tol)
                if not np.any(on_interface):
                    cj = np.full(3, 1 / 3)
                    tip_on_interface[j] = False
                else:
                    cj = np.sum(on_interface[None, :] * Nc, axis=1) / np.sum(
                        on_interface
                    )
                    phi_t_cj = np.dot(cj, phi_t_face)

                    # If phi_t is also 0 here, this is a real front point!
                    if np.isclose(phi_t_cj, 0.0, atol=tol):
                        tip_on_interface[j] = True
                    else:
                        tip_on_interface[j] = False
            tip[[i, r, g], j] = cj
        return tip, tip_on_interface

    def cal_element_matrices(self, eval_mass=False):
        n = (
            self.N_DOFS
            + int(self.h_enrich) * self.H_DOFS
            + int(self.t_enrich) * self.TIP_DOFS
        )
        Ke = np.zeros((n, n))
        np.zeros_like(Ke)
        x_e = self.node_coords

        (rule, correction) = qd.TETR_RULES[1]
        D = self.cal_D()

        _, dN_dxi = self._base_shape_functions(rule[:, :-1].T)
        J = dN_dxi[0, :, :] @ x_e
        try:
            J_inv = np.linalg.inv(J)
        except np.linalg.LinAlgError:
            print(x_e)
            raise ValueError
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
                rule, correction = qd.TETR_RULES[13]
            else:
                rule, correction = qd.TETR_RULES[2]
            # total_weight = 0
            for Ni, detJi in cut_embedding_tetr_iter(Nc, kappa):
                # total_weight += detJi
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
            # print("full cut", total_weight)
        else:
            (rule, correction) = qd.TETR_RULES[13]
            nat_coords = rule[:, :3]

            _, dN_dxi = self.shape_functions(nat_coords.T)
            dN_dxy = J_inv @ dN_dxi
            B = cal_B_3d_vec(dN_dxy)
            w_eff = rule[:, 3] * correction * detJ
            Ke[:, :] = np.sum(
                (B.transpose(0, 2, 1) @ D @ B) * w_eff[:, None, None], axis=0
            )

        if eval_mass:
            raise NotImplementedError
        else:
            return Ke, None

    def _integrate_partial_cut(self, Ke, D, J_inv, detJ, B):
        x_e = self.node_coords
        Nc, _ = self._cal_intersections()
        tip, tip_on_interface = self._cal_front_intersections()
        (rule, correction) = qd.UNIT_HEX_RULES[10]
        # total_weight = 0
        for Ni, detJi, n_on_interface in partial_cut_embedding_tetr_iter(
            Nc, tip, tip_on_interface
        ):
            if detJi < 0:
                print("DetJi smaller than 0")
            # total_weight += detJi
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

            rule_d = duffy.transform(
                rule[:, :3].T, beta1=2, beta2=min(n_on_interface, 2)
            )
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
        # print("partial_cut", total_weight)
        # print("w_eff tot split", w_eff_tot)

    def shape_functions(self, natural_coordinate):
        return enriched_shape_functions(
            self, self._base_shape_functions, None, natural_coordinate
        )

    def cal_stresses(self, nat_coords, Ue):
        Ue = np.asanyarray(Ue, dtype=float).ravel()
        x_e = self.node_coords
        _, dN_dxi = self.shape_functions(nat_coords)
        J = dN_dxi[0, :, : self.N_FN] @ x_e
        dN_dxy = np.linalg.solve(J, dN_dxi)

        B = cal_B_3d_vec(dN_dxy)
        D = self.cal_D()
        eps = B @ Ue
        sig = D @ eps[:, :, None]
        return sig

    # niet helemaal ok want ligt niet gegarandeerd in vlak loodrecht op crack front
    # moet eigenlijk crack front vectoren gebruiken om te constrainen tot vlak
    # def nearest_point_on_crack(self, coords):
    #     coords_2d = np.atleast_2d(coords)
    #     print("hello")
    #     print("coords", coords_2d)
    #     N_pts = coords_2d.shape[0]
    #
    #     Nc, kappa = self._cal_intersections()
    #     phi_n = self.phi_n @ Nc
    #
    #     on_crack = np.isclose(phi_n, 0.0, atol=1e-12)
    #     crack_indices = np.where(on_crack)[0]
    #
    #     if len(crack_indices) == 0:
    #         return None, None
    #
    #     # Extract both physical and natural coordinates of the intersection points
    #     crack_pts = (self.node_coords.T @ Nc)[:, crack_indices].T
    #     crack_nat = (self.NAT_COORDS.T @ Nc[:, crack_indices]).T
    #
    #     # --- 0D Case: Touching Node ---
    #     if len(crack_indices) == 1 or np.all(
    #         np.isclose(crack_pts[1:], crack_pts[0], atol=1e-12)
    #     ):
    #         return (
    #             np.tile(crack_pts[0], (N_pts, 1)),
    #             np.tile(crack_nat[0], (N_pts, 1)),
    #         )
    #
    #     center_phys = np.mean(crack_pts, axis=0)
    #
    #     centered_phys = crack_pts - center_phys
    #
    #     U, S, Vt = np.linalg.svd(centered_phys)
    #
    #     tol = 1e-10
    #     valid_dims = S > tol
    #     basis_vectors = Vt[valid_dims]
    #
    #     W = coords_2d - center_phys
    #     W_proj = (W @ basis_vectors.T) @ basis_vectors
    #     closest_pts = center_phys + W_proj
    #
    #     J = self.jacobian_matrix()
    #     X0 = self.node_coords[0]
    #     print("closest_pts", closest_pts)
    #     dx = closest_pts - X0[None, :]
    #     nat_coords = np.linalg.solve(J, dx.T)
    #     print("nat_coords", nat_coords)
    #
    #     return closest_pts, nat_coords

    def nearest_point_on_crack(self, coords, tip_pos, tip_b):
        Nc, _kappa = self._cal_intersections()
        phi_n_at_intersections = self.phi_n @ Nc
        on_crack = np.isclose(phi_n_at_intersections, 0.0, atol=1e-12)

        if not np.any(on_crack):
            return None, None

        crack_pts = (self.node_coords.T @ Nc)[:, on_crack].T
        X_anchor = np.mean(crack_pts, axis=0)

        _, dN_dxi_full = self._base_shape_functions(np.array([1, 0, 0]))
        dN_dxi = dN_dxi_full[0, :, : self.N_FN]
        J = dN_dxi @ self.node_coords
        J_inv = np.linalg.inv(J)

        grad_phi_n = J_inv @ dN_dxi @ self.phi_n
        n_c = grad_phi_n / np.linalg.norm(grad_phi_n)

        n_s = tip_b / np.linalg.norm(tip_b, axis=1)[:, None]

        # 4. Vectorized intersection of two planes (Crack Plane & Slice Plane)
        dot = np.sum(n_c[None, :] * n_s, axis=1)  # (N,)
        det = 1.0 - dot**2

        # Residuals relative to our anchor points
        # r_c: distance from query to the crack plane
        r_c = np.sum((X_anchor - coords) * n_c[None, :], axis=1)
        # r_s: distance from query to the slice plane
        r_s = np.sum((tip_pos - coords) * n_s, axis=1)

        alpha = (r_c - dot * r_s) / det
        beta = (r_s - dot * r_c) / det

        P_on_line = coords + alpha[:, None] * n_c[None, :] + beta[:, None] * n_s

        # 5. Final Mapping to Natural Coordinates
        # If J = dN_dxi @ self.node_coords, then xi = dx @ J_inv
        dx = P_on_line - self.node_coords[0]
        nat_coords = dx @ J_inv

        v = np.cross(n_c[None, :], n_s, axis=1)
        v /= np.linalg.norm(v, axis=1)[:, None]

        v_nat = v @ J_inv

        xi, eta, zeta = nat_coords[:, 0], nat_coords[:, 1], nat_coords[:, 2]
        dxi, deta, dzeta = v_nat[:, 0], v_nat[:, 1], v_nat[:, 2]

        A_mat = np.array([-dxi, -deta, -dzeta, dxi + deta + dzeta])

        B_mat = np.array([xi, eta, zeta, 1.0 - (xi + eta + zeta)])
        # first 3 >= 0, last one <= 1
        # one of them being one implies another being 0 so can use same equations for both

        eps = 1e-14

        # t_max: Upper bounds (where A > 0) -> gives most restrictive max
        t_max_candidates = np.where(A_mat > eps, B_mat / A_mat, np.inf)
        t_max = np.min(t_max_candidates, axis=0)

        # t_min: Lower bounds (where A < 0) -> gives most restrictive min
        t_min_candidates = np.where(A_mat < -eps, B_mat / A_mat, -np.inf)
        t_min = np.max(t_min_candidates, axis=0)

        # Validity Check: Does the line intersect the element at all?
        is_valid = t_min <= (t_max + 1e-10)

        # The unconstrained projection is exactly at lambda = 0.
        # We clamp 0 into the valid intersection segment [t_min, t_max].
        lambda_clamped = np.clip(0.0, t_min, t_max)

        # Apply the clamping translation
        P_clamped = P_on_line + lambda_clamped[:, None] * v
        nat_coords_clamped = nat_coords + lambda_clamped[:, None] * v_nat

        nat_coords_clamped = np.clip(nat_coords_clamped, 0.0, 1.0)

        return P_clamped, nat_coords_clamped, is_valid

    def jump_shape_functions(self, nat_coords, tip_coords):
        return jump_shape_functions(
            self,
            self._base_shape_functions,
            self._base_shape_functions,
            nat_coords,
            tip_coords,
        )

    # def project_on_crack_front(self, coords):
    #     tip, tip_on_interface = self._cal_front_intersections()
    #     tip = tip[:, tip_on_interface]
    #     phi_t = self.phi_t @ tip
    #     tip_on_front = np.isclose(phi_t, 0.0, atol=1e-12)
    #     front = tip[:, tip_on_front]
    #
    #     x_e_i = (self.node_coords.T @ front).T
    #     coords = np.atleast_2d(coords)
    #
    #     if x_e_i.shape[0] == 1:
    #         return np.tile(x_e_i[0], (coords.shape[0], 1))
    #
    #     elif x_e_i.shape[0] == 0:
    #         return coords
    #
    #     A = x_e_i[0]
    #     B = x_e_i[1]
    #
    #     AB = B - A
    #     AP = coords - A
    #
    #     t = np.dot(AP, AB) / (np.dot(AB, AB) + 1e-16)
    #
    #     projected_coords = A + t[:, None] * AB
    #
    #     if tip.shape[0] == 3:  # crack full cuts a face
    #         C = tip[~tip_on_front]
    #         tip_t = A - C
    #
    #     elif tip.shape[0] == 2:  # cuts through an edge
    #         pass
    #     else:
    #         pass
    #
    #     return projected_coords
