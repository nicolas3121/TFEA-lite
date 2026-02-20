from .Quad4n import Quad4n
import numpy as np
from typing import Final
from .utils import branch_functions

NODES: Final = 4
DOFS: Final = 2
BRANCH_FN: Final = 4  # branch functions
N_FN: Final = NODES
H_FN: Final = NODES
TIP_FN: Final = NODES * BRANCH_FN
N_DOFS: Final = DOFS * N_FN
H_DOFS: Final = DOFS * H_FN
TIP_DOFS: Final = DOFS * TIP_FN


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
    ):
        if not h_enrich and not t_enrich:
            print("creating basic element instead")
            return Quad4n(node_coords, material, real)
        assert h_enrich is not None
        assert t_enrich is not None
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
    ):
        super().__init__(node_coords, material, real)
        self.phi_n = phi_n
        self.phi_t = phi_t
        self.h_enrich = h_enrich
        self.t_enrich = t_enrich
        self.partial_cut = partial_cut

    def cal_element_matrices(self, eval_mass=False):
        raise NotImplementedError

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
