from .Tetr4n import Tetr4n
import numpy as np
from typing import Final


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
        xi = natural_coordinate[0, :]
        eta = natural_coordinate[1, :]
        zeta = natural_coordinate[2, :]
        N = np.array([1 - xi - eta - zeta, xi, eta, zeta]).T
        return N

    def _base_shape_function_derivatives(self, _):
        dN_dnat = np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return dN_dnat[None, :, :]

    def _cal_intersections(self):
        phi_n2 = np.empty(6)
        phi_n2[:3] = self.phi_n[1:]
        phi_n2[3:5] = self.phi_n[2:]
        phi_n2[-1] = self.phi_n[-1]
        phi_n1 = np.empty(6)
        phi_n1[:3] = self.phi_n[:-1]
        phi_n1[3:5] = self.phi_n[0]
        phi_n1[-1] = self.phi_n[1]
        num = phi_n2
        denom = phi_n2 - phi_n1
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
                [N1[0], 0, 0, N1[3], N1[4], 0],
                [1 - N1[0], N1[1], 0, 0, 0, N1[5]],
                [0, 1 - N1[1], N1[2], 1 - N1[3], 0, 0],
                [0, 0, 1 - N1[2], 0, 1 - N1[4], 1 - N1[5]],
            ]
        )
        phi_n12 = N1 * phi_n1 + (1 - N1) * phi_n2
        on_interface = np.isclose(phi_n12, 0.0)
        kappa = np.sum(on_interface[None, :] * Nc, axis=1) / np.sum(on_interface)
        return Nc, kappa

    def _cal_front_intersections(self):
        tip = np.empty((4, 4))
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
                else:
                    cj = np.sum(on_interface[None, :] * Nc, axis=1) / np.sum(
                        on_interface
                    )
            tip[[i, r, g], j] = cj
        return tip
