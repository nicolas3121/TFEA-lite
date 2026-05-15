import numpy as np
from ..core import quadratures as qd
from .utils import (
    cal_B_3d_vec,
)


class Tetr4n:
    def __init__(self, node_coords, material, _):
        self.node_coords = np.asarray(node_coords, dtype=float).reshape(4, 3)
        self.material = material

    # def shape_functions(self, natural_coordinate):
    #     xi, eta, zeta = natural_coordinate
    #     N = np.zeros(4)
    #     N[0] = 1.0 - xi - eta - zeta
    #     N[1] = xi
    #     N[2] = eta
    #     N[3] = zeta
    #     return N

    def shape_functions(self, natural_coordinate):
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

    def shape_function_derivatives(self, natural_coordinate):
        dN_dnat = np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return dN_dnat

    def jacobian_matrix(self):
        x1, x2, x3, x4 = self.node_coords
        J = np.column_stack((x2 - x1, x3 - x1, x4 - x1))
        return J

    def element_volume(self):
        J = self.jacobian_matrix()
        V = np.linalg.det(J) / 6.0
        if V <= 0:
            print(f"Negative/zero detJ ({V:.3e}): inverted or degenerate tet.")
        return abs(V)

    def strain_displacement_matrix(self):
        dN_dnat = self.shape_function_derivatives(None)
        J = self.jacobian_matrix()
        invJ = np.linalg.inv(J)
        grads = dN_dnat @ invJ
        B = np.zeros((6, 12), dtype=float)
        for i in range(4):
            dNdx, dNdy, dNdz = grads[i, :]
            c = 3 * i
            B[0, c + 0] = dNdx
            B[1, c + 1] = dNdy
            B[2, c + 2] = dNdz
            B[3, c + 0] = dNdy
            B[3, c + 1] = dNdx
            B[4, c + 0] = dNdz
            B[4, c + 2] = dNdx
            B[5, c + 1] = dNdz
            B[5, c + 2] = dNdy
        return B

    def cal_D(self, E=None, nu=None):
        if E is None:
            E = self.material["E"]
        if nu is None:
            nu = self.material["nu"]
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        D = np.array(
            [
                [lam + 2 * mu, lam, lam, 0, 0, 0],
                [lam, lam + 2 * mu, lam, 0, 0, 0],
                [lam, lam, lam + 2 * mu, 0, 0, 0],
                [0, 0, 0, mu, 0, 0],
                [0, 0, 0, 0, mu, 0],
                [0, 0, 0, 0, 0, mu],
            ],
            dtype=float,
        )
        return D

    def cal_element_matrices(self, eval_mass=False):
        x_e = self.node_coords
        (rule, correction) = qd.TETR_RULES[1]
        nat_coords = rule[:, :3]
        _, dN_dxi = self.shape_functions(nat_coords.T)
        J = dN_dxi[0, :, :] @ x_e
        detJ = np.linalg.det(J)
        dN_dxy = np.linalg.solve(J, dN_dxi)
        # B = cal_B_3d_vec(dN_dxy)
        # V = self.element_volume()
        # B = self.strain_displacement_matrix()
        D = self.cal_D(self.material["E"], self.material["nu"])
        B = cal_B_3d_vec(dN_dxy)
        w_eff = rule[:, 3] * correction * detJ
        Ke = np.sum((B.transpose(0, 2, 1) @ D @ B) * w_eff[:, None, None], axis=0)
        # Ke = B.T @ D @ B * V
        if not eval_mass:
            return Ke, None
        rho = self.material["rho"]
        M_scalar = (rho * detJ / 20.0) * (
            2.0 * np.eye(4) + (np.ones((4, 4)) - np.eye(4))
        )
        Me = np.kron(M_scalar, np.eye(3))
        return Me, Ke

    def cal_element_stress(self, Ue):
        Ue = np.asarray(Ue, dtype=float)
        if Ue.ndim == 2:
            Ue = Ue.reshape(
                12,
            )
        elif Ue.ndim == 1 and Ue.size == 12:
            pass
        else:
            raise ValueError(f"Ue must be (4,3) or (12,), got shape {Ue.shape}")
        B = self.strain_displacement_matrix()
        D = self.cal_D(self.material["E"], self.material["nu"])
        strain = B @ Ue
        stress = D @ strain
        return stress

    def stresses_at_nodes(self, Ue):
        Ue = np.asarray(Ue, dtype=float)
        if Ue.ndim == 2:
            Ue = Ue.reshape(
                12,
            )
        elif Ue.ndim == 1 and Ue.size == 12:
            pass
        else:
            raise ValueError(f"Ue must be (4,3) or (12,), got shape {Ue.shape}")
        B = self.strain_displacement_matrix()
        D = self.cal_D(self.material["E"], self.material["nu"])
        strain = B @ Ue
        stress = D @ strain
        return stress

    def cal_stresses(self, nat_coords, Ue):
        Ue = np.asanyarray(Ue, dtype=float).ravel()
        x_e = self.node_coords
        _, dN_dxi = self.shape_functions(nat_coords)
        J = dN_dxi[0, :, :] @ x_e
        dN_dxy = np.linalg.solve(J, dN_dxi)

        B = cal_B_3d_vec(dN_dxy)
        D = self.cal_D()
        eps = B @ Ue
        sig = D @ eps[:, :, None]
        return sig
