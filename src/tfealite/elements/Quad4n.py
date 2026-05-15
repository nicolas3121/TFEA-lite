import numpy as np
from .utils import cal_B_2d_vec, cal_N_2d_vec
from ..core.quadratures import QUAD_RULES, BAR_RULES


class Quad4n:
    def __init__(self, node_coords, material, real):
        self.node_coords = node_coords
        self.E = material["E"]
        self.nu = material["nu"]
        self.rho = material["rho"] if ("rho" in material) else 0.0
        self.t = real["t"]

        c1 = self.E / (1.0 - self.nu**2)
        self.C = c1 * np.array(
            [
                [1.0, self.nu, 0.0],
                [self.nu, 1.0, 0.0],
                [0.0, 0.0, (1.0 - self.nu) / 2.0],
            ]
        )

    def cal_element_matrices(self, eval_mass=False):
        rule, correction = QUAD_RULES[3]
        Me = np.zeros((8, 8)) if eval_mass else None
        x_e = self.node_coords
        nat_coords = rule[:, :2].T
        N, dN_dxi = Quad4n.shape_functions(self, nat_coords)
        J = dN_dxi @ x_e
        detJ = np.linalg.det(J)
        dN_dxy = np.linalg.solve(J, dN_dxi)
        B = cal_B_2d_vec(dN_dxy)
        w_eff = rule[:, 2] * correction * detJ
        Ke = np.sum((B.transpose(0, 2, 1) @ self.C @ B) * w_eff[:, None, None], axis=0)
        if eval_mass:
            N_2d = cal_N_2d_vec(N)
            Me = np.sum((N_2d.transpose(0, 2, 1) @ N_2d) * w_eff[:, None, None], axis=0)
            return Ke, Me
        return Ke, None

    @staticmethod
    def cal_traction_loads(
        node_coords, node_on_boundary, traction_expression, real, deg
    ):
        def shape_functions(xi, eta):
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
            return N

        Fe = np.zeros(8)
        rule, correction = BAR_RULES[deg]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        selector = [0, 1, 0, 1]
        nat_edge = [-1, 1, 1, -1]
        nat_2 = np.empty_like(rule[:, 0])
        for (n1, n2), s, n in zip(edges, selector, nat_edge):
            if node_on_boundary[n1] and node_on_boundary[n2]:
                nat_2.fill(n)
                if s == 0:
                    xi = rule[:, 0]
                    eta = nat_2
                else:
                    eta = rule[:, 0]
                    xi = nat_2
                N = shape_functions(xi, eta)
                coordinates = N[:, [n1, n2]] @ node_coords[[n1, n2], :]
                t_x, t_y, _ = traction_expression(
                    coordinates[:, 0], coordinates[:, 1], 0
                )
                L = np.linalg.norm(node_coords[n1, :] - node_coords[n2, :])
                w_eff = rule[:, 1] * L * real["t"] * correction / 2
                Fe[::2] += np.sum(
                    N * np.atleast_1d(t_x)[:, None] * w_eff[:, None], axis=0
                )
                Fe[1::2] += np.sum(
                    N * np.atleast_1d(t_y)[:, None] * w_eff[:, None], axis=0
                )
        return Fe

    def shape_functions(self, nat_coords):
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

    def cal_stresses(self, nat_coords, Ue):
        Ue = np.asarray(Ue, dtype=float).ravel()
        _, dN_dxi = self.shape_functions(nat_coords)
        J = dN_dxi @ self.node_coords
        dN_dxy = np.linalg.solve(J, dN_dxi)
        B = cal_B_2d_vec(dN_dxy)
        eps = B @ Ue
        sig = self.C @ eps[:, :, None]
        return sig.reshape(-1, 3)

    def stresses_at_nodes(self, Ue):
        xi = np.array([-1.0, 1.0, 1.0, -1.0])
        eta = np.array([-1.0, -1.0, 1.0, 1.0])
        return self.cal_stresses(np.stack([xi, eta], axis=0), Ue)
