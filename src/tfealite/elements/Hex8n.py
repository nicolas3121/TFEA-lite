# import numpy as np
# from ..core.quadratures import HEX_RULES
# from .utils import cal_B_3d_vec
#
#
# class Hex8n:
#     def __init__(self, node_coords, material, _):
#         self.node_coords = np.asarray(node_coords, dtype=float).reshape(4, 3)
#         self.material = material
#         self.rho = material["rho"]
#
#         E = material["E"]
#         nu = material["nu"]
#         lam = E * nu / ((1 + nu) * (1 - 2 * nu))
#         mu = E / (2 * (1 + nu))
#
#         self.C = np.array(
#             [
#                 [lam + 2 * mu, lam, lam, 0, 0, 0],
#                 [lam, lam + 2 * mu, lam, 0, 0, 0],
#                 [lam, lam, lam + 2 * mu, 0, 0, 0],
#                 [0, 0, 0, mu, 0, 0],
#                 [0, 0, 0, 0, mu, 0],
#                 [0, 0, 0, 0, 0, mu],
#             ],
#             dtype=float,
#         )
#
#     def cal_element_matrices(self, eval_mass=False):
#         rule, correction = HEX_RULES[3]
#         Me = np.zeros((24, 24)) if eval_mass else None
#         x_e = self.node_coords
#
#         nat_coords = rule[:, :3].T
#
#         N, dN_dxi = self.shape_functions(nat_coords)
#         J = dN_dxi @ x_e
#         detJ = np.linalg.det(J)
#         dN_dxyz = np.linalg.solve(J, dN_dxi)
#
#         B = cal_B_3d_vec(dN_dxyz)
#
#         w_eff = rule[:, 3] * correction * detJ
#
#         Ke = np.sum((B.transpose(0, 2, 1) @ self.C @ B) * w_eff[:, None, None], axis=0)
#
#         if eval_mass:
#             rho_vol = self.rho
#             N_3d = np.zeros((rule.shape[0], 3, 24))
#             N_3d[:, 0, 0::3] = N[:, :]
#             N_3d[:, 1, 1::3] = N[:, :]
#             N_3d[:, 2, 2::3] = N[:, :]
#             Me = np.sum(
#                 rho_vol * (N_3d.transpose(0, 2, 1) @ N_3d) * w_eff[:, None, None],
#                 axis=0,
#             )
#             return Me, Ke
#         return Ke
#
#     def shape_functions(self, nat_coords):
#         nat_coords = np.atleast_2d(nat_coords)
#         xi = nat_coords[0]
#         eta = nat_coords[1]
#         zeta = nat_coords[2]
#
#         xi_m, xi_p = 1.0 - xi, 1.0 + xi
#         eta_m, eta_p = 1.0 - eta, 1.0 + eta
#         zeta_m, zeta_p = 1.0 - zeta, 1.0 + zeta
#
#         N = (
#             0.125
#             * np.array(
#                 [
#                     xi_m * eta_m * zeta_m,
#                     xi_p * eta_m * zeta_m,
#                     xi_p * eta_p * zeta_m,
#                     xi_m * eta_p * zeta_m,
#                     xi_m * eta_m * zeta_p,
#                     xi_p * eta_m * zeta_p,
#                     xi_p * eta_p * zeta_p,
#                     xi_m * eta_p * zeta_p,
#                 ]
#             ).T
#         )
#
#         dN_dxi = 0.125 * np.array(
#             [
#                 [
#                     -eta_m * zeta_m,
#                     eta_m * zeta_m,
#                     eta_p * zeta_m,
#                     -eta_p * zeta_m,
#                     -eta_m * zeta_p,
#                     eta_m * zeta_p,
#                     eta_p * zeta_p,
#                     -eta_p * zeta_p,
#                 ],
#                 [
#                     -xi_m * zeta_m,
#                     -xi_p * zeta_m,
#                     xi_p * zeta_m,
#                     xi_m * zeta_m,
#                     -xi_m * zeta_p,
#                     -xi_p * zeta_p,
#                     xi_p * zeta_p,
#                     xi_m * zeta_p,
#                 ],
#                 [
#                     -xi_m * eta_m,
#                     -xi_p * eta_m,
#                     -xi_p * eta_p,
#                     -xi_m * eta_p,
#                     xi_m * eta_m,
#                     xi_p * eta_m,
#                     xi_p * eta_p,
#                     xi_m * eta_p,
#                 ],
#             ]
#         )
#
#         return N, dN_dxi.transpose(2, 0, 1)
#
#     def cal_stresses(self, nat_coords, Ue):
#         Ue = np.asarray(Ue, dtype=float).ravel()
#         _, dN_dxi = self.shape_functions(nat_coords)
#         J = dN_dxi @ self.node_coords
#         dN_dxyz = np.linalg.solve(J, dN_dxi)
#         B = cal_B_3d_vec(dN_dxyz)
#
#         eps = B @ Ue
#         sig = self.C @ eps[:, :, None]
#         return sig.reshape(-1, 6)
#
#     def stresses_at_nodes(self, Ue):
#         xi = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0])
#         eta = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
#         zeta = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
#         return self.cal_stresses(np.stack([xi, eta, zeta], axis=0), Ue)
