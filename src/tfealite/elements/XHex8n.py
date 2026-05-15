# from .Hex8n import Hex8n
# import numpy as np
# from typing import Final
# from .utils import (
#     ELEM_EDGES,
# )
#
#
# class XHex8n(Hex8n):
#     NODES: Final = 8
#     DOFS: Final = 3
#     BRANCH_FN: Final = 4
#     N_FN: Final = NODES
#     H_FN: Final = NODES
#     LH_FN: Final = 2 * NODES
#     TIP_FN: Final = NODES * BRANCH_FN
#     N_DOFS: Final = DOFS * N_FN
#     H_DOFS: Final = DOFS * H_FN
#     TIP_DOFS: Final = DOFS * TIP_FN
#     NAT_COORDS = np.array(
#         [
#             [-1.0, -1.0, -1.0],
#             [1.0, -1.0, -1.0],
#             [1.0, 1.0, -1.0],
#             [-1.0, 1.0, -1.0],
#             [-1.0, -1.0, 1.0],
#             [1.0, -1.0, 1.0],
#             [1.0, 1.0, 1.0],
#             [-1.0, 1.0, 1.0],
#         ],
#         dtype=float,
#     )
#     NAT_TETS = np.array(
#         [
#             # Tet 1: Nodes (0, 1, 2, 6)
#             [
#                 [-1.0, -1.0, -1.0],
#                 [1.0, -1.0, -1.0],
#                 [1.0, 1.0, -1.0],
#                 [1.0, 1.0, 1.0],
#             ],
#             # Tet 2: Nodes (0, 2, 3, 6)
#             [
#                 [-1.0, -1.0, -1.0],
#                 [1.0, 1.0, -1.0],
#                 [-1.0, 1.0, -1.0],
#                 [1.0, 1.0, 1.0],
#             ],
#             # Tet 3: Nodes (0, 3, 7, 6)
#             [
#                 [-1.0, -1.0, -1.0],
#                 [-1.0, 1.0, -1.0],
#                 [-1.0, 1.0, 1.0],
#                 [1.0, 1.0, 1.0],
#             ],
#             # Tet 4: Nodes (0, 7, 4, 6)
#             [
#                 [-1.0, -1.0, -1.0],
#                 [-1.0, 1.0, 1.0],
#                 [-1.0, -1.0, 1.0],
#                 [1.0, 1.0, 1.0],
#             ],
#             # Tet 5: Nodes (0, 4, 5, 6)
#             [
#                 [-1.0, -1.0, -1.0],
#                 [-1.0, -1.0, 1.0],
#                 [1.0, -1.0, 1.0],
#                 [1.0, 1.0, 1.0],
#             ],
#             # Tet 6: Nodes (0, 5, 1, 6)
#             [
#                 [-1.0, -1.0, -1.0],
#                 [1.0, -1.0, 1.0],
#                 [1.0, -1.0, -1.0],
#                 [1.0, 1.0, 1.0],
#             ],
#         ],
#         dtype=float,
#     )
#     DIAG_FACE_NODES = np.array(
#         [
#             # Face 1 (Bottom): Diagonal 0 -> 2
#             [0, 1, 2, 3],
#             # Face 2 (Top): Diagonal 4 -> 6
#             [4, 5, 6, 7],
#             # Face 3 (Front): Diagonal 0 -> 5
#             [0, 1, 5, 4],
#             # Face 4 (Right): Diagonal 1 -> 6
#             [1, 2, 6, 5],
#             # Face 5 (Back): Diagonal 3 -> 6
#             [3, 2, 6, 7],
#             # Face 6 (Left): Diagonal 0 -> 7
#             [0, 3, 7, 4],
#         ],
#         dtype=int,
#     )
#     num_edges, denom_edges = ELEM_EDGES["Hex8n"]
#
#     def __new__(
#         cls,
#         node_coords,
#         material,
#         real,
#         phi_n=None,
#         phi_t=None,
#         h_enrich: bool = False,
#         t_enrich: bool = False,
#         partial_cut: bool = False,
#         in_range=None,
#     ):
#         if not h_enrich and not t_enrich:
#             # print("creating basic element instead")
#             return Hex8n(node_coords, material, real)
#         assert h_enrich is not None
#         assert t_enrich is not None
#         assert phi_n is not None
#         assert phi_t is not None
#         assert partial_cut is not None
#         return super().__new__(cls)
#
#     def __init__(
#         self,
#         node_coords,
#         material,
#         real,
#         phi_n,
#         phi_t,
#         h_enrich: bool,
#         t_enrich: bool,
#         partial_cut: bool,
#         in_range=np.ones(4, dtype=bool),
#     ):
#         super().__init__(node_coords, material, real)
#         self.phi_n = phi_n
#         self.phi_t = phi_t
#         self.h_enrich = h_enrich
#         self.t_enrich = t_enrich
#         self.partial_cut = partial_cut
#         self.in_range = in_range
#
#     def _cal_intersections(self):
#         phi_num = self.phi_n[self.num_edges]
#         phi_denom = self.phi_n[self.denom_edges]
#         num = phi_num
#         denom = phi_num - phi_denom
#         unsolvable = np.isclose(denom, 0)
#         on_crack = np.isclose(phi_denom, 0)
#         N1 = np.clip(
#             np.divide(
#                 num,
#                 denom,
#                 out=np.ones_like(num, dtype=float),
#                 where=~unsolvable & ~on_crack,
#             ),
#             0,
#             1,
#         )
#
#         phi1 = self.phi_n[self.DIAG_FACE_NODES[:, 0]]  # Diagonal Start
#         phi2 = self.phi_n[self.DIAG_FACE_NODES[:, 1]]  # Adjacent 1
#         phi3 = self.phi_n[self.DIAG_FACE_NODES[:, 2]]  # Diagonal End
#         phi4 = self.phi_n[self.DIAG_FACE_NODES[:, 3]]  # Adjacent 2
#
#         # Quadratic coefficients (As^2 + Bs + C = 0)
#         A = phi1 - phi2 + phi3 - phi4
#         B = -2.0 * phi1 + phi2 + phi4
#         C = phi1
#
#         # Standard linear fallback data
#         num_diag = phi1
#         denom_diag = phi1 - phi3
#         unsolvable_diag = np.isclose(denom_diag, 0)
#
#         s_linear = np.divide(
#             num_diag,
#             denom_diag,
#             out=np.ones_like(phi1, dtype=float),
#             where=~unsolvable_diag,
#         )
#
#         discriminant = B**2 - 4 * A * C
#
#         # If A is ~0, it's a flat plane (linear). If discriminant < 0, no real roots.
#         use_linear = np.isclose(A, 0) | (discriminant < 0)
#
#         safe_A = np.where(use_linear, 1.0, A)
#         sqrt_disc = np.sqrt(np.maximum(discriminant, 0))
#
#         root1 = (-B + sqrt_disc) / (2.0 * safe_A)
#         root2 = (-B - sqrt_disc) / (2.0 * safe_A)
#
#         valid_root1 = (root1 >= 0.0) & (root1 <= 1.0)
#         valid_root2 = (root2 >= 0.0) & (root2 <= 1.0)
#
#         s_quad = np.where(valid_root1, root1, np.where(valid_root2, root2, s_linear))
#
#         N1_diag = np.where(use_linear, s_linear, s_quad)
#
#         N1_diag = np.where(unsolvable_diag | np.isclose(phi3, 0), 1.0, N1_diag)
#         N1_diag = np.clip(N1_diag, 0, 1)
