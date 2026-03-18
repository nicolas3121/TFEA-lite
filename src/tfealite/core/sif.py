import numpy as np

from ..core.dofs import BRANCH_DOFS, HEAVISIDE_DOFS
from ..elements.utils import contains_points, fill_element_displacement
from ..elements.XQuad4n import XQuad4n
from .level_set import CutType


class DisplacementCorrelationMethodSIF:
    def __init__(self, kosolov, shear_mod, r, dr):
        self.kosolov = kosolov
        self.shear_mod = shear_mod
        self.r = r
        self.dr = dr

    def cal_sif(self, level_set, model, cut_info: dict, u_tip: float):
        assert level_set.bspline is not None
        assert level_set.dbspline is not None
        assert level_set.ddbspline is not None

        tip = level_set.bspline(u_tip)
        t = level_set.dbspline(u_tip)

        n = np.array([-t[1], t[0]])
        if level_set.embedded and u_tip == 0.0:
            t *= -1

        t = t / np.linalg.norm(t)
        n = n / np.linalg.norm(n)

        p1 = tip[None, :] - self.r[:, None] * t[None, :]
        bspline = level_set.bspline
        dbspline = level_set.dbspline
        ddbspline = level_set.ddbspline

        u_i = np.full(p1.shape[0], u_tip)

        for _ in range(50):
            S = bspline(u_i)
            dS = dbspline(u_i)
            ddS = ddbspline(u_i)

            distance = S - p1
            f = np.sum(dS * distance, axis=1)
            df = np.sum(ddS * distance, axis=1) + np.sum(dS * dS, axis=1)

            du = f / df
            u_i_next = np.clip(u_i - du, 0.0, 1.0)

            if np.all(np.isclose(u_i, u_i_next, atol=1e-12)):
                break
            u_i = u_i_next
        else:
            print(
                "Warning: Orthogonal projection Newton iterations did not fully converge."
            )

        p1 = bspline(u_i)

        jump = np.empty_like(p1)
        r_1_star = np.empty(p1.shape[0])

        for elem_id, (ls_id, cut_type, tip_number) in cut_info.items():
            if cut_type == CutType.NONE:
                continue

            self._compute_element_jump(
                elem_id, ls_id, cut_type, tip_number, model, p1, tip, jump, r_1_star
            )

        T_matrix = np.array([t, n])
        jump = jump @ T_matrix.T

        if len(r_1_star) < 2:
            raise ValueError(
                "Richardson extrapolation requires at least 2 extraction points."
            )

        coef = (self.shear_mod / (self.kosolov + 1.0)) * np.sqrt(2.0 * np.pi / r_1_star)
        K_I_star = coef * jump[:, 1]
        K_II_star = coef * jump[:, 0]

        r_a, r_b = r_1_star[:-1], r_1_star[1:]

        extrap_multiplier = r_b / (r_b - r_a)
        r_ratio = r_a / r_b

        K_I_ext = extrap_multiplier * (K_I_star[:-1] - r_ratio * K_I_star[1:])
        K_II_ext = extrap_multiplier * (K_II_star[:-1] - r_ratio * K_II_star[1:])

        return np.mean(K_I_ext), np.mean(K_II_ext)

    def _compute_element_jump(
        self, elem_id, ls_id, cut_type, tip_number, model, p1, tip, jump_out, r_star_out
    ):
        element = model.elements[elem_id - 1]
        _, _, mat_id, real_id, elem_nodes = element
        elem_nodes = np.asarray(elem_nodes)
        elem_vertices = model.nodes[elem_nodes - 1, 1:3]

        contains_p1 = contains_points(elem_vertices, p1)
        if not np.any(contains_p1):
            return

        elem_dofs = model.list_dof.get_elem_dofs(elem_nodes)
        local_dofs_per_node = np.bitwise_or.reduce(elem_dofs)
        h_enrich = bool(local_dofs_per_node & HEAVISIDE_DOFS)
        t_enrich = bool(local_dofs_per_node & BRANCH_DOFS)

        phi_n, phi_t = model.level_sets[ls_id].get(elem_nodes, tip_number)
        Ue = fill_element_displacement(elem_nodes, model.list_dof, model.Ug).reshape(
            (-1, 2)
        )

        elem = XQuad4n(
            node_coords=elem_vertices,
            material=model.materials[mat_id - 1][1],
            real=model.reals[real_id - 1][1],
            phi_n=phi_n,
            phi_t=phi_t,
            h_enrich=h_enrich,
            t_enrich=t_enrich,
            partial_cut=(cut_type == CutType.PARTIAL),
            in_range=model.in_range[elem_nodes - 1],
        )

        nearest_p1 = [
            elem.nearest_point_on_crack(p)
            for p, inside in zip(p1, contains_p1)
            if inside
        ]

        if nearest_p1:
            xi_1, eta_1 = elem._cal_nat_coords(nearest_p1)
            jump_shape_fn_1, r_1s = elem.jump_shape_functions(xi_1, eta_1, tip)

            jump_out[contains_p1, :] = jump_shape_fn_1 @ Ue
            r_star_out[contains_p1] = r_1s
