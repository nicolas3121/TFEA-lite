from enum import Enum, auto
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.interpolate import BSpline, splprep
from scipy.spatial import KDTree
from ..elements.utils import ELEM_EDGES


class CutType(Enum):
    NONE = auto()
    CUT = auto()
    PARTIAL = auto()


class LevelSet:
    def __init__(self):
        self.embedded = None
        self.phi_n = None
        self.phi_t = None
        self.phi_t2 = None
        self.phi_t_list = []
        self.partial_cut_elem_list = []
        self.mesh_tree = None
        self.bspline = None
        self.ndbsplines = None

    def gen_from_line_segment(self, nodes, p1, p2, embedded=False):
        coordinates = np.array(nodes, dtype=float)[:, 1:3]
        v = p2 - p1
        n = np.array([-v[1], v[0]])
        n = n / np.linalg.norm(n)
        phi_n = np.sum((coordinates - p1) * n, axis=1)

        t2 = v / np.linalg.norm(v)
        t1 = -t2

        phi_t1 = np.sum((coordinates - p2) * t2, axis=1)
        phi_t2 = np.sum((coordinates - p1) * t1, axis=1)

        pts = np.linspace(p1, p2, 4).T
        tck, u = splprep(pts, s=0, k=3)
        bspline = BSpline(tck[0], np.transpose(tck[1]), tck[2])
        self.bspline = bspline

        self.phi_n = phi_n
        self.phi_t_list.append(phi_t1)
        if embedded:
            self.phi_t_list.append(phi_t2)

    def gen_from_plane(self, nodes, p1, p2, p3, embedded=False):
        coordinates = np.asarray(nodes, dtype=float)[:, 1:4]
        v1 = p3 - p2
        v1 /= np.linalg.norm(v1)
        v2 = p2 - p1
        v2 -= np.dot(v1, v2) * v1
        v2 /= np.linalg.norm(v2)
        v3 = np.cross(v1, v2)
        t1 = v2
        t2 = -v2
        n = v3
        phi_n = np.sum((coordinates - p1) * n, axis=1)
        phi_t1 = np.sum((coordinates - p2) * t1, axis=1)
        phi_t2 = np.sum((coordinates - p1) * t2, axis=1)
        self.phi_n = phi_n
        self.phi_t = phi_t1
        self.phi_t_list.append(phi_t1)
        self.embedded = embedded
        if embedded:
            self.phi_t_list.append(phi_t2)

    def gen_from_bspline(
        self,
        nodes,
        bspline,
        h,
        geometrical_range,
        embedded=False,
        snapping_tolerance=0.03,
    ):
        geometrical_range = max(3 * h, 3 * geometrical_range)
        coordinates = np.asarray(nodes)[:, 1:3]
        self.mesh_tree = KDTree(coordinates)
        self.bspline = bspline

        t2_pt = bspline(np.array([0.0]))[0]
        t1_pt = bspline(np.array([1.0]))[0]
        geometrical_range2 = geometrical_range if embedded else 0.0

        indices_near_tip1 = np.asarray(
            self.mesh_tree.query_ball_point(t1_pt, geometrical_range), dtype=int
        )
        indices_near_tip2 = np.asarray(
            self.mesh_tree.query_ball_point(t2_pt, geometrical_range2), dtype=int
        )

        narrow_band_indices = get_narrow_band_indices(coordinates, bspline, 50, 2 * h)

        indices_subset = np.unique(
            np.concatenate([narrow_band_indices, indices_near_tip1, indices_near_tip2])
        )
        nodes_subset = coordinates[indices_subset]

        u_range = np.linspace(0, 1, 10000)
        u_i = u_range[KDTree(bspline(u_range)).query(nodes_subset)[1]]

        project_on_line(
            lambda a=u_i: bspline(a),
            lambda a=u_i: bspline(a, nu=1),
            lambda a=u_i: bspline(a, nu=2),
            u_i,
            nodes_subset,
        )

        self.phi_n = np.full(coordinates.shape[0], np.nan, dtype=np.float64)

        S = bspline(u_i)
        t_vec = bspline(u_i, nu=1)

        n_vec = np.column_stack([-t_vec[:, 1], t_vec[:, 0]])
        n_vec = n_vec / np.linalg.norm(n_vec, axis=1)[:, None]

        distance = nodes_subset - S
        phi_n_subset = np.sum(n_vec * distance, axis=1)
        self.phi_n[indices_subset] = phi_n_subset

        in_front = np.isclose(u_i, 1.0, 1e-12)
        if embedded:
            in_front |= np.isclose(u_i, 0.0, 1e-12)

        to_snap = np.where(
            (np.isclose(phi_n_subset, 0.0, atol=snapping_tolerance * h) & ~in_front)
        )[0]
        global_to_snap = indices_subset[to_snap]

        nodes[global_to_snap, 1:3] -= phi_n_subset[to_snap, None] * n_vec[to_snap, :]
        coordinates[global_to_snap] = nodes[global_to_snap, 1:3]
        self.phi_n[global_to_snap] = 0.0

        def arc_length(u, _):
            return np.linalg.norm(bspline(np.atleast_1d(np.asarray(u)), nu=1), axis=1)

        def cal_near_tip(tip, u_i_front, direction_behind, indices_near_tip, out_phi_t):
            t_tip = bspline(np.array([u_i_front]), nu=1)[0]
            t_tip /= np.linalg.norm(t_tip)
            t_tip *= -1 * direction_behind

            in_front_mask = np.isclose(u_i, u_i_front, 1e-12)
            in_front_indices = indices_subset[in_front_mask]

            out_phi_t[in_front_indices] = np.sum(
                (coordinates[in_front_indices] - tip) * t_tip, axis=1
            )
            behind_indices = np.intersect1d(
                indices_near_tip, indices_subset[~in_front_mask]
            )

            if len(behind_indices) > 0:
                u_i_behind = u_i[np.searchsorted(indices_subset, behind_indices)]
                u_end = u_i_behind[np.argmax(direction_behind * u_i_behind)]
                sol = solve_ivp(
                    arc_length,
                    t_span=[u_i_front, u_end],
                    y0=np.array([0.0]),
                    dense_output=True,
                    vectorized=True,
                )
                out_phi_t[behind_indices] = -direction_behind * sol.sol(u_i_behind)[0]

        self.phi_t_list = []
        phi_t1 = np.full(coordinates.shape[0], np.nan, dtype=np.float64)
        cal_near_tip(t1_pt, 1.0, -1, indices_near_tip1, phi_t1)
        self.phi_t_list.append(phi_t1)

        if embedded:
            phi_t2 = np.full(coordinates.shape[0], np.nan, dtype=np.float64)
            cal_near_tip(t2_pt, 0.0, 1, indices_near_tip2, phi_t2)
            self.phi_t_list.append(phi_t2)

        self.embedded = embedded

    def gen_from_ndbsplines(
        self,
        nodes,
        ndbsplines,
        h,
        geometrical_range,
        snapping_tolerance=0.03,
    ):
        geometrical_range = max(4 * h, 4 * geometrical_range)
        coordinates = np.asarray(nodes)[:, 1:4]
        self.mesh_tree = KDTree(coordinates)
        self.ndbsplines = ndbsplines

        self.phi_n = np.full(coordinates.shape[0], np.nan, dtype=np.float64)
        self.phi_t = np.full(coordinates.shape[0], np.nan, dtype=np.float64)
        projections = np.full_like(coordinates, np.nan, dtype=np.float64)
        for active, has_pole, ndbspline in ndbsplines:
            local_phi_n = np.full_like(self.phi_n, np.nan, dtype=np.float64)
            local_phi_t = local_phi_n.copy()
            local_projections = np.full_like(projections, np.nan, dtype=np.float64)
            num_u_points = max(100, 3 * ndbspline.c.shape[0])
            num_v_points = max(100, 3 * ndbspline.c.shape[0])
            narrow_band_mask = get_narrow_band_mask_3d(
                coordinates,
                ndbspline,
                num_segments_u=3 * ndbspline.c.shape[0],
                num_segments_v=3 * ndbspline.c.shape[1],
                padding=2 * h,
            )
            num_u_points = max(100, 3 * ndbspline.c.shape[0])
            num_v_points = max(100, 3 * ndbspline.c.shape[1])

            u_range = np.linspace(0, 1, num_u_points)
            v_range = np.linspace(0, 1, num_v_points)

            edge_u_1 = np.stack([np.ones_like(v_range), v_range], axis=1)
            edge_u_1_pts = ndbspline(edge_u_1)

            indices_near_tip_list = self.mesh_tree.query_ball_point(
                edge_u_1_pts,
                geometrical_range,
            )
            flattened_indices = np.concatenate(
                [np.array(x, dtype=int) for x in indices_near_tip_list]
            )

            indices_near_tip = np.unique(flattened_indices)

            narrow_band_mask[indices_near_tip] = True
            narrow_band_indices = np.where(narrow_band_mask)[0]

            narrow_band_coords = coordinates[narrow_band_mask]

            U, V = np.meshgrid(u_range, v_range, indexing="ij")
            uv_pts = np.column_stack((U.ravel(), V.ravel()))
            surf_pts = ndbspline(uv_pts)

            surface_tree = KDTree(surf_pts)
            uv_i = uv_pts[surface_tree.query(narrow_band_coords, k=1)[1]]

            project_on_surface(ndbspline, uv_i, narrow_band_coords)

            edge_configs = [
                (uv_i[:, 0] == 0.0, 1, (0, 1), (0, 2), has_pole),  # Front Edge (Pole)
                (uv_i[:, 1] == 0.0, 0, (1, 0), (2, 0), False),  # Right Base Edge
                (uv_i[:, 1] == 1.0, 0, (1, 0), (2, 0), False),  # Left Base Edge
                (uv_i[:, 0] == 1.0, 1, (0, 1), (0, 2), False),  # Crack Tip Curve
            ]

            for mask, col, nu1, nu2, skip in edge_configs:
                if skip:  # bypass singular pole
                    continue
                arr = uv_i[mask, :]
                project_on_line(
                    lambda a=arr: ndbspline(a),
                    lambda a=arr, n=nu1: ndbspline(a, nu=n),
                    lambda a=arr, n=nu2: ndbspline(a, nu=n),
                    arr[:, col],
                    narrow_band_coords[mask],
                )
                uv_i[mask] = arr

            S = ndbspline(uv_i)
            Su = ndbspline(uv_i, nu=(1, 0))
            Sv = ndbspline(uv_i, nu=(0, 1))

            n = np.cross(Su, Sv)
            n = n / np.linalg.norm(n, axis=1)[:, None]

            distance = narrow_band_coords - S
            local_phi_n[narrow_band_mask] = np.sum(distance * n, axis=1)
            local_projections[narrow_band_mask] = S

            near_tip_mask = np.zeros_like(narrow_band_mask)
            near_tip_mask[indices_near_tip] = True
            near_tip_mask[narrow_band_indices[uv_i[:, 0] == 1.0]] = True

            near_tip_coords = coordinates[near_tip_mask]
            edge_u_1_tree = KDTree(edge_u_1_pts)
            u_1_v_i = edge_u_1[edge_u_1_tree.query(near_tip_coords, k=1)[1]]

            project_on_line(
                lambda a=u_1_v_i: ndbspline(a),
                lambda a=u_1_v_i, n=(0, 1): ndbspline(a, nu=n),
                lambda a=u_1_v_i, n=(0, 2): ndbspline(a, nu=n),
                u_1_v_i[:, 1],
                near_tip_coords,
            )
            S = ndbspline(u_1_v_i)
            Su = ndbspline(u_1_v_i, nu=(1, 0))
            Sv = ndbspline(u_1_v_i, nu=(0, 1))
            n = np.cross(Su, Sv, axis=1)
            t = np.cross(Sv, n, axis=1)
            t = t / np.linalg.norm(t, axis=1)[:, None]
            distance = near_tip_coords - S
            near_tip_phi_t = np.sum(distance * t, axis=1)
            ahead_of_tip = near_tip_phi_t > 0
            near_tip_indices = np.where(near_tip_mask)[0]
            ahead_of_tip_indices = near_tip_indices[ahead_of_tip]
            local_projections[ahead_of_tip_indices, :] = np.nan
            if active:
                local_phi_t[near_tip_mask] = near_tip_phi_t
            else:
                local_phi_n[ahead_of_tip_indices] = np.nan

            valid_mask = ~np.isnan(self.phi_n) & ~np.isnan(local_phi_n)

            closer_mask = np.zeros_like(self.phi_n, dtype=bool)
            closer_mask[valid_mask] = np.abs(self.phi_n[valid_mask]) > np.abs(
                local_phi_n[valid_mask]
            )

            to_update = np.where(
                np.isnan(self.phi_n) & ~np.isnan(local_phi_n) | closer_mask
            )[0]

            to_snap = np.where(
                np.isclose(local_phi_n[to_update], 0.0, atol=snapping_tolerance * h)
                & ~(local_phi_t[to_update] > 0)
            )[0]

            global_to_snap = to_update[to_snap]

            local_phi_n[global_to_snap] = 0.0

            coordinates[global_to_snap] = local_projections[global_to_snap]

            global_idx, tip_idx, _ = np.intersect1d(
                near_tip_indices, global_to_snap, return_indices=True
            )

            proj_snapped = local_projections[global_idx]
            S_snapped = S[tip_idx]
            t_snapped = t[tip_idx]

            new_distance = proj_snapped - S_snapped
            new_phi_t = np.sum(new_distance * t_snapped, axis=1)

            local_phi_t[global_idx] = new_phi_t

            self.phi_n[to_update] = local_phi_n[to_update]
            if active:
                self.phi_t[to_update] = local_phi_t[to_update]
                self.phi_t_list.append(local_phi_t)
            projections[to_update] = local_projections[to_update]

        self.embedded = False

    def get(self, nodes, tip):
        assert self.phi_n is not None
        assert self.phi_t_list is not None
        nodes = np.asarray(nodes) - 1
        phi_n = self.phi_n[nodes]
        if tip is not None:
            phi_t = self.phi_t_list[tip][nodes]
        else:
            phi_t = None
        return phi_n, phi_t

    def is_cut(self, element) -> Tuple[CutType, None | int, bool]:
        assert self.phi_n is not None
        assert self.phi_t_list is not None
        nodes = np.asarray(element[4]) - 1
        elem_type = element[1]
        num_edges, denom_edges = ELEM_EDGES[elem_type]
        phi_n = self.phi_n[nodes]
        if np.any(np.isnan(phi_n)):
            return CutType.NONE, None, False
        n_nodes = len(nodes)
        # no sign change of normal level set inside element or at node / edge
        sign_n = (1 - np.isclose(phi_n, 0, atol=1e-12)) * np.sign(phi_n)
        m = sign_n[denom_edges] * sign_n[num_edges]
        if np.all(m > 0):
            return CutType.NONE, None, False
        for phi_t_i in self.phi_t_list:
            phi_t = phi_t_i[nodes]
            sign_t = (1 - np.isclose(phi_t, 0, atol=1e-12)) * np.sign(phi_t)
            if np.sum(sign_t) == n_nodes:
                return CutType.NONE, None, False
        num = phi_n[num_edges]
        denom = num - phi_n[denom_edges]
        unsolvable = denom == 0
        denom += unsolvable
        N1 = np.divide(num, denom, out=np.zeros_like(num), where=~unsolvable)
        in_element = (N1 >= 0) & (N1 <= 1)
        actual_cuts = ~unsolvable & in_element
        n_actual_cuts = np.sum(actual_cuts)
        touching = not (
            np.any(
                actual_cuts
                & ~np.isclose(N1, 0, atol=1e-12)
                & ~np.isclose(N1, 1, atol=1e-12)
            )
            or n_actual_cuts == actual_cuts.shape[0]
        )
        if n_actual_cuts == 0:
            return CutType.NONE, None, False

        for i, phi_t_i in enumerate(self.phi_t_list):
            phi_t = phi_t_i[nodes]
            d_t = N1 * phi_t[denom_edges] + (1 - N1) * phi_t[num_edges]
            x = np.sum(
                (1 - np.isclose(d_t, 0, atol=1e-12)) * np.sign(d_t) * actual_cuts
            )
            if x == n_actual_cuts:
                return CutType.NONE, None, False
            if x > -n_actual_cuts and n_actual_cuts > 1:
                return CutType.PARTIAL, i, touching
        return CutType.CUT, None, touching

    def in_range(self, element, radius) -> Tuple[bool, None | int, None | NDArray]:
        assert self.phi_n is not None
        assert self.phi_t_list is not None
        if radius == 0.0:
            return (False, None, None)

        nodes = np.asarray(element[4]) - 1
        phi_n = self.phi_n[nodes]
        if np.any(np.isnan(phi_n)):
            return (False, None, None)

        is_in_range_final = False
        in_range_final = None
        i_final = -1
        for i, phi_t_i in enumerate(self.phi_t_list):
            phi_t = phi_t_i[nodes]
            if np.any(np.isnan(phi_t)):
                continue
            r = phi_n**2 + phi_t**2
            in_range = r < radius**2
            if np.all(in_range):
                if is_in_range_final:
                    print("warning: overlapping geometrical enrichment")
                in_range_final = in_range
                i_final = i
        if is_in_range_final:
            return (True, i_final, in_range_final)
        return (False, None, None)


def get_narrow_band_indices(nodes, bspline, num_segments, padding):
    u = np.linspace(0.0, 1.0, num_segments + 1)
    curve_pts = bspline(u)

    min_coords = np.minimum(curve_pts[:-1], curve_pts[1:]) - padding
    max_coords = np.maximum(curve_pts[:-1], curve_pts[1:]) + padding

    in_x = (nodes[:, None, 0] >= min_coords[:, 0]) & (
        nodes[:, None, 0] <= max_coords[:, 0]
    )

    in_y = (nodes[:, None, 1] >= min_coords[:, 1]) & (
        nodes[:, None, 1] <= max_coords[:, 1]
    )

    in_box = in_x & in_y
    is_narrow_band = np.any(in_box, axis=1)

    narrow_band_indices = np.where(is_narrow_band)[0]

    return narrow_band_indices


def get_narrow_band_mask_3d(nodes, ndbspline, num_segments_u, num_segments_v, padding):
    u = np.linspace(0.0, 1.0, num_segments_u + 1)
    v = np.linspace(0.0, 1.0, num_segments_v + 1)
    U, V = np.meshgrid(u, v, indexing="ij")

    uv_pts = np.column_stack((U.ravel(), V.ravel()))
    surf_pts = ndbspline(uv_pts).reshape((num_segments_u + 1, num_segments_v + 1, 3))

    p00 = surf_pts[:-1, :-1, :]
    p10 = surf_pts[1:, :-1, :]
    p01 = surf_pts[:-1, 1:, :]
    p11 = surf_pts[1:, 1:, :]

    min_coords = np.minimum.reduce([p00, p10, p01, p11]).reshape(-1, 3) - padding
    max_coords = np.maximum.reduce([p00, p10, p01, p11]).reshape(-1, 3) + padding

    num_patches = num_segments_u * num_segments_v

    is_narrow_band = np.zeros(len(nodes), dtype=bool)

    for p in range(num_patches):
        in_box = (
            (nodes[:, 0] >= min_coords[p, 0])
            & (nodes[:, 0] <= max_coords[p, 0])
            & (nodes[:, 1] >= min_coords[p, 1])
            & (nodes[:, 1] <= max_coords[p, 1])
            & (nodes[:, 2] >= min_coords[p, 2])
            & (nodes[:, 2] <= max_coords[p, 2])
        )

        is_narrow_band |= in_box

    return is_narrow_band


def project_on_line(S_fn, dS_fn, ddS_fn, u_i_slice, nodes_subset):
    for i in range(1000):
        S = S_fn()
        dS = dS_fn()
        ddS = ddS_fn()

        distance = S - nodes_subset

        f = np.sum(dS * distance, axis=1)
        df = np.sum(ddS * distance, axis=1) + np.sum(dS * dS, axis=1)

        df_gn = np.sum(dS * dS, axis=1)
        df = np.where(df < 1e-12, df_gn, df)

        du = f / df

        pushing_past_1 = (u_i_slice == 1.0) & (f < 0)
        pushing_past_0 = (u_i_slice == 0.0) & (f > 0)

        u_i_next = np.clip(u_i_slice - du, 0.0, 1.0)

        u_i_next = np.where(pushing_past_1 | pushing_past_0, u_i_slice, u_i_next)

        if np.all(np.isclose(u_i_slice, u_i_next, atol=1e-12)):
            break
        elif i == 999:
            bad_mask = ~np.isclose(u_i_slice, u_i_next, atol=1e-12)
            print(f"Failed at u_i: {u_i_slice[bad_mask]}")
            raise ValueError("Newton iterations didn't converge")

        u_i_slice[:] = u_i_next


def project_on_surface(ndbspline, uv_i_slice, nodes_subset):
    for i in range(1000):
        S = ndbspline(uv_i_slice)
        Su = ndbspline(uv_i_slice, nu=(1, 0))
        Sv = ndbspline(uv_i_slice, nu=(0, 1))
        Suu = ndbspline(uv_i_slice, nu=(2, 0))
        Svv = ndbspline(uv_i_slice, nu=(0, 2))
        Suv = ndbspline(uv_i_slice, nu=(1, 1))

        distance = S - nodes_subset

        F1 = np.sum(Su * distance, axis=1)
        F2 = np.sum(Sv * distance, axis=1)

        # Precompute dot products for the Hessian
        Su_dot_Su = np.sum(Su * Su, axis=1)
        Sv_dot_Sv = np.sum(Sv * Sv, axis=1)
        Su_dot_Sv = np.sum(Su * Sv, axis=1)

        # 3. True Hessian Matrix Components (J11, J12, J22)
        J11 = np.sum(Suu * distance, axis=1) + Su_dot_Su
        J12 = np.sum(Suv * distance, axis=1) + Su_dot_Sv  # Symmetric (J21 = J12)
        J22 = np.sum(Svv * distance, axis=1) + Sv_dot_Sv

        det_J = (J11 * J22) - (J12 * J12)

        # 4. MULTIVARIATE HESSIAN SAFEGUARD
        # A 2x2 Hessian is positive definite (convex) if J11 > 0 AND det_J > 0.
        # If the surface is locally concave relative to the point, we fall back to Gauss-Newton.
        bad_hessian = (J11 < 1e-12) | (det_J < 1e-12)

        J11 = np.where(bad_hessian, Su_dot_Su, J11)
        J22 = np.where(bad_hessian, Sv_dot_Sv, J22)
        J12 = np.where(bad_hessian, Su_dot_Sv, J12)

        # Recalculate determinant for the Gauss-Newton fallback
        det_J = np.maximum((J11 * J22) - (J12 * J12), 1e-14)  # Prevent ZeroDivision

        # 5. Compute the Newton Steps (Analytic 2x2 Inverse)
        du = (J22 * F1 - J12 * F2) / det_J
        dv = (-J12 * F1 + J11 * F2) / det_J

        uv_next = np.clip(uv_i_slice - np.column_stack((du, dv)), 0.0, 1.0)

        pushing_u_1 = (uv_i_slice[:, 0] == 1.0) & (F1 < 0)
        pushing_u_0 = (uv_i_slice[:, 0] == 0.0) & (F1 > 0)
        u_caught = pushing_u_1 | pushing_u_0

        pushing_v_1 = (uv_i_slice[:, 1] == 1.0) & (F2 < 0)
        pushing_v_0 = (uv_i_slice[:, 1] == 0.0) & (F2 > 0)
        v_caught = pushing_v_1 | pushing_v_0

        caught = u_caught | v_caught

        uv_next = np.where(caught[:, None], uv_i_slice, uv_next)

        if np.all(np.isclose(uv_i_slice, uv_next, atol=1e-12)):
            break
        elif i == 999:
            bad_mask = ~np.all(np.isclose(uv_i_slice, uv_next, atol=1e-15), axis=1)
            print(f"Failed to converge at nodes: {np.where(bad_mask)[0]}")
            print(uv_i_slice[bad_mask])
            print("coordinates")
            print(nodes_subset[bad_mask])
            raise ValueError("3D Newton iterations didn't converge")

        uv_i_slice = uv_next
