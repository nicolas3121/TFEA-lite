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
        self.mesh_tree = None
        self.bspline = None
        self.dbspline = None
        self.t = None
        self.t2 = None

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
        self.dbspline = bspline.derivative()
        self.ddbspline = bspline.derivative(nu=2)

        self.phi_n = phi_n
        self.phi_t = phi_t1
        self.embedded = embedded
        if embedded:
            self.phi_t2 = phi_t2

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
        self.embedded = embedded
        if embedded:
            self.phi_t2 = phi_t2

    # node snapping snapt momenteel ook nodes die voor de crack liggen, mogelijk aanpassen zodat niet meer gebeurt
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
        # print(geometrical_range)
        coordinates = np.asarray(nodes)[:, 1:3]
        self.mesh_tree = KDTree(coordinates)
        self.bspline = bspline
        dbspline = bspline.derivative(1)
        self.dbspline = dbspline
        ddbspline = bspline.derivative(2)
        self.ddbspline = bspline.derivative(2)

        t2 = bspline(0.0)
        t1 = bspline(1.0)
        geometrical_range2 = 0.0
        if embedded:
            geometrical_range2 = geometrical_range
        indices_near_tip1 = np.asarray(
            self.mesh_tree.query_ball_point(t1, geometrical_range), dtype=int
        )
        indices_near_tip2 = np.asarray(
            self.mesh_tree.query_ball_point(t2, geometrical_range2), dtype=int
        )

        indices_subset = np.union1d(
            np.union1d(
                get_narrow_band_indices(coordinates, bspline, 50, 2 * h),
                indices_near_tip1,
            ),
            indices_near_tip2,
        )
        nodes_subset = coordinates[indices_subset]
        # self.indices_subset = indices_subset
        problematic_nodes = np.array([10, 11, 23, 22]) - 1
        np.where(
            np.bitwise_or.reduce(
                indices_subset[None, :] == problematic_nodes[:, None], axis=0
            )
        )[0]

        u_range = np.linspace(0, 1, 10000)
        u_i = u_range[KDTree(bspline(u_range)).query(nodes_subset)[1]]

        for i in range(1000):
            S = bspline(u_i)
            dS = dbspline(u_i)
            ddS = ddbspline(u_i)
            distance = S - nodes_subset

            f = np.sum(dS * distance, axis=1)
            df = np.sum(ddS * distance, axis=1) + np.sum(dS * dS, axis=1)

            # 1. HESSIAN SAFEGUARD:
            # If df goes negative, the parabola flipped. Fallback to Gauss-Newton.
            df_gn = np.sum(dS * dS, axis=1)
            df = np.where(df < 1e-12, df_gn, df)
            # print("f", f[locations], "df", df[locations])

            du = f / df

            # 2. BOUNDARY LOCK:
            # f < 0 means u wants to increase. f > 0 means u wants to decrease.
            pushing_past_1 = (u_i == 1.0) & (f < 0)
            pushing_past_0 = (u_i == 0.0) & (f > 0)

            u_i_next = np.clip(u_i - du, 0.0, 1.0)

            # Force convergence for nodes stuck pushing against the boundary
            u_i_next = np.where(pushing_past_1 | pushing_past_0, u_i, u_i_next)
            # print("u_i", u_i[locations], "du", du[locations])

            if np.all(np.isclose(u_i, u_i_next, atol=1e-15)):
                break
            elif i == 999:
                # Optional: Print out the specific nodes failing to help debug
                bad_mask = ~np.isclose(u_i, u_i_next, atol=1e-15)
                print(f"Failed at u_i: {u_i[bad_mask]}")
                raise ValueError("Newton iterations didn't converge")

            u_i = u_i_next

        phi_n = np.full(coordinates.shape[0], np.nan, dtype=np.float64)
        S = bspline(u_i)
        t = dbspline(u_i)
        n = t[:, [1, 0]]
        n[:, 0] *= -1
        n = n / np.linalg.norm(n, axis=1)[:, None]
        distance = nodes_subset - S
        phi_n_subset = np.sum(n * distance, axis=1)
        phi_n[indices_subset] = phi_n_subset
        in_front = np.isclose(u_i, 1.0, 1e-12)
        if embedded:
            in_front |= np.isclose(u_i, 0.0, 1e-12)
        to_snap = np.where(
            (np.isclose(phi_n_subset, 0.0, atol=snapping_tolerance * h) & ~in_front)
        )[0]
        global_to_snap = indices_subset[to_snap]
        nodes[global_to_snap, 1:3] -= phi_n_subset[to_snap, None] * n[to_snap, :]
        coordinates[global_to_snap] = nodes[global_to_snap, 1:3]
        phi_n[global_to_snap] = 0.0
        coordinates[global_to_snap] = nodes[global_to_snap][:, 1:3]

        def arc_length(u, _):
            return np.linalg.norm(dbspline(np.atleast_1d(np.asarray(u))), axis=1)

        def cal_near_tip(tip, u_i_front, direction_behind, indices_near_tip, phi_t):
            t_tip = dbspline(u_i_front)
            t_tip /= np.linalg.norm(t_tip)
            t_tip *= -1 * direction_behind

            in_front = np.isclose(u_i, u_i_front, 1e-12)
            in_front_indices = indices_subset[in_front]

            phi_t[in_front_indices] = np.sum(
                (coordinates[in_front_indices] - tip) * t_tip, axis=1
            )
            behind_indices = np.intersect1d(indices_near_tip, indices_subset[~in_front])
            # print(behind_indices)
            if len(behind_indices):
                u_i_behind = u_i[
                    np.searchsorted(indices_subset, behind_indices)
                ]  # indices_subset is sorted due to union1d
                u_end = u_i_behind[np.argmax(direction_behind * u_i_behind)]
                sol = solve_ivp(
                    arc_length,
                    t_span=[u_i_front, u_end],
                    y0=np.array([0.0]),
                    dense_output=True,
                    vectorized=True,
                )
                phi_t[behind_indices] = -direction_behind * sol.sol(u_i_behind)[0]

        phi_t = np.full(coordinates.shape[0], np.nan, dtype=np.float64)
        cal_near_tip(t1, 1, -1, indices_near_tip1, phi_t)
        if embedded:
            phi_t2 = np.full(coordinates.shape[0], np.nan, dtype=np.float64)
            cal_near_tip(t2, 0, 1, indices_near_tip2, phi_t2)
            self.phi_t2 = phi_t2

        self.embedded = embedded
        self.phi_n = phi_n
        self.phi_t = phi_t
        self.t = t1

    def gen_from_ndbsplines(
        self,
        nodes,
        ndbsplines,
        h,
        geometrical_range,
        snapping_tolerance=0.03,
    ):
        geometrical_range = max(3 * h, 3 * geometrical_range)
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

            edge_u_0 = np.stack([np.zeros_like(v_range), v_range], axis=1)
            edge_u_0_tree = KDTree(ndbspline(edge_u_0))
            edge_u_1 = np.stack([np.ones_like(v_range), v_range], axis=1)
            edge_u_1_pts = ndbspline(edge_u_1)
            edge_u_1_tree = KDTree(edge_u_1_pts)
            edge_v_0 = np.stack([u_range, np.zeros_like(u_range)], axis=1)
            edge_v_0_tree = KDTree(ndbspline(edge_v_0))
            edge_v_1 = np.stack([u_range, np.ones_like(u_range)], axis=1)
            edge_v_1_tree = KDTree(ndbspline(edge_v_1))

            indices_near_tip_list = self.mesh_tree.query_ball_point(
                edge_u_1_pts,
                geometrical_range,
            )

            indices_near_tip = np.unique(
                np.concatenate(indices_near_tip_list, dtype=int)
            )

            near_tip_mask = np.zeros_like(narrow_band_mask)
            near_tip_mask[indices_near_tip] = True

            narrow_band_mask[indices_near_tip] = True
            narrow_band_indices = np.where(narrow_band_mask)[0]

            narrow_band_coords = coordinates[narrow_band_mask]

            # Search along edges first to exclude nodes that can't be projected

            u_0_v_i = edge_u_0[edge_u_0_tree.query(narrow_band_coords, k=1)[1]]
            u_i_v_0 = edge_v_0[edge_v_0_tree.query(narrow_band_coords, k=1)[1]]
            u_i_v_1 = edge_v_1[edge_v_1_tree.query(narrow_band_coords, k=1)[1]]
            u_1_v_i = edge_u_1[edge_u_1_tree.query(narrow_band_coords, k=1)[1]]
            phi_t_u_0_v_i = np.full(narrow_band_coords.shape[0], np.nan)
            phi_t_u_i_v_0 = np.full(narrow_band_coords.shape[0], np.nan)
            phi_t_u_i_v_1 = np.full(narrow_band_coords.shape[0], np.nan)
            phi_t_u_1_v_i = np.full(narrow_band_coords.shape[0], np.nan)
            phi_n_u_0_v_i = np.full(narrow_band_coords.shape[0], np.nan)
            phi_n_u_i_v_0 = np.full(narrow_band_coords.shape[0], np.nan)
            phi_n_u_i_v_1 = np.full(narrow_band_coords.shape[0], np.nan)
            phi_n_u_1_v_i = np.full(narrow_band_coords.shape[0], np.nan)
            edge_configs = [
                (u_0_v_i, 1, (0, 1), (0, 2), has_pole),  # Front Edge (Pole)
                (u_i_v_0, 0, (1, 0), (2, 0), False),  # Right Base Edge
                (u_i_v_1, 0, (1, 0), (2, 0), False),  # Left Base Edge
                (u_1_v_i, 1, (0, 1), (0, 2), False),  # Crack Tip Curve
            ]
            distance_configs = [
                (phi_t_u_0_v_i, phi_n_u_0_v_i, (1, 0), (0, 1), 1, 1),
                (phi_t_u_i_v_0, phi_n_u_i_v_0, (0, 1), (1, 0), -1, -1),
                (phi_t_u_i_v_1, phi_n_u_i_v_1, (0, 1), (1, 0), 1, -1),
                (phi_t_u_1_v_i, phi_n_u_1_v_i, (1, 0), (0, 1), -1, 1),
            ]

            search_2d_exclusion_mask = np.zeros(u_i_v_0.shape[0], dtype=bool)

            for (arr, col, nu1, nu2, skip), (
                edge_phi_t,
                edge_phi_n,
                nu_n,
                nu_t,
                tangential_sign,
                normal_sign,
            ) in zip(edge_configs, distance_configs):
                if not skip:  # bypass singular pole
                    project_on_line(
                        lambda a=arr: ndbspline(a),
                        lambda a=arr, n=nu1: ndbspline(a, nu=n),
                        lambda a=arr, n=nu2: ndbspline(a, nu=n),
                        arr[:, col],
                        narrow_band_coords,
                    )

                S = ndbspline(arr)
                dS_n = ndbspline(arr, nu=nu_n)
                dS_t = ndbspline(arr, nu=nu_t)

                # print("o")
                o = normal_sign * np.cross(dS_n, dS_t, axis=1)
                o = o / np.linalg.norm(o, axis=1)[:, None]
                # print(o)
                n = tangential_sign * np.cross(o, dS_t, axis=1)
                n = n / np.linalg.norm(n, axis=1)[:, None]
                print("n")
                print(n)
                # print("n")
                # print(n)
                distance = narrow_band_coords - S

                edge_phi_t[:] = np.sum(distance * n, axis=1)
                edge_phi_n[:] = np.sum(distance * o, axis=1)

                is_outside = edge_phi_t >= 0
                print("is_outside")
                print(is_outside)
                search_2d_exclusion_mask[is_outside] = True
                local_phi_n[narrow_band_indices[is_outside]] = edge_phi_n[is_outside]
            print(search_2d_exclusion_mask)

            U, V = np.meshgrid(u_range, v_range, indexing="ij")
            uv_pts = np.column_stack((U.ravel(), V.ravel()))
            surf_pts = ndbspline(uv_pts)

            surface_tree = KDTree(surf_pts)
            surface_uv_i = uv_pts[
                surface_tree.query(narrow_band_coords[~search_2d_exclusion_mask], k=1)[
                    1
                ]
            ]
            surface_nodes_subset = narrow_band_coords[~search_2d_exclusion_mask]
            project_on_surface(ndbspline, surface_uv_i, surface_nodes_subset)
            S = ndbspline(surface_uv_i)
            Su = ndbspline(surface_uv_i, nu=(1, 0))
            Sv = ndbspline(surface_uv_i, nu=(0, 1))

            n_raw = np.cross(Su, Sv)
            n = n_raw / np.linalg.norm(n_raw, axis=1)[:, None]

            distance = surface_nodes_subset - S
            surface_phi_n = np.sum(distance * n, axis=1)

            # to_exclude_mask = (phi_t_u_i_v_0 < 0) | (phi_t_u_i_v_1 > 0)
            local_near_tip_mask = near_tip_mask[narrow_band_indices]
            local_phi_t[near_tip_mask] = phi_t_u_1_v_i[local_near_tip_mask]
            local_phi_n[near_tip_mask] = phi_n_u_1_v_i[local_near_tip_mask]

            surface_subset_indices = narrow_band_indices[~search_2d_exclusion_mask]
            local_phi_n[surface_subset_indices] = surface_phi_n
            local_projections[surface_subset_indices] = S
            print(local_phi_n)
            self.phi_n = local_phi_n
            self.phi_t = local_phi_t

    def get(self, nodes, tip):
        assert self.phi_n is not None
        assert self.phi_t is not None
        nodes = np.asarray(nodes) - 1
        phi_n = self.phi_n[nodes]
        if tip is None or tip == 1 or tip == 0:
            phi_t = self.phi_t[nodes]
        else:
            assert self.phi_t2 is not None
            phi_t = self.phi_t2[nodes]
        return phi_n, phi_t

    def is_cut(self, element) -> Tuple[CutType, None | int, bool]:
        assert self.phi_n is not None
        assert self.phi_t is not None
        nodes = np.asarray(element[4]) - 1
        elem_type = element[1]
        num_edges, denom_edges = ELEM_EDGES[elem_type]
        phi_n = self.phi_n[nodes]
        phi_t = self.phi_t[nodes]
        if np.any(np.isnan(phi_n)):
            return CutType.NONE, None, False
        phi_t2 = None
        n_nodes = len(nodes)
        # no sign change of normal level set inside element or at node / edge
        sign_n = (1 - np.isclose(phi_n, 0, atol=1e-12)) * np.sign(phi_n)
        m = sign_n[denom_edges] * sign_n[num_edges]
        if np.all(m > 0):
            return CutType.NONE, None, False
        sign_t = (1 - np.isclose(phi_t, 0, atol=1e-12)) * np.sign(phi_t)
        if np.sum(sign_t) == n_nodes:
            return CutType.NONE, None, False
        if self.phi_t2 is not None:
            phi_t2 = self.phi_t2[nodes]
            sign_t2 = (1 - np.isclose(phi_t2, 0, atol=1e-12)) * np.sign(phi_t2)
            if np.sum(sign_t2) == n_nodes:
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
        d_t = N1 * phi_t[denom_edges] + (1 - N1) * phi_t[num_edges]
        d_t2 = None
        x2 = None
        if phi_t2 is not None:
            d_t2 = N1 * phi_t2[denom_edges] + (1 - N1) * phi_t2[num_edges]
            x2 = np.sum(
                (1 - np.isclose(d_t2, 0.0, 1e-12)) * np.sign(d_t2) * actual_cuts
            )
        x = np.sum((1 - np.isclose(d_t, 0.0, 1e-12)) * np.sign(d_t) * actual_cuts)

        if x == n_actual_cuts:
            return CutType.NONE, None, False
        if x2 is not None:
            if x2 == n_actual_cuts:
                return CutType.NONE, None, False
            elif x2 > -n_actual_cuts and n_actual_cuts > 1:
                return CutType.PARTIAL, 2, touching
        if x > -n_actual_cuts and n_actual_cuts > 1:
            return CutType.PARTIAL, 1, touching
        return CutType.CUT, None, touching

    def in_range(self, element, radius) -> Tuple[bool, None | int, None | NDArray]:
        assert self.phi_n is not None
        assert self.phi_t is not None
        if radius == 0.0:
            return (False, None, None)

        nodes = np.asarray(element[4]) - 1
        phi_n = self.phi_n[nodes]
        phi_t = self.phi_t[nodes]
        if np.any(np.isnan(phi_n)) or np.any(np.isnan(phi_t)):
            return (False, None, None)

        phi_t2 = None

        r1 = np.sqrt(phi_n**2 + phi_t**2)
        in_range1 = r1 <= radius
        is_in_range1 = np.any(in_range1)
        in_range2 = None
        if self.phi_t2 is not None:
            phi_t2 = self.phi_t2[nodes]
            if np.any(np.isnan(phi_t2)):
                return (False, None, None)
            r2 = np.sqrt(phi_n**2 + phi_t2**2)
            in_range2 = r2 <= radius
            is_in_range2 = np.any(in_range2)

            if is_in_range1 and is_in_range2:
                print("warning: overlapping geometrical enrichment")
            if is_in_range2:
                return (True, 2, in_range2)
        if is_in_range1:
            return (True, 1, in_range1)
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

        # 6. BOUNDARY LOCKS (Active Set Phase 1)
        # Check if nodes are pushing past boundaries
        pushing_u_1 = (uv_i_slice[:, 0] == 1.0) & (F1 < 0)
        pushing_u_0 = (uv_i_slice[:, 0] == 0.0) & (F1 > 0)
        u_caught = pushing_u_1 | pushing_u_0

        pushing_v_1 = (uv_i_slice[:, 1] == 1.0) & (F2 < 0)
        pushing_v_0 = (uv_i_slice[:, 1] == 0.0) & (F2 > 0)
        v_caught = pushing_v_1 | pushing_v_0

        uv_next[:, 1] = np.where(v_caught, uv_i_slice[:, 1], uv_next[:, 1])

        uv_next[:, 0] = np.where(u_caught, uv_i_slice[:, 0], uv_next[:, 0])

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
