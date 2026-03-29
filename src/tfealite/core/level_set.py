from enum import Enum, auto
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.interpolate import BSpline, splprep
from scipy.spatial import KDTree


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
        # print("u_i_initial", u_i[locations])

        # for i in range(100):
        #     S = bspline(u_i)
        #     dS = dbspline(u_i)
        #     ddS = ddbspline(u_i)
        #     distance = S - nodes_subset
        #     f = np.sum(dS * distance, axis=1)
        #     df = np.sum(ddS * distance, axis=1) + np.sum(dS * dS, axis=1)
        #     du = f / df
        #     pushing_past_1 = (u_i == 1.0) & (f < 0)
        #     pushing_past_0 = (u_i == 0.0) & (f > 0)
        #     # print("u_i", u_i[locations], "du", du[locations])
        #     u_i_next = np.clip(u_i - du, 0.0, 1.0)
        #     u_i_next = np.where(pushing_past_1 | pushing_past_0, u_i, u_i_next)
        #     if np.all(np.isclose(u_i, u_i_next, atol=1e-15)):
        #         break
        #     elif i == 99:
        #         raise ValueError("newton iterations didn't converge")
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
        phi_n = self.phi_n[nodes]
        phi_t = self.phi_t[nodes]
        if np.any(np.isnan(phi_n)):
            return CutType.NONE, None, False
        phi_t2 = None
        n_nodes = len(nodes)
        # no sign change of normal level set inside element or at node / edge
        sign_n = (1 - np.isclose(phi_n, 0, atol=1e-12)) * np.sign(phi_n)
        m1 = sign_n[0] * sign_n[-1]
        m2 = sign_n[:-1] * sign_n[1:]
        if m1 > 0 and np.all(m2 > 0):
            return CutType.NONE, None, False
        sign_t = (1 - np.isclose(phi_t, 0, atol=1e-12)) * np.sign(phi_t)
        if np.sum(sign_t) == n_nodes:
            return CutType.NONE, None, False
        if self.phi_t2 is not None:
            phi_t2 = self.phi_t2[nodes]
            sign_t2 = (1 - np.isclose(phi_t2, 0, atol=1e-12)) * np.sign(phi_t2)
            if np.sum(sign_t2) == n_nodes:
                return CutType.NONE, None, False
        num = np.empty_like(phi_n)
        num[:-1] = phi_n[1:]
        num[-1] = phi_n[0]
        denom = num - phi_n
        unsolvable = denom == 0
        denom += unsolvable
        N1 = np.divide(num, denom, out=np.zeros_like(num), where=~unsolvable)
        # N1 = num / denom
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
        d_t = np.empty_like(phi_n)
        d_t[:-1] = N1[:-1] * phi_t[:-1] + (1 - N1[:-1]) * phi_t[1:]
        d_t[-1] = N1[-1] * phi_t[-1] + (1 - N1[-1]) * phi_t[0]
        d_t2 = None
        x2 = None
        if phi_t2 is not None:
            d_t2 = np.empty_like(phi_n)
            d_t2[:-1] = N1[:-1] * phi_t2[:-1] + (1 - N1[:-1]) * phi_t2[1:]
            d_t2[-1] = N1[-1] * phi_t2[-1] + (1 - N1[-1]) * phi_t2[0]
            x2 = np.sum(
                (1 - np.isclose(d_t2, 0.0, 1e-12)) * np.sign(d_t2) * actual_cuts
            )
        x = np.sum((1 - np.isclose(d_t, 0.0, 1e-12)) * np.sign(d_t) * actual_cuts)

        if x == 2:
            return CutType.NONE, None, False
        if x2 is not None:
            if x2 == 2:
                return CutType.NONE, None, False
            elif x2 >= -1 and n_actual_cuts > 1:
                return CutType.PARTIAL, 2, touching
        if x >= -1 and n_actual_cuts > 1:
            return CutType.PARTIAL, 1, touching
        # print(CutType.CUT, "phi_n", phi_n, "phi_t", phi_t)
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
