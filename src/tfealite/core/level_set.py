import numpy as np
from numpy.typing import NDArray
from enum import Enum, auto
from typing import Tuple


class CutType(Enum):
    NONE = auto()
    CUT = auto()
    PARTIAL = auto()


class LevelSet:
    # phi_n: NDArray
    # phi_t: NDArray
    def __init__(self):
        self.phi_n = None
        self.phi_t = None
        self.phi_t2 = None

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
        self.phi_n = phi_n
        self.phi_t = phi_t1
        if embedded:
            self.phi_t2 = phi_t2

    def get(self, nodes, tip):
        assert self.phi_n is not None
        assert self.phi_t is not None
        nodes = np.asarray(nodes) - 1
        phi_n = self.phi_n[nodes]
        if tip is None or tip == 1:
            phi_t = self.phi_t[nodes]
        else:
            assert self.phi_t2 is not None
            phi_t = self.phi_t2[nodes]
        return phi_n, phi_t

    def is_cut(self, element) -> Tuple[CutType, None | int]:
        assert self.phi_n is not None
        assert self.phi_t is not None
        nodes = np.asarray(element[4]) - 1
        phi_n = self.phi_n[nodes]
        phi_t = self.phi_t[nodes]
        phi_t2 = None
        n_nodes = len(nodes)
        # no sign change of normal level set inside element or at node / edge
        sign_n = (1 - np.isclose(phi_n, 0)) * np.sign(phi_n)
        m1 = sign_n[0] * sign_n[-1]
        m2 = sign_n[:-1] * sign_n[1:]
        if m1 > 0 and np.all(m2 > 0):
            return CutType.NONE, None
        sign_t = (1 - np.isclose(phi_t, 0)) * np.sign(phi_t)
        if np.sum(sign_t) == n_nodes:
            return CutType.NONE, None
        if self.phi_t2 is not None:
            phi_t2 = self.phi_t2[nodes]
            sign_t2 = (1 - np.isclose(phi_t2, 0)) * np.sign(phi_t2)
            if np.sum(sign_t2) == n_nodes:
                return CutType.NONE, None
        num = np.empty_like(phi_n)
        num[:-1] = phi_n[1:]
        num[-1] = phi_n[0]
        denom = num - phi_n
        unsolvable = denom == 0
        denom += unsolvable
        N1 = num / denom
        in_element = (N1 >= 0) & (N1 <= 1)
        if not np.any(~unsolvable & in_element):
            return CutType.NONE, None
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
                (1 - np.isclose(d_t2, 0.0)) * np.sign(d_t2) * (~unsolvable & in_element)
            )
        x = np.sum(
            (1 - np.isclose(d_t, 0.0)) * np.sign(d_t) * (~unsolvable & in_element)
        )

        if x == 2:
            return CutType.NONE, None
        if x2 is not None:
            if x2 == 2:
                return CutType.NONE, None
            elif x2 >= -1:
                return CutType.PARTIAL, 2
        if x >= -1:
            return CutType.PARTIAL, 1
        return CutType.CUT, None

    def in_range(self, element, radius) -> Tuple[bool, None | int]:
        assert self.phi_n is not None
        assert self.phi_t is not None
        nodes = np.asarray(element[4]) - 1
        phi_n = self.phi_n[nodes]
        phi_t = self.phi_t[nodes]
        phi_t2 = None

        r1 = np.sqrt(phi_n**2 + phi_t**2)
        in_range1 = np.any(r1 <= radius)
        in_range2 = False
        if self.phi_t2 is not None:
            phi_t2 = self.phi_t2[nodes]
            r2 = np.sqrt(phi_n**2 + phi_t2**2)
            in_range2 = np.any(r2 <= radius)
            if in_range1 and in_range2:
                print("warning: overlapping geometrical enrichment")
            if in_range2:
                return (True, 2)
        if in_range1:
            return (True, 1)
        return (False, None)


def gen_from_line_segment(
    nodes: list[tuple[int, float, float, float]],
    elements: list[tuple[int, str, int, int, tuple[int, ...]]],
    p1: NDArray,
    p2: NDArray,
) -> tuple[NDArray, NDArray, list[int], list[int]]:
    coordinates = np.array(nodes, dtype=float)[:, 1:3]
    v = p2 - p1
    n = np.array([-v[1], v[0]])
    n = n / np.linalg.norm(n)
    psi_n = np.sum((coordinates - p1) * n, axis=1)

    t2 = v / np.linalg.norm(v)
    # t1 = -t2

    # psi_t1 = np.sum((coordinates - p1) * t1, axis=1)
    psi_t2 = np.sum((coordinates - p2) * t2, axis=1)
    # print(psi_t2)
    # print(nodes.T)
    # print(psi_n)
    # print(psi_t2)

    cut = []
    tip = []
    for [id, _, _, _, nodes] in elements:
        n_vals = psi_n[np.array(nodes) - 1,]
        t_vals = psi_t2[np.array(nodes) - 1,]
        if abs(np.sum(np.sign(n_vals))) != 3:
            # print("here", n_vals)
            if np.sum(np.sign(t_vals)) == -3:
                cut.append(id)
            elif abs(np.sum(np.sign(t_vals))) != 3:
                tip.append(id)

    # assert tip
    return [psi_n, psi_t2, cut, tip]
