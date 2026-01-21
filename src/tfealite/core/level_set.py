import numpy as np
from numpy.typing import NDArray


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

    cut = []
    tip = []
    for [id, _, _, _, nodes] in elements:
        n_vals = psi_n[np.array(nodes) - 1,]
        t_vals = psi_t2[np.array(nodes) - 1,]
        print("here")
        # print("n", n_vals)
        # print("t", t_vals)
        if abs(np.sum(np.sign(n_vals))) != 3:
            print("here", n_vals)
            if np.sum(np.sign(t_vals)) == -3:
                cut.append(id)
            elif abs(np.sum(np.sign(t_vals))) != 3:
                tip.append(id)
    assert tip
    return [psi_n, psi_t2, cut, tip]
