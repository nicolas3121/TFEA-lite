import numpy as np
import pytest
from tfealite.core.level_set import LevelSet, CutType

POINTS_DATA_INSIDE_FULL = [
    ([0.0, 0.5], [1.2, 0.6]),
    ([0.5, 0.0], [1.2, 0.6]),
    ([0.3, 0.0], [0.0, 1.0]),
    ([-1.0, 0.0], [1.5, 0.0]),
    ([0.0, 0.0], [1.5, 1.5]),
    ([1.0, -1.0], [1.0, 2.0]),
    ([0.0, 0.5], [1.01, 0.5]),
]


@pytest.mark.parametrize("p1_coord, p2_coord", POINTS_DATA_INSIDE_FULL)
def test_is_inside_Tri3n_full(p1_coord, p2_coord):
    # Setup shared data
    nodes = np.array([[1, 0.0, 0.0], [2, 1.0, 0.0], [3, 1.0, 1.0]])
    elements = [[1, "Tri3n", 1, 1, (1, 2, 3)]]

    p1 = np.array(p1_coord)
    p2 = np.array(p2_coord)

    ls = LevelSet()
    ls.gen_from_line_segment(nodes, p1, p2, embedded=False)
    cut = ls.is_cut(elements[0])[0]
    assert CutType.CUT == cut, f"Failed with type: {cut} p1:{p1} p2:{p2}"


POINTS_DATA_INSIDE_FULL_EMBEDDED = [
    ([0.0, 0.5], [1.2, 0.6]),
    ([0.5, -0.1], [1.2, 0.6]),
    ([0.3, -0.1], [0.0, 1.0]),
    ([-0.1, 0.5], [1.1, 0.5]),
]


@pytest.mark.parametrize("p1_coord, p2_coord", POINTS_DATA_INSIDE_FULL_EMBEDDED)
def test_is_inside_Tri3n_full_embedded(p1_coord, p2_coord):
    # Setup shared data
    nodes = np.array([[1, 0.0, 0.0], [2, 1.0, 0.0], [3, 1.0, 1.0]])
    elements = [[1, "Tri3n", 1, 1, (1, 2, 3)]]

    p1 = np.array(p1_coord)
    p2 = np.array(p2_coord)

    ls = LevelSet()
    ls.gen_from_line_segment(nodes, p1, p2, embedded=True)
    cut = ls.is_cut(elements[0])[0]
    assert CutType.CUT == cut, f"Failed with type: {cut} p1:{p1} p2:{p2}"


POINTS_DATA_INSIDE_PARTIAL = [
    ([0.0, 0.5], [1.0, 0.6]),
    ([0.5, 0.0], [1.0, 0.6]),
    ([0.0, -1.0], [0.0, 0.0]),
    ([0.0, 0.5], [0.5, 0.5]),
]


@pytest.mark.parametrize("p1_coord, p2_coord", POINTS_DATA_INSIDE_PARTIAL)
def test_is_inside_Tri3n_partial(p1_coord, p2_coord):
    # Setup shared data
    nodes = np.array([[1, 0.0, 0.0], [2, 1.0, 0.0], [3, 1.0, 1.0]])
    elements = [[1, "Tri3n", 1, 1, (1, 2, 3)]]

    p1 = np.array(p1_coord)
    p2 = np.array(p2_coord)

    ls = LevelSet()
    ls.gen_from_line_segment(nodes, p1, p2, embedded=False)
    cut = ls.is_cut(elements[0])[0]
    assert CutType.PARTIAL == cut, f"Failed with type: {cut} p1:{p1} p2:{p2}"


POINTS_DATA_OUTSIDE = [
    ([-1.0, 0.0], [-1.0, 1.0]),
    ([0.0, -1.0], [0.5, -1.0]),
    ([0.0, 0.5], [0.4, 0.5]),
    ([0.1, 0.2], [0.3, 0.4]),
]


@pytest.mark.parametrize("p1_coord, p2_coord", POINTS_DATA_OUTSIDE)
def test_is_outside_Tri3n(p1_coord, p2_coord):
    nodes = np.array([[1, 0.0, 0.0], [2, 1.0, 0.0], [3, 1.0, 1.0]])
    elements = [[1, "Tri3n", 1, 1, (1, 2, 3)]]

    p1 = np.array(p1_coord)
    p2 = np.array(p2_coord)

    # Test Direction A -> B
    ls = LevelSet()
    ls.gen_from_line_segment(nodes, p1, p2)
    assert CutType.NONE == ls.is_cut(elements[0])[0], (
        f"Failed: segment {p1} to {p2} should be outside."
    )

    ls = LevelSet()
    ls.gen_from_line_segment(nodes, p1, p2, embedded=True)
    assert CutType.NONE == ls.is_cut(elements[0])[0], (
        f"Failed: embedded segment {p1} to {p2} should be outside."
    )
