import numpy as np
from tfealite.elements.XQuad4n import XQuad4n
import time


def test_rigid_body_modes_fully_cut():
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    material = {"E": 1, "nu": 0.3, "rho": 1}
    real = {"t": 1}
    quad = XQuad4n(
        nodes,
        material,
        real,
        np.array([-1, -1, 1, 1]),
        np.array([-1, -1, 1, 1]),
        True,
        False,
        False,
    )
    Ke = quad.cal_element_matrices(eval_mass=False)
    eigenvalues = np.abs(np.linalg.eigvals(Ke))

    # Sort eigenvalues to find the smallest ones
    eigenvalues.sort()
    print(eigenvalues)

    assert np.allclose(Ke - Ke.T, 0), "Not symmetric"
    assert np.allclose(eigenvalues[:6], 0), "Less than 6 zero eigenvalues"
    assert not np.any(np.isclose(eigenvalues[7:], 0)), "More than 6 zero eigenvalues"


def test_array_shape_fn():
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    material = {"E": 1, "nu": 0.3, "rho": 1}
    real = {"t": 1}
    quad = XQuad4n(
        nodes,
        material,
        real,
        np.array([-1, -1, 1, 1]),
        np.array([-2, -1, -1, -2]),
        True,
        True,
        False,
    )
    N, dN_dxi = quad.shape_functions2(
        np.array([0.3, -0.1, 0.5, 0.7]), np.array([0.3, 0.0, -0.3, -0.1])
    )
    N01, dN_dxi01 = quad.shape_functions(0.3, 0.3)
    N02, dN_dxi02 = quad.shape_functions(-0.1, 0.0)
    N03, dN_dxi03 = quad.shape_functions(0.5, -0.3)
    assert np.all(N01 == N[0, :])
    assert np.all(N02 == N[1, :])
    assert np.all(N03 == N[2, :])
    assert np.all(dN_dxi01 == dN_dxi[0])
    assert np.all(dN_dxi02 == dN_dxi[1])
    assert np.all(dN_dxi03 == dN_dxi[2])
    quad = XQuad4n(
        nodes,
        material,
        real,
        np.array([-1, -1, 1, 1]),
        np.array([-2, -1, -1, -2]),
        True,
        True,
        False,
        h_enrich_per_node=np.array([1, 0, 0, 1]),
    )
    # N, dN_dxi = quad.shape_functions2(np.array([0.3]), np.array([0.3]))
    N, dN_dxi = quad.shape_functions2(
        np.array([0.3, -0.1, 0.5, 0.7]), np.array([0.3, 0.0, -0.3, -0.1])
    )
    N01, dN_dxi01 = quad.shape_functions(0.3, 0.3)
    N02, dN_dxi02 = quad.shape_functions(-0.1, 0.0)
    N03, dN_dxi03 = quad.shape_functions(0.5, -0.3)
    print(np.isclose(N01, N[0, :], atol=10e-16))
    assert np.all(np.isclose(N01, N[0, :], atol=10e-16))
    assert np.all(N02 == N[1, :])
    assert np.all(N03 == N[2, :])
    assert np.all(dN_dxi01 == dN_dxi[0])
    assert np.all(dN_dxi02 == dN_dxi[1])
    assert np.all(dN_dxi03 == dN_dxi[2])


def test_array_stiffness_mat():
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    material = {"E": 1, "nu": 0.3, "rho": 1}
    real = {"t": 1}
    quad = XQuad4n(
        nodes,
        material,
        real,
        np.array([-1, -1, 1, 1]),
        np.array([-1, 1, 1, -1]),
        False,
        True,
        True,
    )
    start = time.perf_counter()
    old_heaviside = quad.cal_element_matrices(eval_mass=False)
    end = time.perf_counter()
    print(end - start)
    start = time.perf_counter()
    new_heaviside = quad.cal_element_matrices2(eval_mass=False)
    end = time.perf_counter()
    print(end - start)
    # diff = old_heaviside - new_heaviside
    # print(diff)
    assert np.all(np.isclose(old_heaviside, new_heaviside, atol=10e-18))
    quad = XQuad4n(
        nodes,
        material,
        real,
        np.array([-1, -1, 1, 1]),
        np.array([-2, -1, -1, -2]),
        True,
        True,
        False,
    )
    old_heaviside = quad.cal_element_matrices(eval_mass=False)
    start = time.perf_counter()
    old_heaviside = quad.cal_element_matrices(eval_mass=False)
    end = time.perf_counter()
    print(end - start)
    start = time.perf_counter()
    new_heaviside = quad.cal_element_matrices2(eval_mass=False)
    end = time.perf_counter()
    print(end - start)
    # print(diff[:, 8:])
    assert np.all(np.isclose(old_heaviside, new_heaviside, atol=10e-18))
    # assert False
    quad = XQuad4n(
        nodes,
        material,
        real,
        np.array([-1, -1, 1, 1]),
        np.array([-2, -1, -1, -2]),
        True,
        True,
        False,
        h_enrich_per_node=np.array([1, 0, 0, 1]),
    )
    old_heaviside = quad.cal_element_matrices(eval_mass=False)
    start = time.perf_counter()
    old_heaviside = quad.cal_element_matrices(eval_mass=False)
    end = time.perf_counter()
    print(end - start)
    start = time.perf_counter()
    new_heaviside = quad.cal_element_matrices2(eval_mass=False)
    end = time.perf_counter()
    print(end - start)
    # print(diff[:, 8:])
    assert np.all(np.isclose(old_heaviside, new_heaviside, atol=10e-18))
    quad = XQuad4n(
        nodes,
        material,
        real,
        np.array([-1, -1, 1, 1]),
        np.array([2, 1, 1, 2]),
        False,
        True,
        False,
    )
    old_heaviside = quad.cal_element_matrices(eval_mass=False)
    start = time.perf_counter()
    old_heaviside = quad.cal_element_matrices(eval_mass=False)
    end = time.perf_counter()
    print(end - start)
    start = time.perf_counter()
    new_heaviside = quad.cal_element_matrices2(eval_mass=False)
    end = time.perf_counter()
    print(end - start)
    # print(diff[:, 8:])
    assert np.all(np.isclose(old_heaviside, new_heaviside, atol=10e-18))


def test_shape_tip_enrichment():
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    material = {"E": 1, "nu": 0.3, "rho": 1}
    real = {"t": 1}
    quad = XQuad4n(
        nodes,
        material,
        real,
        np.array([-1, -1, 1, 1]),
        np.array([-2, -1, -1, -2]),
        True,
        True,
        False,
        h_enrich_per_node=np.array([0, 0, 0, 0]),
    )
    xi = np.zeros(99) - 1
    eta = np.linspace(-1, 1, 99)

    N, dN_dxi = quad.shape_functions2(xi, eta)
    N.tofile("enrichment_data")
    assert False
