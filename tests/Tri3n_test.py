import numpy as np
from tfealite.elements.Tri3n import Tri3n


def test_rigid_body_modes():
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    material = {"E": 1, "nu": 0.3, "rho": 1}
    real = {"t": 1}

    tri = Tri3n(nodes, material, real)
    Ke = tri.cal_element_matrices(eval_mass=False)
    eigenvalues = np.abs(np.linalg.eigvals(Ke))

    # Sort eigenvalues to find the smallest ones
    eigenvalues.sort()

    assert np.allclose(eigenvalues[:3], 0)
    assert not np.any(np.isclose(eigenvalues[3:], 0))


def test_mass_matrix():
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    material = {"E": 1, "nu": 0.3, "rho": 1}
    real = {"t": 1}

    tri = Tri3n(nodes, material, real)
    Me, _ = tri.cal_element_matrices(eval_mass=True)

    calculated_mass = np.sum(Me[::2, ::2])

    coords = tri.node_coords
    area = 0.5 * abs(
        coords[0, 0] * (coords[1, 1] - coords[2, 1])
        + coords[1, 0] * (coords[2, 1] - coords[0, 1])
        + coords[2, 0] * (coords[0, 1] - coords[1, 1])
    )

    expected_mass = area * tri.t * tri.rho

    assert np.isclose(calculated_mass, expected_mass), "Mass Matrix error!"
