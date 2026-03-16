import numpy as np
from tfealite.elements.XQuad4n import XQuad4n


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
