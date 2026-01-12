import numpy as np
from tfealite.elements.XTri3n import XTri3n


def test_rigid_body_modes():
    nodes = np.array([[0.5, 0], [4, 0], [0, 12]])
    material = {"E": 1, "nu": 0.3, "rho": 1}
    real = {"t": 1}

    tri = XTri3n(nodes, np.array([-1, -1, 1]), None, True, False, material, real)
    Ke = tri.cal_element_matrices(eval_mass=False)
    eigenvalues = np.abs(np.linalg.eigvals(Ke))

    # Sort eigenvalues to find the smallest ones
    eigenvalues.sort()
    print(eigenvalues)

    assert np.allclose(Ke - Ke.T, 0), "Not symmetric"
    assert np.allclose(eigenvalues[:7], 0), "Less than 7 zero eigenvalues"
    assert not np.any(np.isclose(eigenvalues[7:], 0)), "More than 7 zero eigenvalues"
