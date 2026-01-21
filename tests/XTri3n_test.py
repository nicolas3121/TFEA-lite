import numpy as np
from tfealite.elements.XTri3n import XTri3n


def test_rigid_body_modes_fully_cut():
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    material = {"E": 1, "nu": 0.3, "rho": 1}
    real = {"t": 1}

    tri = XTri3n(
        nodes,
        np.array([-1, -1, 1]),
        np.array([-1, -1, 1]),
        True,
        False,
        False,
        material,
        real,
    )
    Ke = tri.cal_element_matrices(eval_mass=False)
    eigenvalues = np.abs(np.linalg.eigvals(Ke))

    # Sort eigenvalues to find the smallest ones
    eigenvalues.sort()
    print(eigenvalues)

    assert np.allclose(Ke - Ke.T, 0), "Not symmetric"
    assert np.allclose(eigenvalues[:6], 0), "Less than 6 zero eigenvalues"
    assert not np.any(np.isclose(eigenvalues[7:], 0)), "More than 6 zero eigenvalues"
    # assert False


# def test_partial_cut():
#     nodes = np.array([[0, 0], [1, 0], [0, 1]])
#     material = {"E": 1, "nu": 0.3, "rho": 1}
#     phi_n = np.array([-1, -1, 1])
#     phi_t = np.array([-1, 1, -1])
#     real = {"t": 1}
#     singularity = (
#         np.linalg.solve(np.array([phi_t, phi_n, [1, 1, 1]]), np.array([0, 0, 1]))
#         @ nodes
#     )
#     centered_nodes = nodes - singularity
#     # integratie in meerdere functies opsplitsen zodat appart kan testen
#
#     tri = XTri3n(
#         nodes,
#         np.array([-1, -1, 1]),
#         np.array([-1, 1, -1]),
#         False,
#         True,
#         True,
#         material,
#         real,
#     )
#     Ke = tri.cal_element_matrices()
#     # # xi = np.array([0.1, 0.2])
#     # Ke = tri.shape_functions(xi, xi)
#     print(Ke)
#     assert False
