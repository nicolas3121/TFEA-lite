import numpy as np
import sympy as sp
from tfealite.elements.XTetr4n import XTetr4n
from tfealite.core.quadratures import DuffySinh3D


def test_rigid_body_modes_fully_cut():
    nodes = np.array(
        [[1, 0.0, 0.0, 0.0], [2, 1.0, 0.0, 0.0], [3, 0.0, 1.0, 0.0], [4, 0.0, 0.0, 1.0]]
    )
    material = {"E": 1, "nu": 0.3, "rho": 1}
    real = {}
    quad = XTetr4n(
        nodes[:, 1:],
        material,
        real,
        np.array([-1, -1, -1, 1]),
        np.array([-1, -1, -1, -1]),
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
    assert np.allclose(eigenvalues[:12], 0), "Less than 6 zero eigenvalues"
    assert not np.any(np.isclose(eigenvalues[12:], 0)), "More than 6 zero eigenvalues"
    # assert False


# def test_tip_enrichment():
#     nodes = np.array(
#         [[1, 0.0, 0.0, 0.0], [2, 1.0, 0.0, 0.0], [3, 0.0, 1.0, 0.0], [4, 0.0, 0.0, 1.0]]
#     )
#     elements = [[1, "Tetr4n", 1, 1, (1, 2, 3, 4)]]
#     p1 = np.array([-1.0, 0.0, 0.5])
#     p2 = np.array([0.5, 0.0, 0.2])
#     p3 = np.array([0.5, 2.0, 0.2])
#
#     ls = LevelSet()
#     ls.gen_from_plane(nodes, p1, p2, p3, embedded=False)
#
#     cut = ls.is_cut(elements[0])[0]
#     print(cut)
#     material = {"E": 1, "nu": 0.3, "rho": 1}
#     real = {}
#     quad = XTetr4n(
#         nodes[:, 1:],
#         material,
#         real,
#         ls.phi_n,
#         ls.phi_t,
#         False,
#         True,
#         True,
#     )
#     quad.cal_element_matrices(eval_mass=False)
#     # assert False


def sympy_edge_singularity_tilted(nodes_phys):
    """
    Analytically integrates 1/rho using Y-Z polar coordinates.
    Handles tetrahedra where Nodes 3 and 4 have non-zero x-coordinates.
    """
    # 1. Setup local coordinates (Node 1 at origin)
    x1, y1, z1 = nodes_phys[0]
    L = nodes_phys[1, 0] - x1

    # Relative coordinates of Nodes 3 and 4
    # Note: x3 and x4 are extracted but will mathematically cancel in the width
    _, y3, z3 = nodes_phys[2, 0] - x1, nodes_phys[2, 1] - y1, nodes_phys[2, 2] - z1
    _, y4, z4 = nodes_phys[3, 0] - x1, nodes_phys[3, 1] - y1, nodes_phys[3, 2] - z1

    rho, phi = sp.symbols("rho phi", positive=True)

    # 2. Map reference coordinates (eta, zeta) to physical Y-Z
    M = sp.Matrix([[y3, y4], [z3, z4]])
    eta_zeta = M.inv() * sp.Matrix([rho * sp.cos(phi), rho * sp.sin(phi)])

    # 3. Robust Angular Sweep
    phi_3 = sp.atan2(z3, y3)
    phi_4 = sp.atan2(z4, y4)

    # Ensure we take the small interior angle of the tetrahedron
    diff = phi_4 - phi_3
    if diff > sp.pi:
        phi_4 -= 2 * sp.pi
    elif diff < -sp.pi:
        phi_4 += 2 * sp.pi

    phi_start, phi_end = (phi_3, phi_4) if phi_3 < phi_4 else (phi_4, phi_3)

    # 4. Limits for rho (where eta + zeta = 1)
    rho_max = 1 / sp.simplify((eta_zeta[0] + eta_zeta[1]) / rho)

    # 5. The Axial Width (x_len)
    # Even with x3 and x4, for a fixed (eta, zeta), the x-range
    # of the tetrahedron is always L * (1 - eta - zeta).
    x_len = L * (1 - eta_zeta[0] - eta_zeta[1])

    # 6. Integrand: (1/rho) * Jacobian_rho * x_len
    # dV = dx * (rho * drho * dphi)  => Integrand = (1/rho) * rho * x_len
    integrand = sp.simplify(x_len)

    # 7. Integrate
    int_rho = sp.integrate(integrand, (rho, 0, rho_max))
    result = sp.integrate(int_rho, (phi, phi_start, phi_end))

    # Strip ghost imaginary parts and return
    return float(sp.re(result.evalf()))


def get_3d_gauss_rule(order=10):
    """Generates a 3D Gauss grid mapped to [0, 1]^3."""
    x1d, w1d = np.polynomial.legendre.leggauss(order)
    x1d = (x1d + 1) / 2
    w1d = w1d / 2

    U, V, W = np.meshgrid(x1d, x1d, x1d, indexing="ij")
    WU, WV, WW = np.meshgrid(w1d, w1d, w1d, indexing="ij")

    coords = np.vstack([U.ravel(), V.ravel(), W.ravel()])
    weights = (WU * WV * WW).ravel()
    return coords, weights


def test_duffy_3d_edge_singularity():
    print("--- Testing Edge Singularity (1 / rho) ---")

    # 1. Define geometry (Node 1-3-4 in Y-Z plane, Edge 1-2 on X axis)
    nodes_phys = np.array(
        [
            [0.0, 0.0, 0.0],  # Node 1
            [2.0, 0.0, 0.0],  # Node 2
            [0.0, 1.5, 0.2],  # Node 3
            [0.0, 0.5, 1.8],  # Node 4
        ]
    )

    # 2. Get Ground Truth
    correct = sympy_edge_singularity_tilted(nodes_phys)

    # 3. Setup Quadrature
    nat_coords, weights = get_3d_gauss_rule(order=2)

    # 4. Initialize Duffy
    duffy = DuffySinh3D(nodes_phys)

    # 5. Transform points (beta1=1, beta2=1 perfectly resolves 1/rho)
    xi, eta, zeta, j_ddt = duffy.transform(nat_coords, beta1=1, beta2=1)

    # 6. Map Natural Points to Physical Space
    # x_phys = nodes_phys[1, 0] * xi
    y_phys = nodes_phys[2, 1] * eta + nodes_phys[3, 1] * zeta
    z_phys = nodes_phys[2, 2] * eta + nodes_phys[3, 2] * zeta

    # 7. Evaluate Integrand: 1 / rho
    rho_phys = np.sqrt(y_phys**2 + z_phys**2)
    f_vals = 1.0 / rho_phys

    # 8. Get Physical Jacobian (Constant for linear tet)
    detJ_phys = np.abs(np.linalg.det(nodes_phys[1:] - nodes_phys[0]))

    # 9. Perform Numerical Integration
    numeric_result = np.sum(f_vals * weights * j_ddt * detJ_phys)

    print(f"Exact:   {correct:.15f}")
    print(f"Numeric: {numeric_result:.15f}")
    print(f"Error:   {abs(numeric_result - correct):.2e}\n")
    assert np.isclose(numeric_result, correct, atol=1e-14), "Failed on edge singularity"


def test_duffy_3d_edge_singularity_tilted():
    print("--- Testing Edge Singularity (1 / rho) ---")

    # 1. Define geometry (Node 1-3-4 in Y-Z plane, Edge 1-2 on X axis)
    nodes_phys = np.array(
        [
            [0.0, 0.0, 0.0],  # Node 1
            [2.0, 0.0, 0.0],  # Node 2
            [-0.3, 1.5, 0.2],  # Node 3
            [0.1, 0.5, 1.8],  # Node 4
        ]
    )

    # 2. Get Ground Truth
    correct = sympy_edge_singularity_tilted(nodes_phys)

    # 3. Setup Quadrature
    nat_coords, weights = get_3d_gauss_rule(order=2)

    # 4. Initialize Duffy
    duffy = DuffySinh3D(nodes_phys)

    # 5. Transform points (beta1=1, beta2=1 perfectly resolves 1/rho)
    _, eta, zeta, j_ddt = duffy.transform(nat_coords, beta1=1, beta2=1)

    # 6. Map Natural Points to Physical Space
    # x_phys = nodes_phys[1, 0] * xi
    y_phys = nodes_phys[2, 1] * eta + nodes_phys[3, 1] * zeta
    z_phys = nodes_phys[2, 2] * eta + nodes_phys[3, 2] * zeta

    # 7. Evaluate Integrand: 1 / rho
    rho_phys = np.sqrt(y_phys**2 + z_phys**2)
    f_vals = 1.0 / rho_phys

    # 8. Get Physical Jacobian (Constant for linear tet)
    detJ_phys = np.abs(np.linalg.det(nodes_phys[1:] - nodes_phys[0]))

    # 9. Perform Numerical Integration
    numeric_result = np.sum(f_vals * weights * j_ddt * detJ_phys)

    print(f"Exact:   {correct:.15f}")
    print(f"Numeric: {numeric_result:.15f}")
    print(f"Error:   {abs(numeric_result - correct):.2e}\n")
    assert np.isclose(numeric_result, correct, atol=1e-14), "Failed on edge singularity"
