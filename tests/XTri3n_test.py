import numpy as np
import sympy as sp
from tfealite.elements.XTri3n import XTri3n
from tfealite.elements.XTri3n import (
    GeneralizedDuffy,
    DuffyDistance,
    DuffySinh,
)
import tfealite.core.quadratures as qd


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


def integrate_singular_triangle(power):
    r, theta = sp.symbols("r theta", positive=True)

    f_polar = (1 / r ** sp.Rational(power, 2)) * sp.cos(theta / 2)

    integrand = f_polar * r

    r_limit = 1 / (sp.cos(theta) + sp.sin(theta))

    result = sp.integrate(integrand, (r, 0, r_limit), (theta, 0, sp.pi / 2))
    # print(result.evalf())
    return float(result.evalf())


def test_generalized_duffy_transform():
    # correct = integrate_singular_triangle(2)
    correct = 1.11945669664310
    (rule, correction) = qd.QUAD_RULES[10]
    rule = rule.copy()
    rule[:, 0:2] = (1 + rule[:, 0:2]) / 2
    rule[:, 2] /= 4
    nodes = np.array([[0, 0], [1, 0], [0, 1]])

    duffy = GeneralizedDuffy(nodes)

    [xi_points, eta_points, j_ddt] = duffy.transform(rule[:, 0], rule[:, 1], 1)

    def fun(xi, eta):
        return 1 / np.sqrt(xi**2 + eta**2) * np.cos(np.atan2(eta, xi) / 2)

    fun_vals = fun(xi_points, eta_points)
    duffy_numeric = np.sum(fun_vals * rule[:, 2] * j_ddt * correction)

    assert np.isclose(duffy_numeric, correct), "1 / r"

    # correct2 = integrate_singular_triangle(3)
    correct2 = 2.51316902334288
    [xi_points, eta_points, j_ddt] = duffy.transform(rule[:, 0], rule[:, 1], 2)

    def fun2(xi, eta):
        return 1 / np.sqrt(xi**2 + eta**2) ** (3 / 2) * np.cos(np.atan2(eta, xi) / 2)

    fun_vals = fun2(xi_points, eta_points)
    duffy_numeric = np.sum(fun_vals * rule[:, 2] * j_ddt * correction)
    assert np.isclose(duffy_numeric, correct2), "1 / r^(3 / 2)"
    # assert False


def test_duffy_distance_transform():
    # correct = integrate_singular_triangle(2)
    correct = 1.11945669664310
    (rule, correction) = qd.QUAD_RULES[10]
    rule = rule.copy()
    rule[:, 0:2] = (1 + rule[:, 0:2]) / 2
    rule[:, 2] /= 4
    nodes = np.array([[0, 0], [1, 0], [0, 1]])

    duffy = DuffyDistance(nodes)

    [xi_points, eta_points, j_ddt] = duffy.transform(rule[:, 0], rule[:, 1], 1)

    def fun(xi, eta):
        return 1 / np.sqrt(xi**2 + eta**2) * np.cos(np.atan2(eta, xi) / 2)

    fun_vals = fun(xi_points, eta_points)
    duffy_numeric = np.sum(fun_vals * rule[:, 2] * j_ddt * correction)

    assert np.isclose(duffy_numeric, correct), "1 / r"

    # correct2 = integrate_singular_triangle(3)
    correct2 = 2.51316902334288
    [xi_points, eta_points, j_ddt] = duffy.transform(rule[:, 0], rule[:, 1], 2)

    def fun2(xi, eta):
        return 1 / np.sqrt(xi**2 + eta**2) ** (3 / 2) * np.cos(np.atan2(eta, xi) / 2)

    fun_vals = fun2(xi_points, eta_points)
    duffy_numeric = np.sum(fun_vals * rule[:, 2] * j_ddt * correction)
    assert np.isclose(duffy_numeric, correct2), "1 / r^(3 / 2)"


def test_duffy_sinh_transform():
    # correct = integrate_singular_triangle(2)
    correct = 1.11945669664310
    (rule, correction) = qd.QUAD_RULES[10]
    rule = rule.copy()
    rule[:, 0:2] = (1 + rule[:, 0:2]) / 2
    rule[:, 2] /= 4
    nodes = np.array([[0, 0], [1, 0], [0, 1]])

    duffy = DuffySinh(nodes)

    [xi_points, eta_points, j_ddt] = duffy.transform(rule[:, 0], rule[:, 1], 1)

    def fun(xi, eta):
        return 1 / np.sqrt(xi**2 + eta**2) * np.cos(np.atan2(eta, xi) / 2)

    fun_vals = fun(xi_points, eta_points)
    duffy_numeric = np.sum(fun_vals * rule[:, 2] * j_ddt * correction)

    assert np.isclose(duffy_numeric, correct), "1 / r"

    # correct2 = integrate_singular_triangle(3)
    correct2 = 2.51316902334288
    [xi_points, eta_points, j_ddt] = duffy.transform(rule[:, 0], rule[:, 1], 2)

    def fun2(xi, eta):
        return 1 / np.sqrt(xi**2 + eta**2) ** (3 / 2) * np.cos(np.atan2(eta, xi) / 2)

    fun_vals = fun2(xi_points, eta_points)
    duffy_numeric = np.sum(fun_vals * rule[:, 2] * j_ddt * correction)
    assert np.isclose(duffy_numeric, correct2), "1 / r^(3 / 2)"


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
#     print("singularity", singularity)
#     # centered_nodes = nodes - singularity
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
