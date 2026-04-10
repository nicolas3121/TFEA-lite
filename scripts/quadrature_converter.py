# convert quadrature rule from https://people.math.sc.edu/Burkardt/c_src/tetrahedron_arbq_rule/tetrahedron_arbq_rule.c
# to one usable for unit tetrahedron

import numpy as np


def convert_xg_to_unit(xs, ys, zs, ws):
    """
    Converts Xiao-Gimbutas quadrature points from a symmetric reference
    tetrahedron to the standard unit tetrahedron [(0,0,0), (1,0,0), (0,1,0), (0,0,1)].
    """
    # 1. Define Burkardt Reference Tetrahedron Vertices (Equilateral)
    s3 = np.sqrt(3.0)
    s6 = np.sqrt(6.0)

    # 1. Define the Vertices you provided
    # v1 = (-1, -1/s3, -1/s6)
    # v2 = ( 0,  2/s3, -1/s6)
    # v3 = ( 1, -1/s3, -1/s6)
    # v4 = ( 0,  0,    3/s6)
    v = np.array(
        [
            [-1.0, 0.0, 1.0, 0.0],
            [-1.0 / s3, 2.0 / s3, -1.0 / s3, 0.0],
            [-1.0 / s6, -1.0 / s6, -1.0 / s6, 3.0 / s6],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    v_inv = np.linalg.inv(v)

    points_unit = []
    weights_unit = []

    actual_original_volume = np.sum(ws)
    weight_scale = (1.0 / 6.0) / actual_original_volume

    for i in range(len(xs)):
        p_ref = np.array([xs[i], ys[i], zs[i], 1.0])
        lambdas = v_inv @ p_ref

        points_unit.append(lambdas[1:4])
        weights_unit.append(ws[i] * weight_scale)

    return np.array(points_unit), np.array(weights_unit)


# points, weights = convert_xg_to_unit(xs, ys, zs, ws)
# print(
#     np.array_repr(
#         np.array(np.hstack([points, weights.reshape((-1, 1))]), dtype=np.float64),
#         precision=17,
#     )
# )
