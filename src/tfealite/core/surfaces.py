import numpy as np
from geomdl import BSpline, utilities
from scipy.interpolate import NdBSpline


def init_crack_plane_geomdl(
    width,
    length,
    rotation=None,
    translation=None,
    num_u_splines=5,
    num_v_pts=5,
    embedded=False,
):
    """
    if edge crack origin rotation at center of crack front
    if embedded crack origin rotation at center of crack
    """
    surfaces = []
    if rotation is None:
        rotation = np.eye(3)
    if translation is None:
        translation = np.zeros(3)

    x_vals = np.linspace(-width / 2.0, width / 2.0, num_v_pts)

    if not embedded:
        y_vals = np.linspace(-length, 0, num_u_splines)

        Y, X = np.meshgrid(y_vals, x_vals, indexing="ij")
        Z = np.zeros_like(X)

        ctrlpts2d = (
            np.stack((X, Y, Z), axis=-1) @ rotation.T[None, :, :]
            + translation[None, None, :]
        )

        surf = BSpline.Surface()
        surf.degree_u = 2
        surf.degree_v = 2
        surf.ctrlpts2d = ctrlpts2d.tolist()
        surf.knotvector_u = utilities.generate_knot_vector(
            surf.degree_u, surf.ctrlpts_size_u
        )
        surf.knotvector_v = utilities.generate_knot_vector(
            surf.degree_v, surf.ctrlpts_size_v
        )
        surfaces.append((True, False, surf))

    else:
        y1_vals = np.linspace(0, -length / 2.0, num_u_splines)

        Y1, X1 = np.meshgrid(y1_vals, x_vals, indexing="ij")
        Z = np.zeros_like(X1)

        ctrlpts2d_1 = (
            np.stack((X1, Y1, Z), axis=-1) @ rotation.T[None, :, :]
            + translation[None, None, :]
        ).tolist()

        ctrlpts2d_2 = (
            np.stack((-X1, -Y1, Z), axis=-1) @ rotation.T[None, :, :]
            + translation[None, None, :]
        ).tolist()

        for ctrlpts in [ctrlpts2d_1, ctrlpts2d_2]:
            surf = BSpline.Surface()
            surf.degree_u = 2
            surf.degree_v = 2
            surf.ctrlpts2d = ctrlpts
            surf.knotvector_u = utilities.generate_knot_vector(
                surf.degree_u, surf.ctrlpts_size_u
            )
            surf.knotvector_v = utilities.generate_knot_vector(
                surf.degree_v, surf.ctrlpts_size_v
            )
            surfaces.append((True, False, surf))

    return surfaces


def init_half_coin_crack_geomdl(
    radius=1.0,
    rotation=None,
    translation=None,
    num_u_splines=50,
    num_v_pts=30,
):
    if rotation is None:
        rotation = np.eye(3)
    if translation is None:
        translation = np.zeros(3)
    """
    crack runs in positive y direction
    at base from -r / 2 -> r / 2 in x direction
    origin for rotation in middle of base line (u = 0)
    """
    surf = BSpline.Surface()
    surf.degree_u = 2
    surf.degree_v = 2

    angles = np.linspace(0, np.pi, num_v_pts)
    radii = np.linspace(
        1e-6, radius, num_u_splines
    )  # tiny hole at center otherwise derivatives undefined there
    x_pts = np.cos(angles)[None, :] * radii[:, None]
    y_pts = np.sin(angles)[None, :] * radii[:, None]
    z_pts = np.zeros_like(x_pts)
    ctrlpts2d = (
        np.stack([x_pts, y_pts, z_pts], axis=-1) @ rotation.T[None, :, :]
        + translation[None, None, :]
    ).tolist()

    surf.ctrlpts2d = ctrlpts2d
    surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, num_u_splines)
    surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, num_v_pts)

    return [(True, True, surf)]


def geomdl_to_NdBSplines(surfaces):
    return [
        (
            active,
            has_pole,
            NdBSpline(
                t=(surf.knotvector_u, surf.knotvector_v), c=surf.ctrlpts2d, k=(2, 2)
            ),
        )
        for (active, has_pole, surf) in surfaces
    ]
