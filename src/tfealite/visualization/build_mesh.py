import numpy as np
import pyvista as pv
from pyvista import CellType
from ..core.dofs import DofType, BASE_DOFS, HEAVISIDE_DOFS, BRANCH_DOFS
from ..elements.utils import (
    cut_embedding_tri_iter,
    cut_embedding_tetr_iter,
    partial_cut_embedding_tri_iter,
    partial_cut_embedding_tetr_iter,
    fill_element_displacement,
)
from ..elements.XQuad4n import XQuad4n
from ..elements.Quad4n import Quad4n
from ..elements.XTetr4n import XTetr4n
from ..elements.Tetr4n import Tetr4n
from ..core.level_set import CutType

DOF_TYPES = np.array(
    [
        BASE_DOFS,
        HEAVISIDE_DOFS,
        BRANCH_DOFS,
    ]
)


def _id_to_index(nodes):
    nodes = np.asarray(nodes)
    return {int(nid): i for i, nid in enumerate(nodes[:, 0].astype(int))}


def build_XQuad4n(model, mult=1.0):
    cut_info = model.cut_info

    point_offset = [0]

    def build_triangles(
        tri_iterator,
        Ue,
        nat_x_e,
        elem_vertices,
        phi_t,
        elem,
        points_ref,
        faces,
        displacements,
        stresses,
    ):
        for Ni, _ in tri_iterator:
            centroid = np.mean(Ni, axis=1, keepdims=True)
            Ni = centroid + (Ni - centroid) * 0.999

            sub_nat_x_e = Ni.T @ nat_x_e
            sub_shape_funcs = elem.shape_functions(sub_nat_x_e.T)[0]

            sub_phi_t = np.sum(sub_shape_funcs[:, :4] * phi_t[None, :], axis=1)

            sub_vertices = sub_shape_funcs[:, :4] @ elem_vertices
            node_disps = sub_shape_funcs @ Ue

            in_front = np.where((sub_phi_t > 0) & ~np.isclose(sub_phi_t, 0, atol=1e-4))[
                0
            ]
            if len(in_front) > 0:
                node_disps[in_front, :] = sub_shape_funcs[in_front, :4] @ Ue[:4, :]

            n_points = point_offset[0]
            faces.extend([3, n_points, n_points + 1, n_points + 2])
            point_offset[0] += 3

            points_ref.append(sub_vertices)
            displacements.append(node_disps)

            stresses.append(elem.cal_stresses(sub_nat_x_e.T, Ue))

    def build_quad(Ue, elem_vertices, elem, points_ref, faces, displacements, stresses):
        sub_vertices = elem_vertices

        node_disps = Ue[:4, :]

        n_points = point_offset[0]
        faces.extend([4, n_points, n_points + 1, n_points + 2, n_points + 3])
        point_offset[0] += 4

        points_ref.append(sub_vertices)
        displacements.append(node_disps)

        nat_x_e = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        stresses.append(elem.cal_stresses(nat_x_e[:, 0], nat_x_e[:, 1], Ue))

    points_ref, faces, displacements, node_stress = [], [], [], []

    for elem_id, (_, cut_type, _) in cut_info.items():
        element = model.elements[elem_id - 1]
        _, _, mat_id, real_id, elem_nodes = element
        elem_nodes = np.asarray(elem_nodes)

        elem_dofs = model.list_dof.get_elem_dofs(elem_nodes)
        local_dofs_per_node = np.bitwise_or.reduce(elem_dofs)
        h_enrich = bool(local_dofs_per_node & HEAVISIDE_DOFS)
        t_enrich = bool(local_dofs_per_node & BRANCH_DOFS)
        partial_cut = cut_type == CutType.PARTIAL

        most_enriched_idx = np.argmax(
            np.bitwise_and(elem_dofs, BRANCH_DOFS | HEAVISIDE_DOFS) != 0
        )
        most_enriched_node = elem_nodes[most_enriched_idx]

        ls = model.ls[
            elem_nodes[
                np.argmax(np.bitwise_and(elem_dofs, BRANCH_DOFS | HEAVISIDE_DOFS) != 0)
            ]
            - 1
        ]

        ls = model.ls[most_enriched_node - 1]
        tip = 0

        if t_enrich:
            tip = model.tip[
                elem_nodes[np.argmax(np.bitwise_and(elem_dofs, BRANCH_DOFS) != 0)] - 1
            ]

        phi_n, phi_t = model.level_sets[ls].get(elem_nodes, tip)

        elem_vertices = model.nodes[elem_nodes - 1, 1:3]
        material = model.materials[mat_id - 1][1]
        real = model.reals[real_id - 1][1]
        if model.corrected:
            in_range = model.in_range[elem_nodes - 1]
        else:
            in_range = np.ones(len(elem_nodes))

        Ue = fill_element_displacement(elem_nodes, model.list_dof, model.Ug).reshape(
            (-1, 2)
        )

        elem = XQuad4n(
            elem_vertices,
            material,
            real,
            phi_n,
            phi_t,
            h_enrich,
            t_enrich,
            partial_cut,
            in_range,
        )

        if cut_type == CutType.NONE:
            build_quad(
                Ue, elem_vertices, elem, points_ref, faces, displacements, node_stress
            )
        else:
            Nc1, Nc2 = elem._cal_intersections()

            if partial_cut:
                xi_tip, eta_tip = elem._cal_tip_nat_coords()
                tri1_coords = np.vstack([elem.NAT_1.T, np.ones(3)])
                tip1 = np.linalg.solve(tri1_coords, [xi_tip, eta_tip, 1.0])

                # tri2_coords = np.array([[-1, 1, -1], [-1, 1, 1], [1, 1, 1]])
                tri2_coords = np.vstack([elem.NAT_2.T, np.ones(3)])
                tip2 = np.linalg.solve(tri2_coords, [xi_tip, eta_tip, 1.0])

                sub_tris_1 = partial_cut_embedding_tri_iter(Nc1, tip1, range(4))
                sub_tris_2 = partial_cut_embedding_tri_iter(Nc2, tip2, range(2, 6))
            else:
                sub_tris_1 = cut_embedding_tri_iter(Nc1)
                sub_tris_2 = cut_embedding_tri_iter(Nc2)

            np.array([[-1, -1], [1, -1], [1, 1]])
            np.array([[-1, -1], [1, 1], [-1, 1]])

            build_triangles(
                sub_tris_1,
                Ue,
                elem.NAT_1,
                elem_vertices,
                phi_t,
                elem,
                points_ref,
                faces,
                displacements,
                node_stress,
            )
            build_triangles(
                sub_tris_2,
                Ue,
                elem.NAT_2,
                elem_vertices,
                phi_t,
                elem,
                points_ref,
                faces,
                displacements,
                node_stress,
            )

    points_ref = np.vstack(points_ref) if points_ref else np.empty((0, 2))
    displacements = np.vstack(displacements) if displacements else np.empty((0, 2))

    points_ref = np.hstack((points_ref, np.zeros((points_ref.shape[0], 1))))
    displacements = np.hstack((displacements, np.zeros((displacements.shape[0], 1))))

    points = points_ref + mult * displacements
    mesh = pv.PolyData(points, faces)

    mesh.point_data["points_ref"] = points_ref
    mesh.point_data["displacement"] = displacements

    node_stress = np.vstack(node_stress) if node_stress else np.empty((0, 3))

    if node_stress.shape[0] > 0:
        s_xx, s_yy, t_xy = node_stress[:, 0], node_stress[:, 1], node_stress[:, 2]
        von_mises = np.sqrt(s_xx**2 - s_xx * s_yy + s_yy**2 + 3 * t_xy**2)
        mesh.point_data["s_xx"] = s_xx
        mesh.point_data["s_yy"] = s_yy
        mesh.point_data["t_xy"] = t_xy
        mesh.point_data["von_mises"] = von_mises

    return mesh


def my_build_Quad4n(model, mult=1.0):
    nodes = np.asarray(model.nodes)
    num_nodes = nodes.shape[0]
    num_elems = len(model.elements)

    points_ref = nodes[:, 1:4]
    faces = np.empty((num_elems, 5), dtype=int)
    cell_eids = np.empty(num_elems, dtype=int)
    cell_dofs_per_node = np.empty(num_elems, dtype=int)
    is_enriched = np.zeros(num_elems, dtype=bool)

    displacements = np.zeros_like(points_ref)
    base_dofs = model.list_dof.get_elem_dof_numbers_flat(
        1 + np.arange(num_nodes), BASE_DOFS
    )
    displacements[:, :2] = model.Ug[base_dofs].reshape((-1, 2))

    node_stress = np.zeros((num_nodes, 3))
    node_counts = np.zeros(num_nodes, dtype=int)

    for i, element in enumerate(model.elements):
        eid, _, mat_id, real_id, elem_nodes = element
        elem_nodes_idx = np.asarray(elem_nodes) - 1

        faces[i, 0] = 4
        faces[i, 1:] = elem_nodes_idx

        cell_eids[i] = eid
        elem_dofs = model.list_dof.get_elem_dofs(elem_nodes)
        cell_dofs_per_node[i] = np.bitwise_or.reduce(elem_dofs)

        if eid in model.cut_info:
            is_enriched[i] = True

        if not is_enriched[i]:
            elem_vertices = points_ref[elem_nodes_idx, :2]
            material = model.materials[mat_id - 1][1]
            real = model.reals[real_id - 1][1]

            elem = Quad4n(elem_vertices, material, real)

            Ue = displacements[elem_nodes_idx, :2].flatten()

            sig_nodes = elem.stresses_at_nodes(Ue)

            node_stress[elem_nodes_idx] += sig_nodes
            node_counts[elem_nodes_idx] += 1

    valid_nodes = node_counts > 0
    node_stress[valid_nodes] /= node_counts[valid_nodes, None]

    points = points_ref + mult * displacements
    mesh = pv.PolyData(points, faces.ravel())

    mesh.point_data["points_ref"] = points_ref
    mesh.point_data["displacement"] = displacements

    s_xx, s_yy, t_xy = node_stress[:, 0], node_stress[:, 1], node_stress[:, 2]
    von_mises = np.sqrt(s_xx**2 - s_xx * s_yy + s_yy**2 + 3 * t_xy**2)

    mesh.point_data["s_xx"] = s_xx
    mesh.point_data["s_yy"] = s_yy
    mesh.point_data["t_xy"] = t_xy
    mesh.point_data["von_mises"] = von_mises

    # Bind Cell Data
    mesh.cell_data["eid"] = cell_eids
    mesh.cell_data["dofs_per_node"] = cell_dofs_per_node
    mesh.cell_data["is_enriched"] = is_enriched

    return mesh


def build_XTetr4n(model, mult=1.0):
    cut_info = model.cut_info

    point_offset = [0]

    def build_sub_tetr(
        tetr_iterator,
        Ue,
        nat_x_e,
        elem_vertices,
        phi_t,
        elem,
        points_ref,
        faces,
        displacements,
        stresses,
    ):
        for Ni, *_ in tetr_iterator:
            centroid = np.mean(Ni, axis=1, keepdims=True)
            Ni = centroid + (Ni - centroid) * 0.99

            sub_nat_x_e = Ni.T @ nat_x_e
            sub_shape_funcs = elem.shape_functions(sub_nat_x_e.T)[0]

            sub_phi_t = np.sum(sub_shape_funcs[:, :4] * phi_t[None, :], axis=1)

            sub_vertices = sub_shape_funcs[:, :4] @ elem_vertices
            node_disps = sub_shape_funcs @ Ue

            in_front = np.where((sub_phi_t > 0) & ~np.isclose(sub_phi_t, 0, atol=1e-4))[
                0
            ]
            if len(in_front) > 0:
                node_disps[in_front, :] = sub_shape_funcs[in_front, :4] @ Ue[:4, :]

            n_points = point_offset[0]
            faces.extend([4, n_points, n_points + 1, n_points + 2, n_points + 3])
            point_offset[0] += 4

            points_ref.append(sub_vertices)
            displacements.append(node_disps)

            stresses.append(elem.cal_stresses(sub_nat_x_e.T, Ue))

    def build_tetr(Ue, elem_vertices, elem, points_ref, faces, displacements, stresses):
        sub_vertices = elem_vertices

        node_disps = Ue[: elem.N_FN, :]

        n_points = point_offset[0]
        faces.extend([4, n_points, n_points + 1, n_points + 2, n_points + 3])
        point_offset[0] += 4

        points_ref.append(sub_vertices)
        displacements.append(node_disps)

        nat_x_e = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        stresses.append(elem.cal_stresses(nat_x_e.T, Ue))

    points_ref, faces, displacements, node_stress = [], [], [], []

    for elem_id, (_, cut_type, _) in cut_info.items():
        element = model.elements[elem_id - 1]
        _, _, mat_id, real_id, elem_nodes = element
        elem_nodes = np.asarray(elem_nodes)

        elem_dofs = model.list_dof.get_elem_dofs(elem_nodes)
        local_dofs_per_node = np.bitwise_or.reduce(elem_dofs)
        h_enrich = bool(local_dofs_per_node & HEAVISIDE_DOFS)
        t_enrich = bool(local_dofs_per_node & BRANCH_DOFS)
        partial_cut = cut_type == CutType.PARTIAL

        most_enriched_idx = np.argmax(
            np.bitwise_and(elem_dofs, BRANCH_DOFS | HEAVISIDE_DOFS) != 0
        )
        most_enriched_node = elem_nodes[most_enriched_idx]

        ls = model.ls[most_enriched_node - 1]
        tip = 1

        if t_enrich:
            tip = model.tip[
                elem_nodes[np.argmax(np.bitwise_and(elem_dofs, BRANCH_DOFS) != 0)] - 1
            ]

        phi_n, phi_t = model.level_sets[ls].get(elem_nodes, tip)

        elem_vertices = model.nodes[elem_nodes - 1, 1:4]
        material = model.materials[mat_id - 1][1]
        real = None

        if model.corrected:
            in_range = model.in_range[elem_nodes - 1]
        else:
            in_range = np.ones(len(elem_nodes))

        Ue = fill_element_displacement(elem_nodes, model.list_dof, model.Ug).reshape(
            (-1, 3)
        )

        elem = XTetr4n(
            elem_vertices,
            material,
            real,
            phi_n,
            phi_t,
            h_enrich,
            t_enrich,
            partial_cut,
            in_range,
        )

        if cut_type == CutType.NONE:
            build_tetr(
                Ue, elem_vertices, elem, points_ref, faces, displacements, node_stress
            )
        else:
            Nc, kappa = elem._cal_intersections()

            if partial_cut:
                tip, tip_on_interface = elem._cal_front_intersections()
                sub_tris = partial_cut_embedding_tetr_iter(Nc, tip, tip_on_interface)
            else:
                sub_tris = cut_embedding_tetr_iter(Nc, kappa)

            build_sub_tetr(
                sub_tris,
                Ue,
                elem.NAT_COORDS,
                elem_vertices,
                phi_t,
                elem,
                points_ref,
                faces,
                displacements,
                node_stress,
            )

    points_ref = np.vstack(points_ref) if points_ref else np.empty((0, 3))
    displacements = np.vstack(displacements) if displacements else np.empty((0, 3))

    points = points_ref + mult * displacements

    n_cells = len(faces) // 5
    cell_types = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
    mesh = pv.UnstructuredGrid(faces, cell_types, points)

    mesh.point_data["points_ref"] = points_ref
    mesh.point_data["displacement"] = displacements

    node_stress = np.vstack(node_stress) if node_stress else np.empty((0, 6))

    if node_stress.shape[0] > 0:
        s_xx, s_yy, s_zz = node_stress[:, 0], node_stress[:, 1], node_stress[:, 2]
        t_xy, t_yz, t_xz = node_stress[:, 3], node_stress[:, 4], node_stress[:, 5]

        von_mises = np.sqrt(
            0.5
            * (
                (s_xx - s_yy) ** 2
                + (s_yy - s_zz) ** 2
                + (s_zz - s_xx) ** 2
                + 6 * (t_xy**2 + t_yz**2 + t_xz**2)
            )
        )

        mesh.point_data["s_xx"] = s_xx
        mesh.point_data["s_yy"] = s_yy
        mesh.point_data["s_zz"] = s_zz
        mesh.point_data["t_xy"] = t_xy
        mesh.point_data["t_yz"] = t_yz
        mesh.point_data["t_xz"] = t_xz
        mesh.point_data["von_mises"] = von_mises

    return mesh


def my_build_Tetr4n(model, mult=1.0):
    nodes = np.asarray(model.nodes)
    num_nodes = nodes.shape[0]
    num_elems = len(model.elements)

    points_ref = nodes[:, 1:4]
    faces = np.empty((num_elems, 5), dtype=int)
    cell_eids = np.empty(num_elems, dtype=int)
    cell_dofs_per_node = np.empty(num_elems, dtype=int)
    is_enriched = np.zeros(num_elems, dtype=bool)

    displacements = np.zeros_like(points_ref)
    base_dofs = model.list_dof.get_elem_dof_numbers_flat(
        1 + np.arange(num_nodes), BASE_DOFS
    )
    displacements[:, :3] = model.Ug[base_dofs].reshape((-1, 3))

    node_stress = np.zeros((num_nodes, 6))
    node_counts = np.zeros(num_nodes, dtype=int)

    for i, element in enumerate(model.elements):
        eid, _, mat_id, real_id, elem_nodes = element
        elem_nodes_idx = np.asarray(elem_nodes) - 1

        faces[i, 0] = 4
        faces[i, 1:] = elem_nodes_idx

        cell_eids[i] = eid
        elem_dofs = model.list_dof.get_elem_dofs(elem_nodes)
        cell_dofs_per_node[i] = np.bitwise_or.reduce(elem_dofs)

        if eid in model.cut_info:
            is_enriched[i] = True

        if not is_enriched[i]:
            elem_vertices = points_ref[elem_nodes_idx, :3]
            material = model.materials[mat_id - 1][1]

            real = None

            elem = Tetr4n(elem_vertices, material, real)

            Ue = displacements[elem_nodes_idx, :3].flatten()

            sig_nodes = elem.stresses_at_nodes(Ue)

            node_stress[elem_nodes_idx] += sig_nodes
            node_counts[elem_nodes_idx] += 1

    valid_nodes = node_counts > 0
    node_stress[valid_nodes] /= node_counts[valid_nodes, None]

    points = points_ref + mult * displacements

    cell_types = np.full(num_elems, pv.CellType.TETRA, dtype=np.uint8)
    mesh = pv.UnstructuredGrid(faces.ravel(), cell_types, points)

    mesh.point_data["points_ref"] = points_ref
    mesh.point_data["displacement"] = displacements

    s_xx, s_yy, s_zz = node_stress[:, 0], node_stress[:, 1], node_stress[:, 2]
    t_xy, t_yz, t_xz = node_stress[:, 3], node_stress[:, 4], node_stress[:, 5]

    von_mises = np.sqrt(
        0.5
        * (
            (s_xx - s_yy) ** 2
            + (s_yy - s_zz) ** 2
            + (s_zz - s_xx) ** 2
            + 6 * (t_xy**2 + t_yz**2 + t_xz**2)
        )
    )

    mesh.point_data["s_xx"] = s_xx
    mesh.point_data["s_yy"] = s_yy
    mesh.point_data["s_zz"] = s_zz
    mesh.point_data["t_xy"] = t_xy
    mesh.point_data["t_yz"] = t_yz
    mesh.point_data["t_xz"] = t_xz
    mesh.point_data["von_mises"] = von_mises

    mesh.cell_data["eid"] = cell_eids
    mesh.cell_data["dofs_per_node"] = cell_dofs_per_node
    mesh.cell_data["is_enriched"] = is_enriched

    return mesh


# def my_build_Quad4n(model, node_stress=None):
#     nodes = np.asarray(model.nodes)
#     points_ref = nodes[:, 1:4]
#     faces = []
#     cell_eids = []
#     cell_dofs_per_node = []
#     is_cut = []
#     displacements = np.zeros_like(points_ref)
#     displacements[:, :2] = model.Ug[
#         model.list_dof.get_elem_dof_numbers_flat(
#             1 + np.arange(nodes.shape[0]), BASE_DOFS
#         )
#     ].reshape((-1, 2))
#     for element in model.elements:
#         eid, _, mat_id, real_id, elem_nodes = element
#         elem_nodes = np.asarray(elem_nodes)
#         elem_dofs = model.list_dof.get_elem_dofs(elem_nodes)
#         elem_dofs_per_node = np.bitwise_or.reduce(elem_dofs)
#         faces.append(4)
#         faces += list(elem_nodes - 1)
#         cell_eids.append(eid)
#         cell_dofs_per_node.append(elem_dofs_per_node)
#         is_cut_elem = model.cut_info.get(eid)
#         if is_cut_elem is not None:
#             _, cut_type, _ = is_cut_elem
#             is_cut.append(cut_type == CutType.CUT or cut_type == CutType.PARTIAL)
#         else:
#             is_cut.append(False)
#     points = points_ref + displacements
#     faces_flat = np.array(faces)
#     points_ref = nodes[:, 1:4]
#     mesh = pv.PolyData(points, faces_flat)
#     mesh.point_data["points_ref"] = points_ref
#     mesh.point_data["displacement"] = displacements
#     mesh.cell_data["eid"] = np.asarray(cell_eids, dtype=int)
#     mesh.cell_data["dofs_per_node"] = np.asarray(cell_dofs_per_node)
#     mesh.cell_data["is_cut"] = np.asarray(is_cut)
#     return mesh


def build_Quad4n(nodes, elements, node_stress=None):
    nodes = np.asarray(nodes)
    id2idx = _id_to_index(nodes)
    quad_elems = [e for e in elements if e[1] == "Quad4n"]
    if not quad_elems:
        return None
    faces = []
    cell_eids = []
    for eid, _, _, _, conn in quad_elems:
        conn_idx = [id2idx[int(n)] for n in conn]
        faces.append([4, *conn_idx])
        cell_eids.append(eid)
    faces_flat = np.hstack(faces).astype(np.int64)
    points = nodes[:, 1:4].astype(float)
    mesh = pv.PolyData(points, faces_flat)
    mesh.cell_data["eid"] = np.asarray(cell_eids, dtype=int)
    mesh.cell_data["etype"] = np.array(["Quad4n"] * len(cell_eids), dtype=object)
    if node_stress is not None:
        s = np.asarray(node_stress, dtype=float).ravel()
        if s.size != points.shape[0]:
            raise ValueError(
                f"[build_Quad4n] node_stress length {s.size} != n_nodes {points.shape[0]}"
            )
        mesh.point_data["node_stress"] = s
    return mesh


def build_Tri3n(nodes, elements, node_stress=None):
    nodes = np.asarray(nodes)
    id2idx = _id_to_index(nodes)
    tri_elems = [e for e in elements if e[1] == "Tri3n"]
    if not tri_elems:
        return None
    faces = []
    cell_eids = []
    for eid, _, _, _, conn in tri_elems:
        conn_idx = [id2idx[int(n)] for n in conn]
        faces.append([3, *conn_idx])
        cell_eids.append(eid)
    faces_flat = np.hstack(faces).astype(np.int64)
    points = nodes[:, 1:4].astype(float)
    mesh = pv.PolyData(points, faces_flat)
    mesh.cell_data["eid"] = np.asarray(cell_eids, dtype=int)
    mesh.cell_data["etype"] = np.array(["Tri3n"] * len(cell_eids), dtype=object)
    if node_stress is not None:
        s = np.asarray(node_stress, dtype=float).ravel()
        if s.size != points.shape[0]:
            raise ValueError(
                f"[build_Tri3n] node_stress length {s.size} != n_nodes {points.shape[0]}"
            )
        mesh.point_data["node_stress"] = s
    return mesh


def build_XTri3n(nodes, elements, cut_info, level_sets, node_stress=None):
    nodes = np.asarray(nodes)


def build_mesh(nodes, elements):
    nodes = np.asarray(nodes)

    coords = nodes[:, 1:].astype(float)
    if coords.shape[1] == 2:
        points = np.column_stack((coords, np.zeros(coords.shape[0])))
    else:
        points = coords[:, :3]

    cells_list = []
    cell_eids = []
    cell_types = []

    etype_map = {
        "Tri3n": CellType.TRIANGLE,
        "Quad4n": CellType.QUAD,
        "Tetr4n": CellType.TETRA,
        "Hex8n": CellType.HEXAHEDRON,
    }

    for eid, etype, _, _, conn in elements:
        conn_idx = [int(n) - 1 for n in conn]

        cells_list.extend([len(conn_idx), *conn_idx])
        cell_eids.append(eid)

        if etype in etype_map:
            cell_types.append(etype_map[etype])
        else:
            raise ValueError(f"Unknown element type '{etype}' for element {eid}.")

    cells_array = np.array(cells_list)
    cell_types_array = np.array(cell_types, dtype=np.uint8)

    mesh = pv.UnstructuredGrid(cells_array, cell_types_array, points)
    mesh.cell_data["eid"] = cell_eids

    return mesh


def build_Tetr4n(nodes, elements, node_stress=None):
    nodes = np.asarray(nodes)
    id2idx = _id_to_index(nodes)
    points = nodes[:, 1:4].astype(float)
    cells_list = []
    cell_eids = []
    for eid, etype, _, _, conn in elements:
        if etype != "Tetr4n":
            continue
        conn_idx = [id2idx[int(n)] for n in conn]
        cells_list.extend([4, *conn_idx])
        cell_eids.append(eid)
    if not cells_list:
        return None
    cells = np.asarray(cells_list, dtype=np.int64)
    n_cells = len(cell_eids)
    VTK_TETRA = getattr(pv.CellType, "TETRA", 10)
    celltypes = np.full(n_cells, VTK_TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, points)
    grid.cell_data["eid"] = np.asarray(cell_eids, dtype=int)
    grid.cell_data["etype"] = np.array(["Tetr4n"] * n_cells, dtype=object)
    if node_stress is not None:
        s = np.asarray(node_stress, dtype=float).ravel()
        if s.size != points.shape[0]:
            raise ValueError(
                f"[build_Tetr4n] node_stress length {s.size} != n_nodes {points.shape[0]}"
            )
        grid.point_data["node_stress"] = s
    return grid


def build_load_arrows(
    nodes,
    Fg,
    list_dof,
    load_size=(1.0, 1.0),
    min_mag=1e-3,
):
    arrow_amp, arrow_scale = load_size
    nodes = np.asarray(nodes, dtype=float)
    Fg = np.asarray(Fg, dtype=float)
    mesh_load = pv.PolyData()
    for _, nd in enumerate(nodes):
        nid = int(nd[0])
        fx = fy = fz = 0.0
        node_dof_number = list_dof.get(nid, DofType.UX)
        if node_dof_number is not None:
            fx = Fg[node_dof_number] * arrow_amp * arrow_scale
        node_dof_number = list_dof.get(nid, DofType.UY)
        if node_dof_number is not None:
            fy = Fg[node_dof_number] * arrow_amp * arrow_scale
        node_dof_number = list_dof.get(nid, DofType.UZ)
        if node_dof_number is not None:
            fz = Fg[node_dof_number] * arrow_amp * arrow_scale
        mag_f = np.sqrt(fx * fx + fy * fy + fz * fz)
        if mag_f <= min_mag:
            continue
        x, y, z = nd[1:4]
        start_pt = [x - fx, y - fy, z - fz]
        dirn = [fx / mag_f, fy / mag_f, fz / mag_f]
        arrow = pv.Arrow(
            start=start_pt,
            direction=dirn,
            tip_length=0.25 / mag_f * arrow_scale,
            tip_radius=0.1 / mag_f * arrow_scale,
            tip_resolution=20,
            shaft_radius=0.03 / mag_f * arrow_scale,
            shaft_resolution=20,
            scale=mag_f,
        )
        mesh_load = mesh_load.merge(arrow)
    if mesh_load.n_points == 0:
        return None
    return mesh_load
