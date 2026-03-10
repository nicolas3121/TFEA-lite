import numpy as np
import pyvista as pv
from ..core.dofs import DofType, BASE_DOFS, HEAVISIDE_DOFS, BRANCH_DOFS
from ..elements.XQuad4n import XQuad4n
from ..core.level_set import CutType
import itertools

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


def build_XQuad4n(model, node_stress=None):
    cut_info = model.cut_info
    n_nodes = 4
    points_ref = []
    displacements = []

    faces = []

    # kan mogelijk extra set barycentrische coordinates gebruiken voor binnen mijn nieuwe driehoek
    # teken bekijken volgende, vorige n
    def build_triangles(iter, Ue, nat_x_e, elem_vertices):
        for Ni, _ in iter:
            centroid = np.mean(Ni, axis=1)
            Ni = centroid[:, None] + (Ni - centroid[:, None]) * 0.99999
            sub_nat_x_e = Ni.T @ nat_x_e
            sub_shape_functions = elem.shape_functions2(
                sub_nat_x_e[:, 0], sub_nat_x_e[:, 1]
            )[0]
            sub_vertices = sub_shape_functions[:, :4] @ elem_vertices

            displacements.append(sub_shape_functions @ Ue)
            n_points = 3 * len(points_ref)
            faces.extend([3, n_points, n_points + 1, n_points + 2])
            points_ref.append(sub_vertices)

    for elem_id, (_, cut_type, _) in cut_info.items():
        element = model.elements[elem_id - 1]
        _, _, mat_id, real_id, elem_nodes = element
        elem_nodes = np.asarray(elem_nodes)
        elem_dofs = model.list_dof.get_elem_dofs(elem_nodes)
        local_dofs_per_node = np.bitwise_or.reduce(elem_dofs)
        h_enrich = local_dofs_per_node & HEAVISIDE_DOFS != 0
        t_enrich = local_dofs_per_node & BRANCH_DOFS != 0
        if cut_type == CutType.NONE:
            continue
        partial_cut = cut_type == CutType.PARTIAL
        most_enriched_node = elem_nodes[
            np.argmax(np.bitwise_and(elem_dofs, BRANCH_DOFS | HEAVISIDE_DOFS) != 0)
        ]
        ls = model.ls[most_enriched_node - 1]
        tip = model.tip[most_enriched_node - 1]
        # print("ls", ls, "tip", tip)
        Ue = np.zeros((4, np.bitwise_count(local_dofs_per_node))).flatten()
        DOFs = np.concatenate(
            (
                model.list_dof.get_elem_dof_numbers_flat(
                    elem_nodes, BASE_DOFS
                ).flatten(),
                model.list_dof.get_elem_dof_numbers_flat(
                    elem_nodes, HEAVISIDE_DOFS
                ).flatten(),
                model.list_dof.get_elem_dof_numbers_flat(
                    elem_nodes, BRANCH_DOFS
                ).flatten(),
            )
        )
        Ueg = model.Ug[DOFs]
        if len(DOFs) < len(Ue):
            is_present = np.bitwise_count(
                np.bitwise_and(DOF_TYPES[:, None], elem_dofs)
            ).flatten()
            is_present_offsets = np.cumsum(is_present)
            absent_offsets = np.cumsum(
                np.bitwise_count(
                    np.bitwise_and(
                        local_dofs_per_node,
                        np.bitwise_and(DOF_TYPES[:, None], np.bitwise_not(elem_dofs)),
                    )
                )
            )
            ranges = [
                range(
                    is_present_offsets[i] - is_present[i] + absent_offsets[i],
                    is_present_offsets[i] + absent_offsets[i],
                )
                for i in range(3 * n_nodes)
                if is_present[i] != 0
            ]
            Ue[list(itertools.chain.from_iterable(ranges))] = Ueg
            # print("here")
        else:
            Ue[:] = Ueg
        elem_vertices = model.nodes[elem_nodes - 1, 1:3]
        material = model.materials[mat_id - 1][1]
        real = model.reals[real_id - 1][1]
        phi_n, phi_t = model.level_sets[ls].get(elem_nodes, tip)
        in_range = model.in_range[elem_nodes - 1]

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
        Ue = Ue.reshape((-1, 2))

        Nc1, Nc2 = elem._cal_intersections()
        if partial_cut:
            tip1 = np.linalg.solve(
                np.array([phi_t[:-1], phi_n[:-1], [1, 1, 1]]),
                np.array([0, 0, 1]),
            )
            tip2 = np.linalg.solve(
                np.array([phi_t[[0, 2, 3]], phi_n[[0, 2, 3]], [1, 1, 1]]),
                np.array([0, 0, 1]),
            )
            iter1 = elem._partial_cut_embedding_iter(Nc1, tip1, range(4))
            iter2 = elem._partial_cut_embedding_iter(Nc2, tip2, range(2, 6))
        else:
            iter1 = elem._cut_embedding_iter(Nc1)
            iter2 = elem._cut_embedding_iter(Nc2)
        build_triangles(iter1, Ue, np.array([[-1, -1], [1, -1], [1, 1]]), elem_vertices)
        build_triangles(iter2, Ue, np.array([[-1, -1], [1, 1], [-1, 1]]), elem_vertices)

    points_ref = np.array(points_ref).reshape((-1, 2))
    displacements = np.array(displacements).reshape((-1, 2))
    points_ref = np.hstack((points_ref, np.zeros((points_ref.shape[0], 1))))
    displacements = np.hstack((displacements, np.zeros((displacements.shape[0], 1))))
    points = points_ref + displacements
    mesh = pv.PolyData(points, faces)
    mesh.point_data["points_ref"] = points_ref
    mesh.point_data["displacement"] = displacements
    return mesh


def my_build_Quad4n(model, node_stress=None):
    nodes = np.asarray(model.nodes)
    points_ref = nodes[:, 1:4]
    faces = []
    cell_eids = []
    cell_dofs_per_node = []
    is_cut = []
    displacements = np.zeros_like(points_ref)
    displacements[:, :2] = model.Ug[
        model.list_dof.get_elem_dof_numbers_flat(
            1 + np.arange(nodes.shape[0]), BASE_DOFS
        )
    ].reshape((-1, 2))
    for element in model.elements:
        eid, _, mat_id, real_id, elem_nodes = element
        elem_nodes = np.asarray(elem_nodes)
        elem_dofs = model.list_dof.get_elem_dofs(elem_nodes)
        elem_dofs_per_node = np.bitwise_or.reduce(elem_dofs)
        faces.append(4)
        faces += list(elem_nodes - 1)
        cell_eids.append(eid)
        cell_dofs_per_node.append(elem_dofs_per_node)
        is_cut_elem = model.cut_info.get(eid)
        if is_cut_elem is not None:
            _, cut_type, _ = is_cut_elem
            is_cut.append(cut_type == CutType.CUT or cut_type == CutType.PARTIAL)
        else:
            is_cut.append(False)
    points = points_ref + displacements
    faces_flat = np.array(faces)
    points_ref = nodes[:, 1:4]
    mesh = pv.PolyData(points, faces_flat)
    mesh.point_data["points_ref"] = points_ref
    mesh.point_data["displacement"] = displacements
    mesh.cell_data["eid"] = np.asarray(cell_eids, dtype=int)
    mesh.cell_data["dofs_per_node"] = np.asarray(cell_dofs_per_node)
    mesh.cell_data["is_cut"] = np.asarray(is_cut)
    return mesh


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
