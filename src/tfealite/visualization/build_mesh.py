import numpy as np
import pyvista as pv

def _id_to_index(nodes):
    nodes = np.asarray(nodes)
    return {int(nid): i for i, nid in enumerate(nodes[:, 0].astype(int))}

def build_Quad4n(nodes, elements, node_stress = None):
    nodes = np.asarray(nodes)
    id2idx = _id_to_index(nodes)
    quad_elems = [e for e in elements if e[1] == 'Quad4n']
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

def build_Tetr4n(nodes, elements, node_stress = None):
    nodes = np.asarray(nodes)
    id2idx = _id_to_index(nodes)
    points = nodes[:, 1:4].astype(float)
    cells_list = []
    cell_eids = []
    for eid, etype, _, _, conn in elements:
        if etype != 'Tetr4n':
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
    dof_per_node,
    load_size = (1.0, 1.0),
    min_mag=1e-3,
):
    arrow_amp, arrow_scale = load_size
    nodes = np.asarray(nodes, dtype=float)
    Fg = np.asarray(Fg, dtype=float)
    has_uz = ('uz' in dof_per_node)
    mesh_load = pv.PolyData()
    for _, nd in enumerate(nodes):
        nid = int(nd[0])
        fx = fy = fz = 0.0
        idx = list_dof.get(f'{nid}ux')
        if idx is not None:
            fx = Fg[idx] * arrow_amp * arrow_scale
        idx = list_dof.get(f'{nid}uy')
        if idx is not None:
            fy = Fg[idx] * arrow_amp * arrow_scale
        if has_uz:
            idx = list_dof.get(f'{nid}uz')
            if idx is not None:
                fz = Fg[idx] * arrow_amp * arrow_scale
        mag_f = np.sqrt(fx*fx + fy*fy + fz*fz)
        if mag_f <= min_mag:
            continue
        x, y, z = nd[1:4]
        start_pt = [x - fx, y - fy, z - fz]
        dirn = [fx / mag_f, fy / mag_f, fz / mag_f]
        arrow = pv.Arrow(
            start=start_pt,
            direction=dirn,
            tip_length=0.25 / mag_f * arrow_scale,
            tip_radius=0.1  / mag_f * arrow_scale,
            tip_resolution=20,
            shaft_radius=0.03 / mag_f * arrow_scale,
            shaft_resolution=20,
            scale=mag_f,
        )
        mesh_load = mesh_load.merge(arrow)
    if mesh_load.n_points == 0:
        return None
    return mesh_load