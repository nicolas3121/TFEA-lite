import numpy as np
from ..elements.Quad4n import Quad4n
from ..elements.Tri3n import Tri3n
from ..elements.Tetr4n import Tetr4n
from .dofs import DofType


def cal_Tetr4n_stresses(model):
    Ug = np.asarray(model.Ug, dtype=float)
    stresses = np.zeros((model.n_elements, 6), dtype=float)
    for element in model.elements:
        eid, _, mat_id, real_id, conn = element
        Ue = np.zeros((4, 3), dtype=float)
        Ue[0, 0] = Ug[model.list_dof[(conn[0], DofType.UX)]]
        Ue[0, 1] = Ug[model.list_dof[(conn[0], DofType.UY)]]
        Ue[0, 2] = Ug[model.list_dof[(conn[0], DofType.UZ)]]
        Ue[1, 0] = Ug[model.list_dof[(conn[1], DofType.UX)]]
        Ue[1, 1] = Ug[model.list_dof[(conn[1], DofType.UY)]]
        Ue[1, 2] = Ug[model.list_dof[(conn[1], DofType.UZ)]]
        Ue[2, 0] = Ug[model.list_dof[(conn[2], DofType.UX)]]
        Ue[2, 1] = Ug[model.list_dof[(conn[2], DofType.UY)]]
        Ue[2, 2] = Ug[model.list_dof[(conn[2], DofType.UZ)]]
        Ue[3, 0] = Ug[model.list_dof[(conn[3], DofType.UX)]]
        Ue[3, 1] = Ug[model.list_dof[(conn[3], DofType.UY)]]
        Ue[3, 2] = Ug[model.list_dof[(conn[3], DofType.UZ)]]
        ele_vertices = np.zeros((4, 3))
        for jj in range(4):
            ele_vertices[jj, :] = model.nodes[int(conn[jj]) - 1, 1:4]
        tetr = Tetr4n(
            ele_vertices, model.materials[mat_id - 1][1], model.reals[real_id - 1][1]
        )
        stresses[eid - 1, :] = tetr.cal_element_stress(Ue)
    return stresses


def compute_quad4n_nodal_stresses(model, Ug=None):
    if Ug is None:
        Ug = model.Ug
    else:
        Ug = np.asarray(Ug, dtype=float).ravel()
    n = model.n_nodes
    id_to_idx = {int(nid): i for i, nid in enumerate(model.nodes[:, 0].astype(int))}
    sums = np.zeros((n, 3), dtype=float)
    cnts = np.zeros(n, dtype=int)
    for eid, etype, mat_id, real_id, conn in model.elements:
        if etype != "Quad4n":
            continue
        conn = [int(nid) for nid in conn]
        idxs = [id_to_idx[nid] for nid in conn]
        coords2d = model.nodes[idxs, 1:3].astype(float)
        material = model.materials[mat_id - 1][1]
        real = model.reals[real_id - 1][1]
        elem = Quad4n(coords2d, material, real)
        dofs = []
        for nid in conn:
            dofs.append(model.list_dof[(nid, DofType.UX)])
            dofs.append(model.list_dof[(nid, DofType.UY)])
        dofs = np.asarray(dofs, dtype=int)
        Ue = Ug[dofs]
        sig_nodes = elem.stresses_at_nodes(Ue)
        for a, nid in enumerate(conn):
            i = id_to_idx[nid]
            sums[i, :] += sig_nodes[a, :]
            cnts[i] += 1
    nodal_sigma = np.full((n, 3), np.nan, dtype=float)
    mask = cnts > 0
    nodal_sigma[mask, :] = sums[mask, :] / cnts[mask][:, None]
    return nodal_sigma


def compute_tri3n_nodal_stresses(model, Ug=None):
    if Ug is None:
        Ug = model.Ug
    else:
        Ug = np.asarray(Ug, dtype=float).ravel()
    n = model.n_nodes
    id_to_idx = {int(nid): i for i, nid in enumerate(model.nodes[:, 0].astype(int))}
    sums = np.zeros((n, 3), dtype=float)
    cnts = np.zeros(n, dtype=int)
    for eid, etype, mat_id, real_id, conn in model.elements:
        if etype != "Tri3n":
            continue
        conn = [int(nid) for nid in conn]
        idxs = [id_to_idx[nid] for nid in conn]
        coords2d = model.nodes[idxs, 1:3].astype(float)
        material = model.materials[mat_id - 1][1]
        real = model.reals[real_id - 1][1]
        elem = Tri3n(coords2d, material, real)
        dofs = []
        for nid in conn:
            dofs.append(model.list_dof[(nid, DofType.UX)])
            dofs.append(model.list_dof[(nid, DofType.UY)])
        dofs = np.asarray(dofs, dtype=int)
        Ue = Ug[dofs]
        sig_nodes = elem.stresses_at_nodes(Ue)
        for a, nid in enumerate(conn):
            i = id_to_idx[nid]
            sums[i, :] += sig_nodes[a, :]
            cnts[i] += 1
    nodal_sigma = np.full((n, 3), np.nan, dtype=float)
    mask = cnts > 0
    nodal_sigma[mask, :] = sums[mask, :] / cnts[mask][:, None]
    return nodal_sigma


def eval_node_average(model, sxx):
    nodes = model.nodes
    elements = model.elements
    n_nodes = len(nodes)
    node_ids = nodes[:, 0].astype(int)
    nid2idx = {nid: i for i, nid in enumerate(node_ids)}
    stress_sum = np.zeros(n_nodes, dtype=float)
    vol_sum = np.zeros(n_nodes, dtype=float)
    for e, elem in enumerate(elements):
        eid, etype, mat_id, real_id, conn = elem
        n1, n2, n3, n4 = conn
        i1, i2, i3, i4 = nid2idx[n1], nid2idx[n2], nid2idx[n3], nid2idx[n4]
        X = nodes[[i1, i2, i3, i4], 1:4].astype(float)
        v = abs(np.linalg.det(np.array([X[1] - X[0], X[2] - X[0], X[3] - X[0]]))) / 6.0
        for i in (i1, i2, i3, i4):
            stress_sum[i] += sxx[e] * v
            vol_sum[i] += v
    sxx_nodes = stress_sum / np.maximum(vol_sum, 1e-16)
    return sxx_nodes
