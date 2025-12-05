import numpy as np
from collections import Counter

def model_print(model):
    print(f'=> Model created successfuly, including:')
    print(f'   - {model.n_nodes} nodes')
    n_elements = len(model.elements)
    counts = Counter([etype for _, etype, *_ in model.elements])
    details = ", ".join(f"{counts[t]} {t}" for t in sorted(counts))
    print(f"   - {n_elements} elements, including: {details}")

def gen_list_dof(
        model,
        dof_per_node = ['ux', 'uy', 'uz']
):
    model.dof_per_node = dof_per_node
    model.list_dof = {}
    counter = 0
    for node in model.nodes:
        for dof in dof_per_node:
            model.list_dof |= {str(int(node[0])) + dof: counter}
            counter += 1
    model.n_dof = len(model.list_dof)

def gen_rect_Quad4n(
        L, H,
        nx = 20,
        ny = 20
):
    dx = L / nx
    dy = H / ny
    nodes = []
    nid = 1
    for j in range(ny + 1):
        y = j * dy
        for i in range(nx + 1):
            x = i * dx
            nodes.append([nid, x, y, 0.0])
            nid += 1
    nodes = np.array(nodes, dtype = float)
    elements = []
    eid = 1
    for j in range(ny):
        for i in range(nx):
            if i < 3 and j == 8:
                continue
            n1 = j * (nx + 1) + i + 1
            n2 = n1 + 1
            n3 = n2 + (nx + 1)
            n4 = n1 + (nx + 1)
            elements.append([eid, 'Quad4n', 1, 1, (n1, n2, n3, n4)])
            eid += 1
    return nodes, elements

def gen_ibeam_Tetr4n(
    L, h, bf, tw, tf,
    nx = 50,
    ny_f = 4,
    ny_w = 1,
    nz_f = 1,
    nz_w = 10
):
    x = np.linspace(0.0, L, nx + 1)
    y1 = np.linspace(-bf / 2, -tw / 2, ny_f + 1)
    y2 = np.linspace(-tw / 2,  tw / 2, ny_w + 1)
    y3 = np.linspace( tw / 2,  bf / 2, ny_f + 1)
    y = np.concatenate((y1, y2[1:], y3[1:]))
    z1 = np.linspace(0.0, tf, nz_f + 1)
    z2 = np.linspace(tf, h - tf, nz_w + 1)
    z3 = np.linspace(h - tf, h, nz_f + 1)
    z = np.concatenate((z1, z2[1:], z3[1:]))
    nxn, nyn, nzn = len(x), len(y), len(z)

    nodes = []
    nid_map = {}
    nid = 1
    for iz, zz in enumerate(z):
        for iy, yy in enumerate(y):
            for ix, xx in enumerate(x):
                nid_map[(ix, iy, iz)] = nid
                nodes.append((nid, float(xx), float(yy), float(zz)))
                nid += 1

    def nid_at(ix, iy, iz):
        return nid_map[(ix, iy, iz)]

    elems = []
    eid = 1
    eps = 1e-9
    for kz in range(nzn - 1):
        z0, z1_ = z[kz], z[kz + 1]
        for ky in range(nyn - 1):
            y0, y1_ = y[ky], y[ky + 1]
            in_bottom_flange = (z1_ <= tf + eps)
            in_top_flange = (z0 >= h - tf - eps)
            in_web_z = (z0 >= tf - eps) and (z1_ <= h - tf + eps)
            in_web_y = (y0 >= -tw / 2 - eps) and (y1_ <= tw / 2 + eps)
            inside = False
            if in_bottom_flange or in_top_flange:
                inside = True
            elif in_web_z and in_web_y:
                inside = True
            if not inside:
                continue
            for kx in range(nxn - 1):
                n000 = nid_at(kx,     ky,     kz)
                n100 = nid_at(kx + 1, ky,     kz)
                n110 = nid_at(kx + 1, ky + 1, kz)
                n010 = nid_at(kx,     ky + 1, kz)
                n001 = nid_at(kx,     ky,     kz + 1)
                n101 = nid_at(kx + 1, ky,     kz + 1)
                n111 = nid_at(kx + 1, ky + 1, kz + 1)
                n011 = nid_at(kx,     ky + 1, kz + 1)
                tets = [
                    (n000, n100, n110, n111),
                    (n000, n110, n010, n111),
                    (n000, n010, n011, n111),
                    (n000, n011, n001, n111),
                    (n000, n001, n101, n111),
                    (n000, n101, n100, n111),
                ]
                for (n1, n2, n3, n4) in tets:
                    elems.append([eid, 'Tetr4n', 1, 1, (n1, n2, n3, n4)])
                    eid += 1    

    nodes = np.array(nodes)
    used_nids = set()
    for _, _, _, _, conn in elems:
        used_nids.update(conn)
    used_nids = sorted(used_nids)
    old2new = {old: i + 1 for i, old in enumerate(used_nids)}
    new_nodes = []
    for row in nodes:
        nid_old = int(row[0])
        if nid_old not in old2new:
            continue
        nid_new = old2new[nid_old]
        xcoord = float(row[1])
        ycoord = float(row[2])
        zcoord = float(row[3])
        new_nodes.append((nid_new, xcoord, ycoord, zcoord))
    new_elems = []
    eid_new = 1
    for _, etype, mat_id, sec_id, conn in elems:
        n1, n2, n3, n4 = conn
        new_conn = (
            old2new[int(n1)],
            old2new[int(n2)],
            old2new[int(n3)],
            old2new[int(n4)],
        )
        new_elems.append([eid_new, etype, mat_id, sec_id, new_conn])
        eid_new += 1
    nodes = np.array(new_nodes)
    elems = new_elems

    return nodes, elems