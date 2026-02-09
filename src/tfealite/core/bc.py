import numpy as np
import scipy as sp


def dirichlet_Lagrange_II(model, fix_dofs):
    model.fix_dofs = fix_dofs
    print(f"=> P matrix started, including {len(fix_dofs)} dofs to eliminate")
    n = model.list_dof.n_dof
    fix_dofs = np.asarray(fix_dofs, dtype=np.int64).ravel()
    if fix_dofs.size:
        fix_dofs = np.unique(fix_dofs[(fix_dofs >= 0) & (fix_dofs < n)])
    keep = np.ones(n, dtype=bool)
    keep[fix_dofs] = False
    colmap = np.full(n, -1, dtype=np.int64)
    colmap[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    rows = np.arange(n, dtype=np.int64)
    cols = colmap.copy()
    mask = cols >= 0
    data = np.ones(int(mask.sum()), dtype=float)
    P = sp.sparse.coo_matrix(
        (data, (rows[mask], cols[mask])), shape=(n, int(keep.sum()))
    ).tocsr()
    model.P = P
    print(
        f"=> P built fast: removed {n - keep.sum()} cols (fixed+slaves); shape = {P.shape}"
    )


def gen_dirichlet_bc(model, sel_condition, tol=1e-8):
    fix_dofs = []
    for node in model.nodes:
        nid = int(node[0])
        x, y, z = map(float, node[1:4])

        if abs(sel_condition(x, y, z)) < tol:
            for offset in range(0, np.bitwise_count(model.dof_per_node)):
                fix_dofs.append(model.list_dof[(nid, 1 << offset)])
            # for d in model.dof_per_node:
            #     key = f"{nid}{d}"
            #     dof_id = model.list_dof[key]
            #     fix_dofs.append(dof_id)
    if fix_dofs:
        fix_dofs = np.array(sorted(set(fix_dofs)), dtype=int)
    else:
        fix_dofs = np.zeros(0, dtype=int)
    model.gen_P(fix_dofs)
