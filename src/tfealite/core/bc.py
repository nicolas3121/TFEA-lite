import numpy as np
import scipy as sp

def dirichlet_Lagrange_II(model, fix_dofs):
    model.fix_dofs = fix_dofs
    print(f"=> P matrix started, including {len(fix_dofs)} dofs to eliminate")
    n = len(model.list_dof)
    fix_dofs = np.asarray(fix_dofs, dtype=np.int64).ravel()
    if fix_dofs.size:
        fix_dofs = np.unique(fix_dofs[(fix_dofs >= 0) & (fix_dofs < n)])
    keep = np.ones(n, dtype=bool)
    keep[fix_dofs] = False
    colmap = np.full(n, -1, dtype=np.int64)
    colmap[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    rows = np.arange(n, dtype=np.int64)
    cols = colmap.copy()
    mask = (cols >= 0)
    data = np.ones(int(mask.sum()), dtype=float)
    P = sp.sparse.coo_matrix((data, (rows[mask], cols[mask])),
                    shape=(n, int(keep.sum()))).tocsr()
    model.P = P
    print(f"=> P built fast: removed {n - keep.sum()} cols (fixed+slaves); shape = {P.shape}")