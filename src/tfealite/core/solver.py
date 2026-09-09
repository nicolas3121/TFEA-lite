import numpy as np
import scipy as sp


def static(model, Fg=None):
    if Fg is None:
        Fg = []
    print("=> Static solver started:")
    if hasattr(model, "Fg") and len(Fg) == 0:
        Fg = model.Fg
        print("   - Force vector has already been existing.")
    Kg_bc = model.P.T @ model.ortho_T.T @ model.Kg @ model.ortho_T @ model.P
    print("   - Kg_bc evaluated.")
    Fg_bc = model.P.T @ model.ortho_T.T @ Fg
    print("   - Fg_bc evaluated.")
    D = Kg_bc.diagonal()

    # 2. Create the sparse diagonal scaling matrix (D^-1/2)
    D_inv_sqrt = sp.sparse.diags(1.0 / np.sqrt(D))

    # 3. Scale both the matrix and the load vector
    Kg_scaled = D_inv_sqrt @ Kg_bc @ D_inv_sqrt
    Fg_scaled = D_inv_sqrt @ Fg_bc
    print("   - Diagonal scaling applied.")
    print("   - Start solving for U = inv(K)F ...")
    Ug_scaled = sp.sparse.linalg.spsolve(Kg_scaled, Fg_scaled)
    Ug_bc = D_inv_sqrt @ Ug_scaled
    print("   - Ug_bc evaluated.")
    model.Ug = model.ortho_T @ model.P @ Ug_bc
    print("   - Ug evaluated.")
    print(".. Finished")


def modal(model, tol=1e-3, return_eigs=False, num_eigs=15, sigma=1e-6):
    if hasattr(model, "P"):
        Kg_csr = model.P.transpose() @ model.Mg.tocsr() @ model.P
        Mg_csr = model.P.transpose() @ model.Mg.tocsr() @ model.P
    else:
        Kg_csr = model.Kg.tocsr()
        Mg_csr = model.Mg.tocsr()
    eigenvals, eigenvecs = sp.sparse.linalg.eigsh(
        A=Kg_csr, k=num_eigs, M=Mg_csr, sigma=sigma, which="LM", tol=tol
    )
    model.eigenvals = eigenvals
    model.eigenvecs = eigenvecs

    for ii in range(num_eigs):
        print(
            f"   - f_{(ii + 1):d} = {(np.sign(model.eigenvals[ii]) * np.sqrt(np.abs(model.eigenvals[ii])) / 2 / np.pi):.4f} Hz"
        )
    print(".. Completed")

    if return_eigs:
        return eigenvals, eigenvecs
