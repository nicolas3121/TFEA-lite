import numpy as np
import scipy as sp

def static(model, Fg = []):
    print('=> Static solver started:')
    if hasattr(model, "Fg") and len(Fg) == 0:
        Fg = model.Fg
        print('   - Force vector has already been existing.')
    Kg_bc = model.P.T @ model.Kg @ model.P
    print('   - Kg_bc evaluated.')
    Fg_bc = model.P.T @ Fg
    print('   - Fg_bc evaluated.')
    print('   - Start solving for U = inv(K)F ...')
    Ug_bc = sp.sparse.linalg.spsolve(Kg_bc, Fg_bc)
    print('   - Ug_bc evaluated.')
    model.Ug = model.P @ Ug_bc
    print('   - Ug evaluated.')
    print('.. Finished')

def modal(
        model,
        tol = 1e-3,
        return_eigs = False,
        num_eigs = 15,
        sigma = 1e-6
):
    if hasattr(model, 'P'):        
        Kg_csr = model.P.transpose() @ model.Mg.tocsr() @ model.P
        Mg_csr = model.P.transpose() @ model.Mg.tocsr() @ model.P
    else:
        Kg_csr = model.Kg.tocsr()
        Mg_csr = model.Mg.tocsr()
    eigenvals, eigenvecs = sp.sparse.linalg.eigsh(
        A = Kg_csr,
        k = num_eigs,
        M = Mg_csr,
        sigma = sigma,
        which = 'LM',
        tol = tol
    )
    model.eigenvals = eigenvals
    model.eigenvecs = eigenvecs

    for ii in range(num_eigs):
        print(f"   - f_{(ii+1):d} = {(np.sign(model.eigenvals[ii])*np.sqrt(np.abs(model.eigenvals[ii]))/2/np.pi):.4f} Hz")
    print(".. Completed")

    if return_eigs:
        return eigenvals, eigenvecs