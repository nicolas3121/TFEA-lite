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