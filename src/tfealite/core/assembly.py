import numpy as np
import scipy as sp
from ..elements.Quad4n import Quad4n
from ..elements.Tetr4n import Tetr4n

def cal_KgMg(model, eval_mass = False, skip_elements = {}):
    print("=> Start evaluating stiffness matrix:")
    Kg = sp.sparse.lil_matrix((len(model.list_dof),len(model.list_dof)))
    if eval_mass == True:
        Mg = sp.sparse.lil_matrix((len(model.list_dof),len(model.list_dof)))
    for i_e, ele_info in enumerate(model.elements):
        # if (i_e + 1) % 100 == 0:
        #     print(i_e+1)
        if (ele_info[0]) in skip_elements:
            continue
        mat_id = ele_info[2]
        real_ie = ele_info[3]
        if ele_info[1] == 'Quad4n':
            ele_vertices = np.zeros((4, 2))
            for jj in range(4):
                i_node = int(ele_info[4][jj])
                ele_vertices[jj, :] = model.nodes[i_node - 1, 1:3]
            material = model.materials[mat_id - 1][1]
            real     = model.reals[real_ie - 1][1]
            elem = Quad4n(ele_vertices, material, real)
            if eval_mass:
                Me, Ke = elem.cal_element_matrices(eval_mass=True)
            else:
                Ke = elem.cal_element_matrices(eval_mass=False)
            DOFs = np.zeros(4 * 2, dtype=int)
            counter = 0
            for ii in range(4):
                i_node = ele_info[4][ii]
                DOFs[counter] = model.list_dof[f'{i_node}ux']; counter += 1
                DOFs[counter] = model.list_dof[f'{i_node}uy']; counter += 1
            for ii in range(8):
                for jj in range(8):
                    Kg[DOFs[ii], DOFs[jj]] += Ke[ii, jj]
                    if eval_mass:
                        Mg[DOFs[ii], DOFs[jj]] += Me[ii, jj]
            if (i_e + 1) % 1000 == 0:
                print(f'   - e {i_e+1} (Quad4n) of {len(model.elements)} evaluated')
        if ele_info[1] == 'Tetr4n':
            ele_vertices = np.zeros((4,3))
            for jj in range(4):
                ele_vertices[jj,:] = model.nodes[int(ele_info[4][jj])-1,1:4]
            material = model.materials[mat_id-1][1]
            elem = Tetr4n(ele_vertices, material)
            if eval_mass:
                Me, Ke = elem.cal_element_matrices(eval_mass = True)
            else:
                Ke = elem.cal_element_matrices(eval_mass = False)
            DOFs = np.zeros(4*3, dtype=int)
            counter = 0
            for ii in range(4):
                i_node = ele_info[4][ii]
                DOFs[counter] = model.list_dof[f'{i_node}ux']; counter += 1
                DOFs[counter] = model.list_dof[f'{i_node}uy']; counter += 1
                DOFs[counter] = model.list_dof[f'{i_node}uz']; counter += 1
            for ii in range(12):
                for jj in range(12):
                    Kg[DOFs[ii], DOFs[jj]] += Ke[ii, jj]
                    if eval_mass:
                        Mg[DOFs[ii], DOFs[jj]] += Me[ii, jj]
            if (i_e+1) % 1000 == 0:
                print(f'   - e {i_e+1} (Tetr4n) of {len(model.elements)} evaluated')
                
    print(".. Stiffness & mass matrix completed!")
    
    Kg = 0.5*(Kg + Kg.transpose())
    if eval_mass == True:
        Mg = 0.5*(Mg + Mg.transpose())

    model.Kg = Kg
    if eval_mass == True:
        model.Mg = Mg
    
    print("=> Check sparsity of Kg: ")
    n_rows, n_cols = Kg.shape
    total_entries = n_rows * n_cols
    nonzero_entries = Kg.nnz
    density = nonzero_entries / total_entries        
    print(f"   - Matrix shape: {n_rows} x {n_cols}")
    print(f"   - Non-zero entries: {nonzero_entries}")
    print(f"   - Total entries: {total_entries}")
    print(".. Finished")

    if eval_mass:
        print("=> Check sparsity of Mg: ")
        n_rows, n_cols = Mg.shape
        total_entries = n_rows * n_cols
        nonzero_entries = Mg.nnz
        density = nonzero_entries / total_entries        
        print(f"   - Matrix shape: {n_rows} x {n_cols}")
        print(f"   - Non-zero entries: {nonzero_entries}")
        print(f"   - Total entries: {total_entries}")
        print(".. Finished")