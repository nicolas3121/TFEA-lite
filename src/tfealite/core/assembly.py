import numpy as np
import scipy as sp
from ..elements.Quad4n import Quad4n
from ..elements.Tri3n import Tri3n
from ..elements.XTri3n import XTri3n
from ..elements.Tetr4n import Tetr4n


def cal_KgMg(model, eval_mass=False, skip_elements={}):
    print("=> Start evaluating stiffness matrix:")
    Kg = sp.sparse.lil_matrix((len(model.list_dof), len(model.list_dof)))
    if eval_mass:
        Mg = sp.sparse.lil_matrix((len(model.list_dof), len(model.list_dof)))
    for i_e, ele_info in enumerate(model.elements):
        # if (i_e + 1) % 100 == 0:
        #     print(i_e+1)
        if (ele_info[0]) in skip_elements:
            continue
        mat_id = ele_info[2]
        real_ie = ele_info[3]
        if ele_info[1] == "Quad4n" or ele_info[1] == "Tri3n":
            n_nodes = {"Quad4n": 4, "Tri3n": 3}[ele_info[1]]
            elem_func = {"Quad4n": Quad4n, "Tri3n": Tri3n}[ele_info[1]]
            ele_vertices = np.zeros((n_nodes, 2))
            for jj in range(n_nodes):
                i_node = int(ele_info[4][jj])
                ele_vertices[jj, :] = model.nodes[i_node - 1, 1:3]
            material = model.materials[mat_id - 1][1]
            real = model.reals[real_ie - 1][1]
            elem = elem_func(ele_vertices, material, real)
            if eval_mass:
                Me, Ke = elem.cal_element_matrices(eval_mass=True)
            else:
                Ke = elem.cal_element_matrices(eval_mass=False)
            DOFs = np.zeros(n_nodes * 2, dtype=int)
            counter = 0
            for ii in range(n_nodes):
                i_node = ele_info[4][ii]
                DOFs[counter] = model.list_dof[f"{i_node}ux"]
                counter += 1
                DOFs[counter] = model.list_dof[f"{i_node}uy"]
                counter += 1
            for ii in range(2 * n_nodes):
                for jj in range(2 * n_nodes):
                    Kg[DOFs[ii], DOFs[jj]] += Ke[ii, jj]
                    if eval_mass:
                        Mg[DOFs[ii], DOFs[jj]] += Me[ii, jj]
            if (i_e + 1) % 1000 == 0:
                print(f"   - e {i_e + 1} (Quad4n) of {len(model.elements)} evaluated")
        if ele_info[1] == "Tetr4n":
            ele_vertices = np.zeros((4, 3))
            for jj in range(4):
                ele_vertices[jj, :] = model.nodes[int(ele_info[4][jj]) - 1, 1:4]
            material = model.materials[mat_id - 1][1]
            elem = Tetr4n(ele_vertices, material)
            if eval_mass:
                Me, Ke = elem.cal_element_matrices(eval_mass=True)
            else:
                Ke = elem.cal_element_matrices(eval_mass=False)
            DOFs = np.zeros(4 * 3, dtype=int)
            counter = 0
            for ii in range(4):
                i_node = ele_info[4][ii]
                DOFs[counter] = model.list_dof[f"{i_node}ux"]
                counter += 1
                DOFs[counter] = model.list_dof[f"{i_node}uy"]
                counter += 1
                DOFs[counter] = model.list_dof[f"{i_node}uz"]
                counter += 1
            for ii in range(12):
                for jj in range(12):
                    Kg[DOFs[ii], DOFs[jj]] += Ke[ii, jj]
                    if eval_mass:
                        Mg[DOFs[ii], DOFs[jj]] += Me[ii, jj]
            if (i_e + 1) % 1000 == 0:
                print(f"   - e {i_e + 1} (Tetr4n) of {len(model.elements)} evaluated")

    print(".. Stiffness & mass matrix completed!")

    Kg = 0.5 * (Kg + Kg.transpose())
    if eval_mass:
        Mg = 0.5 * (Mg + Mg.transpose())

    model.Kg = Kg
    if eval_mass:
        model.Mg = Mg

    print("=> Check sparsity of Kg: ")
    n_rows, n_cols = Kg.shape
    total_entries = n_rows * n_cols
    nonzero_entries = Kg.nnz
    density = nonzero_entries / total_entries
    print(f"   - Matrix shape: {n_rows} x {n_cols}")
    print(f"   - Non-zero entries: {nonzero_entries}")
    print(f"   - Total entries: {total_entries}")
    print(f"   - Matrix density: {density}")
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
        print(f"   - Matrix density: {density}")
        print(".. Finished")


def cal_KgMg_XFEM(model, eval_mass=False, skip_elements={}):
    print("=> Start evaluating stiffness matrix:")
    Kg = sp.sparse.lil_matrix((len(model.list_dof), len(model.list_dof)))
    print(model.n_dof)
    print(Kg.shape)
    cut_elem = set(model.level_set[2])
    # partial_cut_elem = set(model.level_set[3])
    if eval_mass:
        Mg = sp.sparse.lil_matrix((len(model.list_dof), len(model.list_dof)))
    for i_e, ele_info in enumerate(model.elements):
        if (ele_info[0]) in skip_elements:
            continue
        mat_id = ele_info[2]
        real_ie = ele_info[3]
        if ele_info[1] == "Quad4n" or ele_info[1] == "Tri3n":
            n_nodes = {"Quad4n": 4, "Tri3n": 3}[ele_info[1]]
            elem_func = {"Quad4n": Quad4n, "Tri3n": XTri3n}[ele_info[1]]
            ele_vertices = np.zeros((n_nodes, 2))
            phi_n = np.zeros(n_nodes)
            phi_t = np.zeros(n_nodes)
            for jj in range(n_nodes):
                i_node = int(ele_info[4][jj])
                ele_vertices[jj, :] = model.nodes[i_node - 1, 1:3]
                phi_n[jj] = model.level_set[0][i_node - 1]
                phi_t[jj] = model.level_set[1][i_node - 1]
            material = model.materials[mat_id - 1][1]
            real = model.reals[real_ie - 1][1]
            h_enrich = ele_info[0] in cut_elem
            # t_enrich = ele_info[0] in partial_cut_elem
            t_enrich = False
            elem = elem_func(
                ele_vertices, phi_n, phi_t, h_enrich, t_enrich, t_enrich, material, real
            )
            if eval_mass:
                Me, Ke = elem.cal_element_matrices(eval_mass=True)
            else:
                Ke = elem.cal_element_matrices(eval_mass=False)
            DOFs = np.zeros(Ke.shape[0], dtype=int)
            counter = 0
            for ii in range(n_nodes):
                i_node = ele_info[4][ii]
                DOFs[counter] = model.list_dof[f"{i_node}ux"]
                counter += 1
                DOFs[counter] = model.list_dof[f"{i_node}uy"]
                counter += 1
            if h_enrich:
                for ii in range(n_nodes):
                    i_node = ele_info[4][ii]
                    DOFs[counter] = model.list_dof[f"{i_node}uxH"]
                    counter += 1
                    DOFs[counter] = model.list_dof[f"{i_node}uyH"]
                    counter += 1
            if t_enrich:
                for ii in range(n_nodes):
                    i_node = ele_info[4][ii]
                    for i in range(4):
                        DOFs[counter] = model.list_dof[f"{i_node}uxB{i}"]
                        counter += 1
                        DOFs[counter] = model.list_dof[f"{i_node}uyB{i}"]
                        counter += 1

            for ii in range(Ke.shape[0]):
                for jj in range(Ke.shape[0]):
                    Kg[DOFs[ii], DOFs[jj]] += Ke[ii, jj]
                    if eval_mass:
                        Mg[DOFs[ii], DOFs[jj]] += Me[ii, jj]
            if (i_e + 1) % 1000 == 0:
                print(f"   - e {i_e + 1} (Quad4n) of {len(model.elements)} evaluated")

    print(".. Stiffness & mass matrix completed!")

    Kg = 0.5 * (Kg + Kg.transpose())
    if eval_mass:
        Mg = 0.5 * (Mg + Mg.transpose())

    model.Kg = Kg
    if eval_mass:
        model.Mg = Mg

    print("=> Check sparsity of Kg: ")
    n_rows, n_cols = Kg.shape
    total_entries = n_rows * n_cols
    nonzero_entries = Kg.nnz
    density = nonzero_entries / total_entries
    print(f"   - Matrix shape: {n_rows} x {n_cols}")
    print(f"   - Non-zero entries: {nonzero_entries}")
    print(f"   - Total entries: {total_entries}")
    print(f"   - Matrix density: {density}")
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
        print(f"   - Matrix density: {density}")
        print(".. Finished")
