import numpy as np
import scipy as sp
from .dofs import BASE_DOFS, HEAVISIDE_DOFS, BRANCH_DOFS


def cal_KgMg(model, elem_func, eval_mass=False, xfem=False, skip_elements={}):
    print("=> Start evaluating stiffness matrix:")
    Kg = sp.sparse.lil_matrix((len(model.list_dof), len(model.list_dof)))
    if xfem:
        cut_elem = set(model.level_set[2])
    if eval_mass:
        Mg = sp.sparse.lil_matrix((len(model.list_dof), len(model.list_dof)))
    for i_e, ele_info in enumerate(model.elements):
        if (ele_info[0]) in skip_elements:
            continue
        mat_id = ele_info[2]
        real_ie = ele_info[3]
        # n_nodes = len(ele_info[4])
        n_dofs = model.dof_per_node.bit_count()
        elem_nodes = np.array(ele_info[4], dtype=np.uint32)
        elem_vertices = model.nodes[elem_nodes - 1, 1 : 1 + n_dofs]
        material = model.materials[mat_id - 1][1]
        real = model.reals[real_ie - 1][1]
        mask = BASE_DOFS
        if xfem:
            phi_n = model.level_set[0][elem_nodes - 1]
            phi_t = model.level_set[1][elem_nodes - 1]
            h_enrich = ele_info[0] in cut_elem
            t_enrich = False
            if h_enrich:
                mask |= HEAVISIDE_DOFS
            elem = elem_func(
                elem_vertices,
                phi_n,
                phi_t,
                h_enrich,
                t_enrich,
                t_enrich,
                material,
                real,
            )
        else:
            elem = elem_func(elem_vertices, material, real)
        if eval_mass:
            Me, Ke = elem.cal_element_matrices(eval_mass=True)
        else:
            Ke = elem.cal_element_matrices(eval_mass=False)
        DOFs = np.concatenate(
            (
                model.list_dof.get_elem_dof_numbers(
                    elem_nodes, mask & BASE_DOFS
                ).flatten(),
                model.list_dof.get_elem_dof_numbers(
                    elem_nodes, mask & HEAVISIDE_DOFS
                ).flatten(),
                model.list_dof.get_elem_dof_numbers(
                    elem_nodes, mask & BRANCH_DOFS
                ).flatten(),
            )
        )
        for ii in range(Ke.shape[0]):
            for jj in range(Ke.shape[0]):
                Kg[DOFs[ii], DOFs[jj]] += Ke[ii, jj]
                if eval_mass:
                    Mg[DOFs[ii], DOFs[jj]] += Me[ii, jj]
        if (i_e + 1) % 1000 == 0:
            print(
                f"   - e {i_e + 1} ({ele_info[1]}) of {len(model.elements)} evaluated"
            )
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
