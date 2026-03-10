import numpy as np
import scipy as sp
from .dofs import (
    BASE_DOFS,
    HEAVISIDE_DOFS,
    BRANCH_DOFS,
    BRANCH_1_DOFS,
    BRANCH_2_DOFS,
    BRANCH_3_DOFS,
    BRANCH_4_DOFS,
)
from .level_set import CutType
import itertools

DOF_TYPES = np.array(
    [
        BASE_DOFS,
        HEAVISIDE_DOFS,
        BRANCH_DOFS,
    ]
)

ENRICHMENT_TYPES = np.array(
    [
        HEAVISIDE_DOFS,
        BRANCH_1_DOFS,
        BRANCH_2_DOFS,
        BRANCH_3_DOFS,
        BRANCH_4_DOFS,
    ]
)


def cal_KgMg(
    model,
    elem_func,
    eval_mass=False,
    xfem=False,
    tip_enrich=False,
    corrected=False,
    skip_elements={},
):
    print("=> Start evaluating stiffness matrix:")
    Kg = sp.sparse.lil_matrix((len(model.list_dof), len(model.list_dof)))
    cut_info = None
    ci = None
    if xfem:
        cut_info = model.cut_info
        print(cut_info)
    if eval_mass:
        Mg = sp.sparse.lil_matrix((len(model.list_dof), len(model.list_dof)))
    for i_e, ele_info in enumerate(model.elements):
        if (ele_info[0]) in skip_elements:
            continue
        mat_id = ele_info[2]
        real_ie = ele_info[3]
        n_nodes = len(ele_info[4])
        n_dofs = model.dof_per_node.bit_count()
        elem_nodes = np.array(ele_info[4], dtype=np.uint32)
        elem_vertices = model.nodes[elem_nodes - 1, 1 : 1 + n_dofs]
        elem_dofs = model.list_dof.get_elem_dofs(elem_nodes)
        material = model.materials[mat_id - 1][1]
        real = model.reals[real_ie - 1][1]
        local_dofs_per_node = np.bitwise_or.reduce(elem_dofs)
        if xfem:
            h_enrich = local_dofs_per_node & HEAVISIDE_DOFS != 0
            if corrected:
                in_range = model.in_range[elem_nodes - 1]
            else:
                in_range = np.ones(len(ele_info[4]))
            t_enrich = local_dofs_per_node & BRANCH_DOFS != 0
            # h_enrich = np.any(np.bitwise_and(HEAVISIDE_DOFS, elem_dofs))
            # t_enrich = np.any(np.bitwise_and(BRANCH_DOFS, elem_dofs))

            # print(h_enrich, t_enrich)
            if h_enrich or t_enrich:
                # voor elke node van een doorsneden element level set en tip bijhouden
                tip = 1
                ls = model.ls[
                    elem_nodes[
                        np.argmax(
                            np.bitwise_and(elem_dofs, BRANCH_DOFS | HEAVISIDE_DOFS) != 0
                        )
                    ]
                    - 1
                ]
                if t_enrich:
                    tip = model.tip[
                        elem_nodes[
                            np.argmax(np.bitwise_and(elem_dofs, BRANCH_DOFS) != 0)
                        ]
                        - 1
                    ]
                phi_n, phi_t = model.level_sets[ls].get(elem_nodes, tip)
                assert cut_info
                ci = cut_info.get(ele_info[0])
                partial_cut = False
                if ci is not None:
                    _, cut_type, _ = ci
                    partial_cut = cut_type == CutType.PARTIAL
                elem = elem_func(
                    elem_vertices,
                    material,
                    real,
                    phi_n,
                    phi_t,
                    h_enrich,
                    t_enrich,
                    partial_cut,
                    in_range,
                )
                # print("shape_functions", elem.shape_functions(0, 0)[1])
            else:
                elem = elem_func(elem_vertices, material, real)
        else:
            elem = elem_func(elem_vertices, material, real)
        if eval_mass:
            Me, Ke = elem.cal_element_matrices(eval_mass=True)
        else:
            Ke = elem.cal_element_matrices(eval_mass=False)
        DOFs = np.concatenate(
            (
                model.list_dof.get_elem_dof_numbers_flat(elem_nodes, BASE_DOFS).ravel(),
                model.list_dof.get_elem_dof_numbers_flat(
                    elem_nodes, HEAVISIDE_DOFS
                ).ravel(),
                model.list_dof.get_elem_dof_numbers_flat(
                    elem_nodes, BRANCH_DOFS
                ).ravel(),
            )
        )
        if len(DOFs) < Ke.shape[0]:
            is_present = np.bitwise_count(
                np.bitwise_and(DOF_TYPES[:, None], elem_dofs)
            ).ravel()
            is_present_offsets = np.cumsum(is_present)
            absent_offsets = np.cumsum(
                np.bitwise_count(
                    np.bitwise_and(
                        local_dofs_per_node,
                        np.bitwise_and(DOF_TYPES[:, None], np.bitwise_not(elem_dofs)),
                    )
                )
            )
            ranges = [
                range(
                    is_present_offsets[i] - is_present[i] + absent_offsets[i],
                    is_present_offsets[i] + absent_offsets[i],
                )
                for i in range(3 * n_nodes)
                if is_present[i] != 0
            ]
            ranges = list(itertools.chain.from_iterable(ranges))
        else:
            ranges = range(Ke.shape[0])
        for ii, gii in enumerate(ranges):
            for jj, gjj in enumerate(ranges):
                Kg[DOFs[ii], DOFs[jj]] += Ke[gii, gjj]
                if eval_mass:
                    Mg[DOFs[ii], DOFs[jj]] += Me[gii, gjj]
        if (i_e + 1) % 1000 == 0:
            print(
                f"   - e {i_e + 1} ({ele_info[1]}) of {len(model.elements)} evaluated"
            )
    print(".. Stiffness & mass matrix completed!")
    Kg = Kg.tocsr()
    Kg = 0.5 * (Kg + Kg.transpose())
    if eval_mass:
        Mg = Mg.tocsr()
        Mg = 0.5 * (Mg + Mg.transpose())
    model.Kg = Kg
    if eval_mass:
        model.Mg = Mg
    print(".. Starting orthogonalization")
    if xfem:
        T_global = quasi_gram_schmidt(model, Kg)
        model.ortho_T = T_global
    else:
        model.ortho_T = sp.sparse.eye(Kg.shape[0], format="csr")

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


def quasi_gram_schmidt(model, Kg):
    n_dof_per_node = model.dof_per_node.bit_count()
    row_list = []
    col_list = []
    data_list = []
    processed_dofs = np.zeros(Kg.shape[0], dtype=bool)

    def orthogonalize_at_node_batched(node_numbers):
        Kg_local = np.zeros(
            (node_numbers.shape[0], node_numbers.shape[1], node_numbers.shape[1])
        )
        for n, numbers in enumerate(node_numbers):
            Kg_local[n] = Kg[
                numbers[0] : numbers[-1] + 1, numbers[0] : numbers[-1] + 1
            ].toarray()
        T_local = np.tile(np.eye(node_numbers.shape[1]), (Kg_local.shape[0], 1, 1))

        for nj in range(n_dof_per_node, node_numbers.shape[1], n_dof_per_node):
            j_start = nj
            j_end = nj + n_dof_per_node
            for ni in range(0, nj, n_dof_per_node):
                i_start = ni
                i_end = ni + n_dof_per_node
                denom = np.trace(
                    Kg_local[:, i_start:i_end, i_start:i_end], axis1=1, axis2=2
                )
                num = np.trace(
                    Kg_local[:, i_start:i_end, j_start:j_end], axis1=1, axis2=2
                )

                coef = np.divide(
                    num, denom, out=np.zeros_like(num), where=(np.abs(denom) >= 1e-12)
                )

                Kg_local[:, j_start:j_end, :] -= (
                    coef[:, None, None] * Kg_local[:, i_start:i_end, :]
                )
                Kg_local[:, :, j_start:j_end] -= (
                    coef[:, None, None] * Kg_local[:, :, i_start:i_end]
                )
                T_local[:, :, j_start:j_end] -= (
                    coef[:, None, None] * T_local[:, :, i_start:i_end]
                )
        processed_dofs[node_numbers.ravel()] = True
        n_local, r_local, c_local = np.nonzero(T_local)
        vals = T_local[n_local, r_local, c_local]
        row_list.append(node_numbers[n_local, r_local])
        col_list.append(node_numbers[n_local, c_local])
        data_list.append(vals)

    node_numbers = [
        1
        + np.where(
            (model.list_dof.list_dof & HEAVISIDE_DOFS == 0)
            & (model.list_dof.list_dof & BRANCH_DOFS != 0)
        )[0],
        1
        + np.where(
            (model.list_dof.list_dof & HEAVISIDE_DOFS != 0)
            & (model.list_dof.list_dof & BRANCH_DOFS != 0)
        )[0],
    ]
    dof_types = [BRANCH_DOFS, HEAVISIDE_DOFS | BRANCH_DOFS]

    dof_numbers = [
        model.list_dof.get_elem_dof_numbers_flat(i, d).reshape((len(i), -1))
        for i, d in zip(node_numbers, dof_types)
        if len(i) != 0
    ]

    batch_size = 1000
    for dn in dof_numbers:
        n_nodes = len(dn)
        for i0 in range(0, n_nodes, batch_size):
            orthogonalize_at_node_batched(dn[i0 : i0 + batch_size])

    unprocessed_dofs = np.where(~processed_dofs)[0]
    row_list.append(unprocessed_dofs)
    col_list.append(unprocessed_dofs)
    data_list.append(np.ones(len(unprocessed_dofs)))
    row_list_combined = np.concatenate(row_list)
    col_list_combined = np.concatenate(col_list)
    data_list_combined = np.concatenate(data_list)
    T_global = sp.sparse.coo_matrix(
        (data_list_combined, (row_list_combined, col_list_combined)),
        shape=(Kg.shape[0], Kg.shape[0]),
    ).tocsr()

    return T_global
