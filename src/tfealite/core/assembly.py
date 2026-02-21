import numpy as np
import scipy as sp
from .dofs import BASE_DOFS, HEAVISIDE_DOFS, BRANCH_DOFS
from .level_set import CutType
import itertools

DOF_TYPES = np.array(
    [
        BASE_DOFS,
        HEAVISIDE_DOFS,
        BRANCH_DOFS,
    ]
)


def cal_KgMg(
    model, elem_func, eval_mass=False, xfem=False, tip_enrich=False, skip_elements={}
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
                    # print(
                    #     "tip",
                    #     tip,
                    #     elem_dofs,
                    #     np.argmax(np.bitwise_and(elem_dofs, BRANCH_DOFS) != 0),
                    #     elem_nodes[
                    #         np.argmax(np.bitwise_and(elem_dofs, BRANCH_DOFS) != 0)
                    #     ],
                    #     elem_nodes,
                    # )
                    # if tip == 1:
                    #     print("tip is 1", ele_info[0])
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
                )
            else:
                elem = elem_func(elem_vertices, material, real)
        else:
            elem = elem_func(elem_vertices, material, real)
        if eval_mass:
            Me, Ke = elem.cal_element_matrices(eval_mass=True)
        else:
            Ke = elem.cal_element_matrices(eval_mass=False)
        # print("base", model.list_dof.get_elem_dof_numbers_flat(elem_nodes, BASE_DOFS))
        # print(
        #     "heaviside",
        #     model.list_dof.get_elem_dof_numbers_flat(elem_nodes, HEAVISIDE_DOFS),
        # )
        # print(
        #     "branch",
        #     model.list_dof.get_elem_dof_numbers_flat(elem_nodes, BRANCH_DOFS).flatten(),
        # )
        DOFs = np.concatenate(
            (
                model.list_dof.get_elem_dof_numbers_flat(
                    elem_nodes, BASE_DOFS
                ).flatten(),
                model.list_dof.get_elem_dof_numbers_flat(
                    elem_nodes, HEAVISIDE_DOFS
                ).flatten(),
                model.list_dof.get_elem_dof_numbers_flat(
                    elem_nodes, BRANCH_DOFS
                ).flatten(),
            )
        )
        if len(DOFs) < Ke.shape[0]:
            is_present = np.bitwise_count(
                np.bitwise_and(DOF_TYPES[:, None], elem_dofs)
            ).flatten()
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
