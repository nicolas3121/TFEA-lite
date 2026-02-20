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
        # mask = BASE_DOFS
        if xfem:
            h_enrich = np.any(elem_dofs & HEAVISIDE_DOFS)
            t_enrich = np.any(elem_dofs & BRANCH_DOFS)
            print(h_enrich, t_enrich)
            # if h_enrich:
            #     mask |= HEAVISIDE_DOFS
            # if t_enrich:
            #     mask |= BRANCH_DOFS
            if h_enrich or t_enrich:
                # voor elke node van een doorsneden element level set en tip bijhouden
                phi_n, phi_t = model.level_sets[0].get(elem_nodes, 1)
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
        print(elem_nodes)
        print("base", model.list_dof.get_elem_dof_numbers_flat(elem_nodes, BASE_DOFS))
        print(
            "heaviside",
            model.list_dof.get_elem_dof_numbers_flat(elem_nodes, HEAVISIDE_DOFS),
        )
        print(
            "branch",
            model.list_dof.get_elem_dof_numbers_flat(elem_nodes, BRANCH_DOFS).flatten(),
        )
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
        # print(len(DOFs), Ke.shape[0])
        # print(elem_dofs)
        # print(Ke)
        np.set_printoptions(linewidth=1000, threshold=np.inf, suppress=True)
        print(Ke)
        assert np.all(np.isclose(Ke, Ke.T))

        if len(DOFs) < Ke.shape[0]:
            local_dofs_per_node = np.bitwise_or.reduce(elem_dofs)
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
            print(ranges)
            # range_ii = itertools.chain.from_iterable(ranges)
            # range_jj = itertools.chain.from_iterable(ranges)
        else:
            ranges = [range(Ke.shape[0])]
            # range_ii = range(Ke.shape[0])
            # range_jj = range(Ke.shape[0])
        for ii, gii in enumerate(itertools.chain.from_iterable(ranges)):
            for jj, gjj in enumerate(itertools.chain.from_iterable(ranges)):
                print("ii, gii", ii, gii, "jj, gjj", jj, gjj)
                Kg[DOFs[ii], DOFs[jj]] += Ke[gii, gjj]
                if eval_mass:
                    Mg[DOFs[ii], DOFs[jj]] += Me[gii, gjj]
        # print("nodes", elem_vertices)
        # print("DOFs", DOFs)
        # print(model.list_dof.list_dof_number)
        # print(model.list_dof.list_dof)
        if (i_e + 1) % 1000 == 0:
            print(
                f"   - e {i_e + 1} ({ele_info[1]}) of {len(model.elements)} evaluated"
            )
    print(".. Stiffness & mass matrix completed!")

    Kg__ = Kg.todense()
    assert np.all(np.isclose(Kg__, Kg__.T))

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
