import numpy as np
from .dofs import DofType
from .dofs import (
    BASE_DOFS,
)


def gen_nodal_forces(model, sel_condition, force_expression, tol=1e-8, reset=True):
    if reset or getattr(model, "Fg", None) is None:
        Fg = np.zeros(model.list_dof.n_dof, dtype=float)
    else:
        Fg = np.array(model.Fg, copy=True, dtype=float)

    for node in model.nodes:
        nid = int(node[0])
        x, y, z = map(float, node[1:4])

        if abs(sel_condition(x, y, z)) < tol:
            fx, fy, fz = force_expression(x, y, z)
            if fx != 0.0:
                dof_id = model.list_dof[(nid, DofType.UX)]
                Fg[dof_id] += fx
            if fy != 0.0:
                dof_id = model.list_dof[(nid, DofType.UY)]
                # print(dof_id)
                Fg[dof_id] += fy
            if fz != 0.0:
                dof_id = model.list_dof[(nid, DofType.UZ)]
                Fg[dof_id] += fz

    model.Fg = Fg


def gen_surface_tractions(
    model, sel_condition, traction_expression, elem_func, deg, tol=1e-8, reset=True
):
    if reset or getattr(model, "Fg", None) is None:
        Fg = np.zeros(model.list_dof.n_dof, dtype=float)
    else:
        Fg = np.array(model.Fg, copy=True, dtype=float)
    node_on_boundary = (
        np.abs(sel_condition(model.nodes[:, 1], model.nodes[:, 2], model.nodes[:, 3]))
        < tol
    )

    for _, ele_info in enumerate(model.elements):
        elem_nodes = np.array(ele_info[4], dtype=np.uint32)
        elem_bools = node_on_boundary[elem_nodes - 1]

        if not np.any(elem_bools):
            continue
        n_dofs = model.dof_per_node.bit_count()
        elem_vertices = model.nodes[elem_nodes - 1, 1 : 1 + n_dofs]
        real_ie = ele_info[3]
        real = model.reals[real_ie - 1][1]

        Fe_local = elem_func.cal_traction_loads(
            elem_vertices, elem_bools, traction_expression, real, deg
        )

        g_dofs = model.list_dof.get_elem_dof_numbers_flat(elem_nodes, BASE_DOFS).ravel()

        for i_dof, global_dof_index in enumerate(g_dofs):
            Fg[global_dof_index] += Fe_local[i_dof]
    model.Fg = Fg
