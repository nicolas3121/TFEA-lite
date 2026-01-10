import numpy as np


def gen_nodal_forces(model, sel_condition, force_expression, tol=1e-8, reset=True):
    if reset or getattr(model, "Fg", None) is None:
        Fg = np.zeros(model.n_dof, dtype=float)
    else:
        Fg = np.array(model.Fg, copy=True, dtype=float)

    for node in model.nodes:
        nid = int(node[0])
        x, y, z = map(float, node[1:4])

        if abs(sel_condition(x, y, z)) < tol:
            fx, fy, fz = force_expression(x, y, z)

            if fx != 0.0:
                dof_id = model.list_dof[f"{nid}ux"]
                Fg[dof_id] += fx
            if fy != 0.0:
                dof_id = model.list_dof[f"{nid}uy"]
                Fg[dof_id] += fy
            if fz != 0.0:
                dof_id = model.list_dof[f"{nid}uz"]
                Fg[dof_id] += fz

    model.Fg = Fg
