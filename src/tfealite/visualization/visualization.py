import numpy as np
import pyvista as pv

from . import build_mesh as bm


def build_gcs(model, length=1.0):
    mesh_gcs = pv.MultiBlock()
    L = float(length)
    arr_x = pv.Line((-L, 0, 0), (L, 0, 0))
    mesh_gcs["X"] = arr_x
    arr_y = pv.Line((0, -L, 0), (0, L, 0))
    mesh_gcs["Y"] = arr_y
    arr_z = pv.Line((0, 0, -L), (0, 0, L))
    mesh_gcs["Z"] = arr_z
    return mesh_gcs


def show(
    model,
    gcs_length=0.0,
    show_edges=True,
    node_size=0.0,
    nid_size=0.0,
    eid_size=0.0,
    Ug=None,
    nbc_size=0.0,
    node_stress=None,
    clim=None,
    load_size=None,
    window_size=None,
    show_elements=True,
    file_name=None,
    show_axes=False,
    colorbar_title="Stress",
    show_undef=False,
):
    do_plot = file_name is None
    nodes_def = model.nodes.copy().astype(float)
    if Ug is not None:
        Ug = np.asarray(Ug, dtype=float).ravel()
        if not hasattr(model, "dof_per_node"):
            raise RuntimeError("gen_list_dof() not found.")
        ndpn = len(model.dof_per_node)
        disp = Ug.reshape(model.n_nodes, ndpn)
        nodes_def[:, 1 : ndpn + 1] += disp[:, 0:ndpn]
    if (node_stress is not None) and (clim is None):
        c_max = np.max(np.abs(node_stress))
        clim = (-c_max, c_max)

    pl = None
    if do_plot:
        pl = pv.Plotter()
        if window_size is not None:
            pl.window_size = window_size
        if file_name is not None:
            pl.off_screen = True

    def _add_element_mesh(mesh, has_stress=False):
        if mesh is None or not do_plot:
            return
        base_kwargs = dict(show_edges=show_edges, edge_color=(0.2, 0.2, 0.2))
        if not has_stress or node_stress is None:
            pl.add_mesh(
                mesh,
                color="LightGray",
                **base_kwargs,
            )
        else:
            pl.add_mesh(
                mesh,
                scalars="node_stress",
                cmap="coolwarm",
                clim=clim,
                scalar_bar_args={"title": colorbar_title},
                **base_kwargs,
            )

    mesh_elements = {}
    if show_elements and model.elements:
        types_present = {e[1] for e in model.elements}
        if "Quad4n" in types_present:
            if show_undef:
                mesh_q0 = bm.build_Quad4n(model.nodes, model.elements)
            else:
                mesh_q0 = None
            mesh_q = bm.build_Quad4n(nodes_def, model.elements, node_stress=node_stress)
            mesh_elements["Quad4n"] = {"undeformed": mesh_q0, "deformed": mesh_q}
            _add_element_mesh(mesh_q, has_stress=(node_stress is not None))
            if show_undef and Ug is not None and do_plot and mesh_q0 is not None:
                pl.add_mesh(mesh_q0, style="wireframe", color=(0.2, 0.2, 0.2))
        if "Tri3n" in types_present:
            if show_undef:
                mesh_q0 = bm.build_Tri3n(model.nodes, model.elements)
            else:
                mesh_q0 = None
            mesh_q = bm.build_Tri3n(nodes_def, model.elements, node_stress=node_stress)
            mesh_elements["Tri3n"] = {"undeformed": mesh_q0, "deformed": mesh_q}
            _add_element_mesh(mesh_q, has_stress=(node_stress is not None))
            if show_undef and Ug is not None and do_plot and mesh_q0 is not None:
                pl.add_mesh(mesh_q0, style="wireframe", color=(0.2, 0.2, 0.2))
        if "Tetr4n" in types_present:
            if show_undef:
                mesh_t0 = bm.build_Tetr4n(model.nodes, model.elements)
            else:
                mesh_t0 = None
            mesh_t = bm.build_Tetr4n(nodes_def, model.elements, node_stress=node_stress)
            mesh_elements["Tetr4n"] = {"undeformed": mesh_t0, "deformed": mesh_t}
            _add_element_mesh(mesh_t, has_stress=(node_stress is not None))
            if show_undef and Ug is not None and do_plot and mesh_t0 is not None:
                pl.add_mesh(mesh_t0, style="wireframe", color=(0.2, 0.2, 0.2))

    mesh_gcs = None
    if gcs_length > 0.0:
        mesh_gcs = model.build_gcs(length=gcs_length)
        if do_plot:
            for name, color in zip(["X", "Y", "Z"], ["red", "green", "blue"]):
                pl.add_mesh(
                    mesh_gcs[name],
                    color=color,
                    label=name,
                    line_width=2,
                )

    mesh_nodes = None
    if node_size > 0.0 and do_plot:
        mesh_nodes = pv.PolyData(nodes_def[:, 1:4])
        pl.add_mesh(
            mesh_nodes,
            color="steelblue",
            render_points_as_spheres=True,
            point_size=node_size,
        )

    cents_pd = None
    if model.elements and eid_size != 0.0:
        cents = np.vstack(
            [
                nodes_def[:, 1:4][np.asarray(elem[4], dtype=int) - 1].mean(axis=0)
                for elem in model.elements
            ]
        )
        cents_pd = pv.PolyData(cents)
        elem_ids = np.arange(1, len(model.elements) + 1)

    mesh_node_bc = None
    if nbc_size > 0.0 and hasattr(model, "fix_dofs"):
        node_coord_fixed = []
        for node in nodes_def:
            nid = int(node[0])
            if (
                model.list_dof.get(f"{nid}ux") in model.fix_dofs
                or model.list_dof.get(f"{nid}uy") in model.fix_dofs
                or model.list_dof.get(f"{nid}uz") in model.fix_dofs
                or model.list_dof.get(f"{nid}rx") in model.fix_dofs
                or model.list_dof.get(f"{nid}ry") in model.fix_dofs
                or model.list_dof.get(f"{nid}rz") in model.fix_dofs
            ):
                node_coord_fixed.append(node[1:4])
        mesh_node_bc = (
            pv.PolyData(node_coord_fixed) if node_coord_fixed else pv.PolyData()
        )
        if do_plot:
            pl.add_mesh(
                mesh_node_bc,
                color="red",
                render_points_as_spheres=True,
                point_size=nbc_size,
            )

    if nid_size > 0.0 and do_plot:
        mesh_nlb = pv.PolyData(nodes_def[:, 1:4])
        pl.add_point_labels(
            mesh_nlb,
            [str(ii + 1) for ii in range(len(model.nodes))],
            font_size=nid_size,
            shape=None,
            point_size=0.0,
            always_visible=True,
        )

    if eid_size > 0.0 and do_plot and len(elem_ids):
        pl.add_point_labels(
            cents_pd,
            [f"[{ii}]" for ii in elem_ids],
            font_size=eid_size,
            shape=None,
            always_visible=True,
            point_size=0.0,
        )

    mesh_load = None
    if load_size is not None and hasattr(model, "Fg"):
        mesh_load = bm.build_load_arrows(
            nodes_def, model.Fg, model.list_dof, model.dof_per_node, load_size=load_size
        )
        if do_plot and mesh_load is not None:
            pl.add_mesh(mesh_load, color="red")

    if do_plot:
        if show_axes:
            pl.show_axes()
        if model.nodes.shape[1] >= 4 and np.allclose(model.nodes[:, -1], 0.0):
            pl.view_xy()
            pl.enable_parallel_projection()
        if file_name is not None:
            pl.screenshot(f"{file_name}.png", scale=2.0)
        else:
            pl.show()

