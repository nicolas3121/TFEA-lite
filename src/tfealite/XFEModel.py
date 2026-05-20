from numpy.typing import NDArray
import numpy as np
from .FEModel import FEModel
from .core.level_set import LevelSet, CutType
from .core import model
from .core import assembly as asm
from .core import dofs
from .visualization.build_mesh import build_mesh


class XFEModel(FEModel):
    def __init__(
        self,
        nodes,
        elements,
        materials,
        reals,
        tip_enrichment=False,
        geometrical_range=0.0,
        corrected=False,
    ):
        FEModel.__init__(self, nodes, elements, materials, reals)
        self.base_list_dof = None
        self.level_set = []
        self.level_sets = []
        self.cut_info = {}
        self.tip_enrichment = tip_enrichment
        self.geometrical_range = geometrical_range
        self.ls = np.zeros(self.n_nodes, dtype=np.int32)
        self.tip = np.zeros(self.n_nodes, dtype=np.int32)
        self.corrected = corrected
        self.mesh = None
        self.mesh_surface = None
        self.degenerate_quads = None
        if corrected:
            self.in_range = np.zeros(self.n_nodes, dtype=np.int32)
        else:
            self.in_range = np.ones(self.n_nodes, dtype=np.int32)

    def gen_list_dof(self, dof_per_node):
        if dof_per_node == dofs.IS_2D:
            IS_BRANCH = dofs.IS_2D_BRANCH
            IS_HEAVISIDE = dofs.IS_2D_HEAVISIDE
        elif dof_per_node == dofs.IS_3D:
            IS_BRANCH = dofs.IS_3D_BRANCH
            IS_HEAVISIDE = dofs.IS_3D_HEAVISIDE
        else:
            raise NotImplementedError
        self.dof_per_node = dof_per_node
        model.gen_list_dof(self, dof_per_node)
        assert self.list_dof is not None
        assert self.level_sets
        partial_cuts = []
        degenerate_quads = []
        diagonal_nodes = []
        for elem in self.elements:
            id = elem[0]
            nodes = np.asarray(elem[4])
            for i, ls in enumerate(self.level_sets):
                cut_type, tip, touching = ls.is_cut(elem)
                if cut_type != CutType.NONE:
                    if id in self.cut_info:
                        print("warning: element already cut by other level set")
                    phi_n, _ = ls.get(nodes, None)
                    is_zero = np.isclose(phi_n, 0.0, atol=1e-10)
                    if touching:
                        print("warning somethign is touching still")
                    if not touching and np.any(is_zero):
                        degenerate_quads.append(elem)
                        diagonal_nodes.append(np.where(is_zero)[0])
                    if cut_type == CutType.PARTIAL:
                        partial_cuts.append((id, (i, cut_type, tip)))
                        if self.tip_enrichment:
                            self.ls[nodes - 1] = i
                            self.tip[nodes - 1] = tip
                            self.cut_info[id] = (i, cut_type, tip)
                            self.list_dof.add_dofs(nodes, IS_BRANCH)
                            if self.corrected:
                                self.in_range[nodes - 1] = 1
                    else:
                        # TODO: pas echt aan het touchen als alle intersecties 0 of 1 zijn
                        # touching = np.argwhere(np.isclose(phi_n, 0))
                        if touching:
                            # print("touching")
                            filtered = nodes[
                                np.argwhere(np.isclose(phi_n, 0, atol=1e-10))
                            ]
                        else:
                            filtered = nodes
                        phi_n, _ = ls.get(nodes, None)
                        # print(
                        #     "id",
                        #     id,
                        #     "phi_n",
                        #     np.array_repr(phi_n),
                        #     "phi_t",
                        #     np.array_repr(phi_t),
                        # )

                        self.list_dof.add_dofs(filtered, IS_HEAVISIDE)
                        if self.tip_enrichment:
                            is_in_range, tip, in_range = ls.in_range(
                                elem, self.geometrical_range
                            )
                            if is_in_range:
                                self.tip[nodes - 1] = tip

                                if self.corrected:
                                    self.in_range[nodes - 1] |= in_range
                                    self.list_dof.add_dofs(nodes, IS_BRANCH)
                                else:
                                    self.list_dof.add_dofs(nodes[in_range], IS_BRANCH)

                        self.ls[nodes - 1] = i
                        self.cut_info[id] = (i, cut_type, tip)
                else:
                    if self.tip_enrichment:
                        is_in_range, tip, in_range = ls.in_range(
                            elem, self.geometrical_range
                        )
                        if is_in_range:
                            self.tip[nodes - 1] = tip
                            self.ls[nodes - 1] = i
                            self.cut_info[id] = (i, CutType.NONE, tip)
                            if self.corrected:
                                self.in_range[nodes - 1] |= in_range
                                self.list_dof.add_dofs(nodes, IS_BRANCH)
                            else:
                                self.list_dof.add_dofs(nodes[in_range], IS_BRANCH)
            # for elem_id, ci in self.cut_info.items():
        for elem_id, ci in partial_cuts:
            i, cut_type, tip = ci
            # print(cut_type)
            if cut_type == CutType.PARTIAL:
                nodes = self.elements[elem_id - 1][4]
                nodes = np.asarray(nodes)
                self.tip[nodes - 1] = tip
                self.ls[nodes - 1] = i
                self.list_dof.remove_dofs(nodes, IS_HEAVISIDE)

        self.degenerate_quads = degenerate_quads
        # tri_id = self.elements[-1][0] + 1
        # for elem in degenerate_quads:
        #     nodes = elem[4]

        self.list_dof.update()

    def cal_global_matrices(self, elem, eval_mass=False, skip_elements={}):
        asm.cal_KgMg(
            self,
            elem,
            eval_mass=eval_mass,
            xfem=True,
            tip_enrich=self.tip_enrichment,
            corrected=self.corrected,
            skip_elements=skip_elements,
        )

    def insert_crack_segment(self, p1: NDArray, p2: NDArray, embedded):
        ls = LevelSet()
        ls.gen_from_line_segment(self.nodes, p1, p2, embedded=embedded)
        self.level_sets.append(ls)
        self.mesh = build_mesh(self.nodes, self.elements)

    def insert_crack_spline(self, bspline, embedded, h=0.05, snapping_tolerance=0.03):
        ls = LevelSet()
        ls.gen_from_bspline(
            self.nodes,
            bspline,
            h=h,
            geometrical_range=1.5 * self.geometrical_range,
            embedded=embedded,
            snapping_tolerance=snapping_tolerance,
        )
        self.level_sets.append(ls)
        self.mesh = build_mesh(self.nodes, self.elements)

    def insert_planar_crack_segment(self, p1, p2, p3, embedded):
        ls = LevelSet()
        ls.gen_from_plane(self.nodes, p1, p2, p3, embedded)
        self.level_sets.append(ls)
        self.mesh = build_mesh(self.nodes, self.elements)
        self.mesh_surface = self.mesh.extract_surface().triangulate()

    def insert_ndbsplines_crack(self, ndbsplines, h, snapping_tolerance=0.03):
        ls = LevelSet()
        ls.gen_from_ndbsplines(
            self.nodes,
            ndbsplines,
            h,
            self.geometrical_range,
            snapping_tolerance=snapping_tolerance,
        )
        self.level_sets.append(ls)
        self.mesh = build_mesh(self.nodes, self.elements)
        self.mesh_surface = self.mesh.extract_surface().triangulate()
