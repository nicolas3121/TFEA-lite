from numpy.typing import NDArray
import numpy as np
from .FEModel import FEModel
from .core.level_set import LevelSet, CutType
from .core import model
from .core import assembly as asm
from .core import dofs


class XFEModel(FEModel):
    def __init__(
        self,
        nodes,
        elements,
        materials,
        reals,
        tip_enrichment=False,
        geometrical_range=0.0,
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

    def gen_list_dof(self, dof_per_node):
        self.dof_per_node = dof_per_node
        model.gen_list_dof(self, dof_per_node)
        assert self.list_dof is not None
        assert self.level_sets
        for elem in self.elements:
            id = elem[0]
            nodes = np.asarray(elem[4])
            for i, ls in enumerate(self.level_sets):
                cut_type, tip = ls.is_cut(elem)
                if cut_type != CutType.NONE:
                    if id in self.cut_info:
                        print("warning: element already cut by other level set")
                    if cut_type == CutType.PARTIAL:
                        if self.tip_enrichment:
                            self.ls[nodes - 1] = i
                            self.tip[nodes - 1] = tip
                            self.cut_info[id] = (i, cut_type, tip)
                            self.list_dof.add_dofs(nodes, dofs.IS_2D_BRANCH)
                    else:
                        phi_n, _ = ls.get(nodes, None)
                        touching = np.argwhere(np.isclose(phi_n, 0))
                        if len(touching):
                            print("touching")
                            filtered = nodes[touching]
                        else:
                            filtered = nodes

                        self.list_dof.add_dofs(filtered, dofs.IS_2D_HEAVISIDE)
                        if self.tip_enrichment:
                            in_range, tip = ls.in_range(elem, self.geometrical_range)
                            if in_range:
                                self.tip[nodes - 1] = tip
                                self.list_dof.add_dofs(nodes, dofs.IS_2D_BRANCH)
                        self.ls[nodes - 1] = i
                        self.cut_info[id] = (i, cut_type, tip)
                else:
                    if self.tip_enrichment:
                        in_range, tip = ls.in_range(elem, self.geometrical_range)
                        if in_range:
                            self.tip[nodes - 1] = tip
                            self.ls[nodes - 1] = i
                            self.list_dof.add_dofs(nodes, dofs.IS_2D_BRANCH)
                            self.cut_info[id] = (i, CutType.NONE, tip)
        for elem_id, ci in self.cut_info.items():
            i, cut_type, tip = ci
            print(cut_type)
            if cut_type == CutType.PARTIAL:
                nodes = self.elements[elem_id - 1][4]
                nodes = np.asarray(nodes)
                self.tip[nodes - 1] = tip
                self.ls[nodes - 1] = i
                self.list_dof.remove_dofs(nodes, dofs.IS_2D_HEAVISIDE)

        self.list_dof.update()

    # def gen_list_dof(self, dof_per_node):
    #     self.dof_per_node = dof_per_node
    #     model.gen_list_dof(self, dof_per_node)
    #     cut_elem = self.level_set[2]
    #     partial_cut_elem = self.level_set[3]
    #     assert self.list_dof is not None
    #     for id in cut_elem:
    #         element = self.elements[id - 1]
    #         assert id == element[0]
    #         nodes = element[4]
    #         self.list_dof.add_dofs(nodes, dofs.IS_2D_HEAVISIDE)
    #     if self.tip_enrichment:
    #         for id in partial_cut_elem:
    #             element = self.elements[id - 1]
    #             assert id == element[0]
    #             nodes = element[4]
    #             self.list_dof.add_dofs(nodes, dofs.IS_2D_BRANCH)
    #     self.list_dof.update()

    def cal_global_matrices(self, elem, eval_mass=False, skip_elements={}):
        asm.cal_KgMg(
            self,
            elem,
            eval_mass=eval_mass,
            xfem=True,
            tip_enrich=self.tip_enrichment,
            skip_elements=skip_elements,
        )

    # def insert_crack_segment(self, p1: NDArray, p2: NDArray):
    #     self.level_set = level_set.gen_from_line_segment(
    #         self.nodes, self.elements, p1, p2
    #     )

    def insert_crack_segment(self, p1: NDArray, p2: NDArray, embedded):
        ls = LevelSet()
        ls.gen_from_line_segment(self.nodes, p1, p2, embedded=embedded)
        self.level_sets.append(ls)
