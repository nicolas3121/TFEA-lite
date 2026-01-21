from numpy.typing import NDArray
from .FEModel import FEModel
from .core import level_set
from .core import model
from .core import assembly as asm
import copy


class XFEModel(FEModel):
    def __init__(
        self,
        nodes,
        elements,
        materials,
        reals,
    ):
        FEModel.__init__(self, nodes, elements, materials, reals)
        self.base_list_dof = None
        self.level_set = []

    def gen_list_dof(self, dof_per_node=["ux", "uy", "uz"]):
        if self.base_list_dof is None:
            model.gen_list_dof(self, dof_per_node=dof_per_node)
            self.base_list_dof = copy.deepcopy(self.list_dof)
        else:
            self.list_dof = copy.deepcopy(self.base_list_dof)
        cut_elem = self.level_set[2]
        self.level_set[3]
        assert self.list_dof
        assert self.n_dof
        counter = self.n_dof
        for id in cut_elem:
            element = self.elements[id - 1]
            assert id == element[0]
            nodes = element[4]
            for node in nodes:
                for dof in dof_per_node:
                    key = str(int(node)) + dof + "H"
                    if key not in self.list_dof:
                        self.list_dof |= {key: counter}
                        counter += 1
        # for id in partial_cut_elem:
        #     element = self.elements[id - 1]
        #     assert id == element[0]
        #     nodes = element[4]
        #     for node in nodes:
        #         for dof in dof_per_node:
        #             for i in range(4):
        #                 key = str(int(node)) + dof + "B" + str(i)
        #                 if key not in self.list_dof:
        #                     self.list_dof |= {key: counter}
        #                     counter += 1
        self.n_dof = len(self.list_dof)

    def cal_global_matrices(self, eval_mass=False, skip_elements={}):
        asm.cal_KgMg_XFEM(self, eval_mass=eval_mass, skip_elements=skip_elements)

    def insert_crack_segment(self, p1: NDArray, p2: NDArray):
        self.level_set = level_set.gen_from_line_segment(
            self.nodes, self.elements, p1, p2
        )
