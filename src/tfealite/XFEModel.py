from numpy.typing import NDArray
from .FEModel import FEModel
from .core import level_set
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
    ):
        FEModel.__init__(self, nodes, elements, materials, reals)
        self.base_list_dof = None
        self.level_set = []

    def gen_list_dof(self, dof_per_node):
        self.dof_per_node = dof_per_node
        model.gen_list_dof(self, dof_per_node)
        cut_elem = self.level_set[2]
        assert self.list_dof is not None
        for id in cut_elem:
            element = self.elements[id - 1]
            assert id == element[0]
            nodes = element[4]
            self.list_dof.add_dofs(nodes, dofs.IS_2D_HEAVISIDE)
        self.list_dof.update()

    def cal_global_matrices(self, elem, eval_mass=False, skip_elements={}):
        asm.cal_KgMg(
            self, elem, eval_mass=eval_mass, xfem=True, skip_elements=skip_elements
        )

    def insert_crack_segment(self, p1: NDArray, p2: NDArray):
        self.level_set = level_set.gen_from_line_segment(
            self.nodes, self.elements, p1, p2
        )
