import numpy as np
from .core import model
from .core import bc
from .core import assembly as asm
from .core import solver as sol
from .core import stress as sts
from .visualization import visualization as vis

class FEModel:

    def __init__(self, nodes, elements, materials, reals):
        self.nodes = np.array(nodes)
        self.elements = elements
        self.materials = materials
        self.reals = reals
        self.n_nodes = len(self.nodes)
        self.n_elements = len(self.elements)
        model.model_print(self)
    
    def gen_list_dof(
        self,
        dof_per_node = ['ux', 'uy', 'uz']
    ):
        model.gen_list_dof(self, dof_per_node = dof_per_node)

    def cal_global_matrices(self, eval_mass = False, skip_elements = {}):
        asm.cal_KgMg(self, eval_mass = eval_mass, skip_elements = skip_elements)
    
    def gen_P(self, fix_dofs):
        bc.dirichlet_Lagrange_II(self, fix_dofs)

    def solve_static(self, Fg = []):
        sol.static(self, Fg = Fg)
        
    def cal_Tetr4n_stresses(self):
        return sts.cal_Tetr4n_stresses(self)
    
    def compute_quad4n_nodal_stresses(self, Ug = None):
        return sts.compute_quad4n_nodal_stresses(self, Ug = Ug)

    def eval_node_average(self, sxx):
        return sts.eval_node_average(self, sxx)

    def build_gcs(self, length = 1.0):
        vis.build_gcs(self, length = length)
    
    def show(
            self,
            gcs_length = 0.0,
            show_edges = True,
            node_size = 0.0,
            nid_size = 0.0,
            eid_size = 0.0,
            Ug = None,
            nbc_size = 0.0,
            node_stress = None,
            clim = None,
            load_size = None,
            window_size = None,
            show_elements = True,
            file_name = None,
            show_axes = False,
            colorbar_title = 'Stress',
            show_undef = False
        ):
        vis.show(
            self,
            gcs_length = gcs_length,
            show_edges = show_edges,
            node_size = node_size,
            nid_size = nid_size,
            eid_size = eid_size,
            Ug = Ug,
            nbc_size = nbc_size,
            node_stress = node_stress,
            clim = clim,
            load_size = load_size,
            window_size = window_size,
            show_elements = show_elements,
            file_name = file_name,
            show_axes = show_axes,
            colorbar_title = colorbar_title,
            show_undef = show_undef
        )