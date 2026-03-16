import numpy as np
from .level_set import LevelSet


class DisplacementCorrelationMethodSIF:
    def __init__(self, kosolov, shear_mod, r):
        self.kosolov = kosolov
        self.shear_mod = shear_mod
        self.r = r

    def cal_sif(self, level_set: LevelSet, Ug, elements, cut_info, tip_elem, tip):
        tip_nodes = np.asarray(tip_elem[4])
        tip_phi_n, tip_phi_t = level_set.get(tip_nodes, tip)
        # als pad colombo neemt voor local base crack front te bepalen heb shape
        # functies van element nodig
        # klein probleem momenteel want base element shape functies afhankelijk van element
        # kan niet statisch maken want dan kunnen child versies niet meer afhankelijk van self zijn
        # eens gradient bij tip heb naar projectie op crack binnen elementen beginnen berekenen om punt met juiste straal te zoeken
