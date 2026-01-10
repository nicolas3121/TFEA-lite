import numpy as np
from .Quad4n import Quad4n


class XQuad4n(Quad4n):
    def __init__(self, node_coords, material, real, enrich: bool):
        Quad4n.__init__(self, node_coords, material, real)
        self.enrich = enrich
