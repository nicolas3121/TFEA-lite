from enum import IntFlag, auto
import numpy as np
from typing import Final, Tuple


# if add extra dofs check if still fits in 32 bits otherwise otherwise increase size in model.gen_list_dof
class DofType(IntFlag):
    UX = auto()
    UY = auto()
    UZ = auto()
    HX = auto()
    HY = auto()
    HZ = auto()
    B1X = auto()
    B1Y = auto()
    B1Z = auto()
    B2X = auto()
    B2Y = auto()
    B2Z = auto()
    B3X = auto()
    B3Y = auto()
    B3Z = auto()
    B4X = auto()
    B4Y = auto()
    B4Z = auto()


# BASE_DOFS: Final = DofType.UX | DofType.UY | DofType.UZ
# HEAVISIDE_DOFS: Final = DofType.HX | DofType.HY | DofType.HZ
# BRANCH_DOFS: Final = (
#     DofType.B1X
#     | DofType.B1Y
#     | DofType.B1Z
#     | DofType.B2X
#     | DofType.B2Y
#     | DofType.B2Z
#     | DofType.B3X
#     | DofType.B3Y
#     | DofType.B3Z
#     | DofType.B4X
#     | DofType.B4Y
#     | DofType.B4Z
# )


class DofList:
    def __init__(self, n_nodes, dof_per_node):
        self.dof_per_node = dof_per_node
        self.list_dof = np.bitwise_or(np.zeros(n_nodes, dtype=np.uint32), dof_per_node)
        self.update()

    def get(self, node, dof: DofType):
        assert dof.bit_count() == 1, "can only access one dof number at a time"
        dofs = self.list_dof[node - 1]  # nodes start at 1
        dof_number = self.list_dof_number[node - 1]
        if dofs & dof == dof:
            return dof_number + (dofs & (dof - 1)).bit_count()
        else:
            return None

    def __getitem__(self, key: Tuple[int, DofType]):
        node = key[0]
        dof = key[1]
        res = self.get(node, dof)
        assert res is not None, "node doesn't have requested dof"
        return res

    def update(self):
        total = np.cumsum(np.bitwise_count(self.list_dof))
        self.n_dof = total[-1]
        self.list_dof_number = np.zeros_like(self.list_dof)
        self.list_dof_number[1:] = total[:-1]


IS_2D: Final = DofType.UX | DofType.UY
IS_2D_HEAVISIDE: Final = DofType.HX | DofType.HY
IS_2D_BRANCH: Final = (
    DofType.B1X
    | DofType.B1Y
    | DofType.B2X
    | DofType.B2Y
    | DofType.B3X
    | DofType.B3Y
    | DofType.B4X
    | DofType.B4Y
)


IS_3D: Final = DofType.UX | DofType.UY | DofType.UZ
IS_3D_HEAVISIDE: Final = DofType.HX | DofType.HY | DofType.HZ
IS_3D_BRANCH: Final = (
    DofType.B1X
    | DofType.B1Y
    | DofType.B1Z
    | DofType.B2X
    | DofType.B2Y
    | DofType.B2Z
    | DofType.B3X
    | DofType.B3Y
    | DofType.B3Z
    | DofType.B4X
    | DofType.B4Y
    | DofType.B4Z
)
