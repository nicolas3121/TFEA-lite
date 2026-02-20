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


BASE_DOFS: Final = DofType.UX | DofType.UY | DofType.UZ
HEAVISIDE_DOFS: Final = DofType.HX | DofType.HY | DofType.HZ
BRANCH_DOFS: Final = (
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

    def __len__(self):
        return self.n_dof

    def __bool__(self):
        return self.n_dof != 0

    def update(self):
        total = np.cumsum(np.bitwise_count(self.list_dof))
        self.n_dof = total[-1]
        self.list_dof_number = np.zeros_like(self.list_dof)
        self.list_dof_number[1:] = total[:-1]

    def get_elem_dofs(self, nodes):
        nodes = np.asarray(nodes)
        dofs = self.list_dof[nodes - 1]
        return dofs

    def get_elem_dof_numbers(self, nodes, mask):
        nodes = np.asarray(nodes)
        dofs = self.list_dof[nodes - 1]
        selected = np.bitwise_and(dofs, mask)
        preceding_mask = (selected[0] & ((~selected[0]) + 1)) - 1
        offset = np.bitwise_count(dofs & preceding_mask)
        dof_numbers = self.list_dof_number[nodes - 1]
        return (
            dof_numbers[:, None]
            + offset[:, None]
            + np.arange(selected[0].bit_count(), dtype=int)
        )

    def get_elem_dof_numbers_flat(self, nodes, mask):
        nodes = np.asarray(nodes)
        dofs = self.list_dof[nodes - 1]
        selected = np.bitwise_and(dofs, mask)
        preceding_mask = (selected & ((~selected) + 1)) - 1
        offset = np.bitwise_count(dofs & preceding_mask)
        dof_numbers = self.list_dof_number[nodes - 1]
        start = dof_numbers + offset
        selected_count = np.bitwise_count(selected)
        if np.any(selected):
            return np.concatenate(
                [
                    start[i] + np.arange(selected_count[i], dtype=int)
                    for i in range(len(nodes))
                    if selected_count[i] != 0
                ]
            )
        else:
            return np.array([], dtype=int)

        # return np(
        #     dof_numbers[:, None]
        #     + offset[:, None]
        #     + np.arange(selected[0].bit_count(), dtype=int)
        # )

    def get_dofs(self, node):
        return self.list_dof[node - 1]

    # def set_dofs(self, node, dofs):
    #     self.list_dof[node - 1] = dofs
    #
    def add_dofs(self, nodes, dofs):
        self.list_dof[np.asarray(nodes) - 1] |= dofs

    def remove_dofs(self, nodes, dofs):
        self.list_dof[np.asarray(nodes) - 1] &= ~dofs

    # def query_dofs(self, node, query):
    #     return self.list_dof[node - 1] & query == query
    #
    # def query_nodes(self, query): ...
    #
    #


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
IS_2D_ALL = IS_2D | IS_2D_HEAVISIDE | IS_2D_BRANCH


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
IS_3D_ALL = IS_3D | IS_3D_HEAVISIDE | IS_3D_BRANCH
