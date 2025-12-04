from collections import Counter

def model_print(model):
    print(f'=> Model created successfuly, including:')
    print(f'   - {model.n_nodes} nodes')
    n_elements = len(model.elements)
    counts = Counter([etype for _, etype, *_ in model.elements])
    details = ", ".join(f"{counts[t]} {t}" for t in sorted(counts))
    print(f"   - {n_elements} elements, including: {details}")

def gen_list_dof(
        model,
        dof_per_node = ['ux', 'uy', 'uz']
):
    model.dof_per_node = dof_per_node
    model.list_dof = {}
    counter = 0
    for node in model.nodes:
        for dof in dof_per_node:
            model.list_dof |= {str(int(node[0])) + dof: counter}
            counter += 1
    model.n_dof = len(model.list_dof)