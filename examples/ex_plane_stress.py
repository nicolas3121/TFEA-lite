import tfealite as tf
import numpy as np

L = 1.0
H = 1.0
nx = 20
ny = 20

dx = L / nx
dy = H / ny

nodes = []
nid = 1
for j in range(ny + 1):
    y = j * dy
    for i in range(nx + 1):
        x = i * dx
        nodes.append([nid, x, y, 0.0])
        nid += 1
nodes = np.array(nodes, dtype = float)

elements = []
eid = 1
for j in range(ny):
    for i in range(nx):
        if i < 3 and j == 8:
            continue
        n1 = j * (nx + 1) + i + 1
        n2 = n1 + 1
        n3 = n2 + (nx + 1)
        n4 = n1 + (nx + 1)
        elements.append([eid, 'Quad4n', 1, 1, (n1, n2, n3, n4)])
        eid += 1

materials = [
    [1, {'E': 2e11, 'nu': 0.33, 'rho': 7850}]
]

reals = [
    [1, {'t': 0.01}]
]

model = tf.FEModel(nodes, elements, materials, reals)

model.gen_list_dof(dof_per_node = ['ux', 'uy'])
model.cal_global_matrices()

tol = 1e-8
fix_dofs = []
for node in model.nodes:
    nid = int(node[0])
    x, y, z = node[1:4]
    if abs(y - 0.0) < tol:
        fix_dofs.append(model.list_dof[f"{nid}ux"])
        fix_dofs.append(model.list_dof[f"{nid}uy"])

fix_dofs = np.array(fix_dofs, dtype = int)
model.gen_P(fix_dofs)

Fg = np.zeros(model.n_dof, dtype = float)

for node in model.nodes:
    nid = int(node[0])
    x, y, z = node[1:4]
    if abs(y - H) < tol:
        fy = np.sin(np.pi * x / L)
        dof_id = model.list_dof[f"{nid}uy"]
        Fg[dof_id] += fy

model.Fg = Fg

model.solve_static()
stress = model.compute_quad4n_nodal_stresses(Ug = model.Ug)

model.show(
    node_size = 10, 
    nbc_size = 15, 
    load_size = (0.8, 0.15), 
    Ug = 1e9 * model.Ug, 
    node_stress = stress[:,0]
)