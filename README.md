
![Schermafbeelding 2025-03-06 165441](https://github.com/user-attachments/assets/f6368a20-fc94-4980-8f83-79356945029a)

# TFEA-lite - A lightweight public subset of the TFEA (Toolbox for Finite Element Analysis)

**TFEA-lite** is a lightweight public subset of the ```TFEA``` (Toolbox for Fininte Element Analysis) codebase (2022–now). It provides a compact and scriptable environment for small-strain, linear elastic finite element analysis, suitable for research prototyping, teaching, and reproducible examples. ```TFEA-lite``` contains only the essential element types and regular solvers but provides a cleaner and more powerful high-level API that makes modelling, analysis and visualization significantly easier for users.

The full ```TFEA``` framework remains private at this stage due to ongoing maintenance and active development. For background theories on the finite element formulations used in this tool, refer to **Songhan Zhang, _Finite Element Method - Lecture Notes_, 2021** [Download](https://drive.google.com/file/d/14nab0rflYc-9BXTZNoSYYE6DMlTvOKxm/view?usp=drive_link). If you are interested in exploring more advanced modules (or Julia version), please feel free to contact [Songhan Zhang](mailto:songhan.zhang@ugent.be).

## Preparation
Before use, please check if the following have been well prepared:
 - vs code (recommended) or other IDEs
 - python (3.12 or higher) or Julia (1.10 or higher)
 - conda (recommened) or .venv

## Installation
The installation via [conda](https://www.anaconda.com/download) is the most recommended.
For Windows, Mac OS and Linux users, the miniconda can always be downloaded from the [link].
After installation, you can start ```miniconda``` and install ```TFEA-lite``` following the steps:
### Step 1: Create a new environment
```cmd
conda create -n myenv python=3.12
```
where ```myenv``` is the name of the virtual environment.
You may use any preferred name instead, just remember it well.
### Step 2: Activate the environment
```cmd
conda activate myenv
```
### Step 3: Install TFEA into the environment
```cmd
pip install git+https://github.com/songhanzhang/TFEA-lite
```
### Step 4: Validity check
Create a new .py file in ```VS Code```, and write the script:
```python
import tfealite as tf
```
Select the interpreter ```python x.xx.xx (myenv)``` and run.
If the package can be imported without any error message, it is confirmed that the package is installed successfully.

## Quick start

### Node definition
The node coordinates are stored in a 4-column numpy array. Each row takes the form ```[ node_id, $x$, $y$, $z$ ]```. If the model is planar, it is required to take ```z = 0``` for all the nodes. Note that the node_id is 1-based, not 0-based. For example:

```python
import numpy as np
nodes = np.array([
    [ 1, 0.0, 0.0, 0.0 ],
    [ 2, 1.0, 0.0, 0.0 ],
    [ 3, 1.0, 1.0, 0.0 ],
    [ 4, 0.0, 1.0, 0.0 ],
    [ 5, 2.0, 0.0, 0.0 ],
    [ 6, 2.5, 1.3, 0.0 ],
])

```

### Element definition
The element information is stored in form of a python list. For each element, the properties includes: ```[ element_id, element_type, material_id, real_id, connectivity ]```. Note that the element_id is also 1-based. In the current lite version, the element types include:

 - ```Quad4n```: Bi-linear quadrilateral element (2D), each node includes the dofs: [ 'ux', 'uy' ]
 - ```Tetr4n```: Tetrahedral brick element (3D), each node includes the dofs: [ 'ux', 'uy', 'uz' ]

The material_id and real_id will be mentioned in the following sections. The connectivity is a tuple containing the node ids of the element vertices.
For example, a planar 'Quad4n' model can be defined as:

```python
elements = [
    [ 1, 'Quad4n', 1, 1, (1,3,4,2) ],
    [ 2, 'Quad4n', 2, 1, (3,5,6,4) ]
]
```

### Material properties
A model may involve a number of different material properties to be assigned into the elements (recall ```material_id``` in element information).
The material properties are collected into a list, each sublist include the material id and a dictionary.
Normally, the elastic modulus $E$, Poisson ratio $\nu$ and density $\rho$ should be provided.
For example, if we wish, respectively, to assign steel and rubber for element 1 and 2 defined above, the material properties can be defined as:

```python
materials = [
    [ 1, {'E': 2e11, 'nu': 0.33, 'rho': 7850} ],
    [ 2, {'E': 5e7, 'rho': 1100, 'nu': 0.49} ]
]
```

Note that, for each material, the definitions of the material properties can be disordered since they are collected in a dictionary. 

### Geometric properties
For a part of the elements, the geometric properties are required:

 - ```Quad4n```: thickness (t) (The value will be used only for plane stress problem when the external force is defined as lumped values)
For example, the geometric properties for ```Quad4n``` can be defined as:

```python
reals = [
    [ 1, {'t': 0.01} ]
]
```

### Compute stiffness (mass) matrices
The model has a global degree of freedom (DOF) list by assigning a set of DOFs to every node. Such DOF list is a dictionary that maps a string key (for example, "12ux" for node id 12, translational displacement in $x$) to a unique global DOF index (0-based). Users can create the list from:

```python
model.gen_list_dof(model, dof_per_node = ['ux', 'uy'])
```

Note that the definition of ```dof_per_node``` is not a mandatory input. The value is ```['ux', 'uy', 'uz']``` by default. After creating the DOF list, the stiffness (mass) matrices can be evaluated from:

```python
model.cal_global_matrices()
```

The mass matrix is not computed by default. If you wish to perform dynamic analysis, please specify ```eval_mass = True```
Both mass and stiffness matrices are stored in form of sparse matrices.

### Boundary conditions
The boundary conditions (B.C.s) are formulated based on the Lagrange's equation of the second kind (least number of DOFs). In the lite version, Dirichlet B.C.s can be easily applied by selecting nodes through a user-defined condition. A selection function ```sel_condition(x, y, z)``` returns a value indicating whether the node should be fixed or not. For the selected nodes, all the DOFs are constrained to zero. For example, to constrain the face at x = 0:

```python
def sel_condition(x,y,z):
    return x - 0.0
model.gen_dirichlet_bc(sel_condition)
```

For advanced modeling tasks, users can define a list ```fix_dofs``` by themselves, and generate the B.C.s alternatively through, for example:

```python
fix_dofs = []
fix_dofs.append(model.list_dof['2ux'])
fix_dofs.append(model.list_dof['2uy'])
fix_dofs.append(model.list_dof['5ux'])
fix_dofs.append(model.list_dof['5uz'])
fix_dofs.append(model.list_dof['5ux'])
fix_dofs.append(model.list_dof['5uz'])
model.gen_P(fix_dofs)
```

### Load definition
Nodal forces can be defined by selecting nodes through a user-defined condition and specifying a force expression.
The selection condition ```sel_condition(x, y, z)``` defines which nodes should receive the load, and the force expression ```force_expression(x, y, z)``` returns the force components ```(fx, fy, fz)``` to be applied at each selected node. For example, to apply a bending moment (with the stress gradient ```alpha```) at the face ```x = L``` (where neutral surface is ```z = 0.5```):

```python
def sel_condition(x, y, z):
    return x - L
def force_expression(x, y, z):
    return alpha*(z-0.5), 0.0, 0.0
model.gen_nodal_forces(sel_condition, force_expression)
```

### Solver
#### Static analysis
The statc displacement vector ```Ug``` can be obtained from the static solver:

```python
model.solve_static()
```

Note that the B.C.s must be properly defined before starting solving procedure. Otherwise the stiffness matrix is singular is the constraint is insufficient.

#### Model analysis
The modal solver returns the natural frequencies and mode shapes from:

```python
model.solve_modal(num_eigs = 10)
```

Users can specify the number of modes to be evaluated depending on their needs and computational resources. The value is ```num_eigs = 15``` by defult.

### Postprocessing
After solving the displacement vector, element/nodal stresses can be computed in the postprocessing module. For Tetr4n elements, the element stresses can be evaluated by:

```python
e_stress = model.cal_Tetr4n_stresses()
```

This returns an array of size (n_elements, 6), containing the standard Voigt components ```$(\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \tau_{xy}, \tau_{xz}, \tau_{yz})$``` for each element. A volume-weighted nodal average of a specified stress component can be computed using:

```python
n_stress = model_eval_node_average(e_stress[:,0])
```

This returns an array having the length ```n_nodes```. For the 2D Quad4n elements, nodal stresses can be evaluated directly:

```python
n_stress = model.compute_quad4n_nodal_stresses()
```

This returns a ```(n_nodes, 3)``` array containing the stress components ```$(\sigma_{xx}, \sigma_{yy}, \sigma_{zz})$```.

### Visualization
The method ```show()``` provides 2D/3D visualization of the mesh, boundary conditions, loads, deformations, nodal stresses. A minimum call is:

```python
model.show()
```

which displays the undeformed mesh with element edges. The optional parameters are listed below.

| Parameter       | Physical meaning                                                        | Default value |
|-----------------|-------------------------------------------------------------------------|-----------|
| `gcs_length`    | Length of the global coordinate system axes (X/Y/Z).                    | `0.0`     |
| `show_edges`    | Draw element edges (wireframe overlay).                                 | `True`    |
| `node_size`     | Size of the nodes.                                                      | `0.0`     |
| `nid_size`      | Font size for node ID labels.                                           | `0.0`     |
| `eid_size`      | Font size for element ID labels.                                        | `0.0`     |
| `Ug`            | Displacement vector for plotting the deformation.                       | `None`    |
| `nbc_size`      | Size of the fixed nodes (Dirichlet BC), shown in red.                   | `0.0`     |
| `node_stress`   | Array of nodal stress values for color contours.                        | `None`    |
| `clim`          | Color scale limits for stress contours.                                 | `None`    |
| `load_size`     | Tuple `(arrow_length, arrow_thickness)` for load visualization.         | `None`    |
| `window_size`   | Rendering window size `(width, height)`.                                | `(1024,768)` |
| `show_elements` | Whether to show elements.                                               | `True`    |
| `file_name`     | Saves as `file_name.png`.                                               | `None`    |
| `show_axes`     | Show the coordinate frame (pyvista style).                              | `False`   |
| `colorbar_title`| Title of the colorbar.                                                  | `"Stress"`|
| `show_undef`    | Plot undeformed mesh as wireframe style.                                | `False`   |

## Examples
You can find a few typical cases in the folder ```examples```.
These are recommended to refer as a template.

## License
Copyright (C) 2022-2025  Songhan Zhang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

## Acknowledgements
The development of the full TFEA framework has benefited from auxiliary contributions by my former students, including **Yong Zhao**, **Jun Cao**, **Xingchao Sun**, **Hui Jiang**, and **Miao Zhang**. Their assistance with testing and debugging tasks is gratefully acknowledged.