
![Schermafbeelding 2025-03-06 165441](https://github.com/user-attachments/assets/f6368a20-fc94-4980-8f83-79356945029a)

# Toolbox for fininte element analysis (lite version)

**TFEA-lite** is a lightweight public subset of the ```TFEA``` (Toolbox for Fininte Element Analysis) codebase (2022–now). It provides a compact and scriptable environment for small-strain, linear elastic finite element analysis, suitable for research prototyping, teaching, and reproducible examples. ```TFEA-lite``` contains only the essential element types and regular solvers. The full ```TFEA``` framework remains private at this stage due to ongoing maintenance and active development. For background theories on the finite element formulations used in this tool, refer to  
**Songhan Zhang, _Finite Element Method - Lecture Notes_, 2021** [Download](https://drive.google.com/file/d/14nab0rflYc-9BXTZNoSYYE6DMlTvOKxm/view?usp=drive_link).

If you are interested in exploring more advanced modules (or Julia version), please feel free to contact the leading developer [Songhan Zhang](mailto:songhan.zhang@ugent.be).

## Preparation
Before use, please check if the following have been well prepared:
 - vs code
 - python or Julia
 - conda or .venv

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

## Start modeling (python version)
### Node definition
The node coordinates are stored in a 4-column numpy array.
Each row takes the form
[ node_id, $x$, $y$, $z$ ]
If the model is planar, it is required to take $$z = 0$$ for all the nodes.
Note that the node_id is 1-based, not 0-based.
For example:
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
The element information is stored in form of a python list. For each element, the properties includes:
[ element_id, element_type, material_id, real_id, connectivity ]
Note that the element_id is also 1-based. In the current lite version, the element types include:
 - 'Quad4n': Bi-linear quadrilateral element (2D), each node includes the dofs: [ 'ux', 'uy' ]
 - 'Tetr4n': Tetrahedral brick element (3D), each node includes the dofs: [ 'ux', 'uy', 'uz' ]
The material_id and real_id will be mentioned in the following sections. The connectivity is a tuple containing the node ids of the element vertices.
For example, a planar 'Quad4n' model can be defined as:
```python
elements = [
    [ 1, 'Quad4n', 1, 1, (1,3,4,2) ],
    [ 2, 'Quad4n', 2, 1, (3,5,6,4) ]
]
```
### Material properties
A model may involve a number of different material properties to be assigned into the elements (recall 'material_id' in element information).
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
 - 'Quad4n': thickness (t) (The value will be used only for plane stress problem when the external force is defined as lumped values)
For example, the geometric properties for 'Quad4n' can be defined as:
```python
reals = [
    [ 1, {'t': 0.01} ]
]
```
### Generate list of DOFs
### Boundary conditions
### Load definition
### Solver
### Postprocessing
### Visualization

## Examples
You can find a few typical cases in the folder ```examples```.
These are recommended to refer as a template.

## Acknowledgements
The development of the full TFEA framework has benefited from auxiliary contributions by my former students, including **Yong Zhao**, **Jun Cao**, **Xingchao Sun**, **Hui Jiang**, and **Miao Zhang**. Their assistance with testing and debugging tasks is gratefully acknowledged.