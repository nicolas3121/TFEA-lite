import gmsh

# CTS01: K = 8.3, C = 8.1
# CTS02: K = 8.4, C = 6.9
# CTS03: K = 8.1, C = 8.1
# CTS04: K = 7.7, C = 6.7
# thickness plate is 8mm

# 1. Initialize Gmsh
gmsh.initialize()
gmsh.model.add("MCTS_Quad_Mesh_Full")

# 2. Define Parameters (in millimeters)
W = 40.0  # Width of the specimen
H = 40.0  # Total height of the specimen

# --- V-Notch Parameters ---
dist_opp_edge = 23.0  # Distance from the V-notch tip to the right edge

notch_w = 3.0  # Width of the machined slot
v_depth = notch_w / 2  # notch has 90° angle

K = 8.1  # X distance hole center from notch tip
C = 8.1  # Y distance hole center from notch tip

hole_x = W - dist_opp_edge + K
hole_y = C
hole_r = 3.5

load_hole_x = 10.5
load_hole_y = 20 - 9.2  # Y distance from the center line
load_hole_r = 9.5 / 2

lc = 2.0  # Default coarse mesh size

# 3. Build the Geometry using OpenCASCADE
# Add the main plate
gmsh.model.occ.addRectangle(0, -H / 2, 0, W, H, tag=1)

# Add the main modifying hole
gmsh.model.occ.addDisk(hole_x, hole_y, 0, hole_r, hole_r, tag=2)

# --- Add the Loading Holes ---
gmsh.model.occ.addDisk(load_hole_x, load_hole_y, 0, load_hole_r, load_hole_r, tag=10)
gmsh.model.occ.addDisk(load_hole_x, -load_hole_y, 0, load_hole_r, load_hole_r, tag=11)

# --- Build the V-Notch ---
p1 = gmsh.model.occ.addPoint(0, notch_w / 2, 0)
p2 = gmsh.model.occ.addPoint(W - dist_opp_edge - v_depth, notch_w / 2, 0)
p3 = gmsh.model.occ.addPoint(W - dist_opp_edge, 0, 0)
p4 = gmsh.model.occ.addPoint(W - dist_opp_edge - v_depth, -notch_w / 2, 0)
p5 = gmsh.model.occ.addPoint(0, -notch_w / 2, 0)

# Connect the points with lines
l1 = gmsh.model.occ.addLine(p1, p2)
l2 = gmsh.model.occ.addLine(p2, p3)
l3 = gmsh.model.occ.addLine(p3, p4)
l4 = gmsh.model.occ.addLine(p4, p5)
l5 = gmsh.model.occ.addLine(p5, p1)

# Create the surface from the lines to form the notch shape
cl = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4, l5])
notch_tag = gmsh.model.occ.addPlaneSurface([cl])

# Perform Boolean Operations: Plate MINUS (Main Hole AND V-Notch AND Both Loading Holes)
# We force the resulting cut surface to be tag 4 so it matches our recombine command
gmsh.model.occ.cut([(2, 1)], [(2, 2), (2, notch_tag), (2, 10), (2, 11)], tag=4)

# Synchronize the CAD kernel with the Gmsh mesh model
gmsh.model.occ.synchronize()

# 4. Set up Quad-Dominant Meshing
gmsh.model.mesh.setRecombine(2, 4)
gmsh.option.setNumber("Mesh.Algorithm", 8)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)

# 5. Define the Dense Crack Path (Box Field)
gmsh.model.mesh.field.add("Box", 1)

gmsh.model.mesh.field.setNumber(1, "VIn", 0.2)
gmsh.model.mesh.field.setNumber(1, "VOut", 2.0)

gmsh.model.mesh.field.setNumber(1, "XMin", W - dist_opp_edge - 2.0)
gmsh.model.mesh.field.setNumber(1, "XMax", W)
gmsh.model.mesh.field.setNumber(1, "YMin", -10.0)
gmsh.model.mesh.field.setNumber(1, "YMax", hole_y + 5.0)
gmsh.model.mesh.field.setNumber(1, "Thickness", 8.0)

gmsh.model.mesh.field.setAsBackgroundMesh(1)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

# 6. Generate and Save the Mesh
gmsh.model.mesh.generate(2)
gmsh.write("mcts_xfem_mesh.msh")

# 7. Launch the GUI to inspect the result
gmsh.fltk.run()

# 8. Clean up
gmsh.finalize()
