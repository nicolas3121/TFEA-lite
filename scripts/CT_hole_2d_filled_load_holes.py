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
W = 40.0
H = 40.0

dist_opp_edge = 23.0
notch_w = 3.0
v_depth = notch_w / 2

K = 7.7
C = 6.7

hole_x = W - dist_opp_edge + K
hole_y = C
hole_r = 3.5

load_hole_x = 10.5
load_hole_y = 20 - 9.2
load_hole_r = 9.5 / 2

lc = 2.0

# 3. Build the Geometry using OpenCASCADE
plate = gmsh.model.occ.addRectangle(0, -H / 2, 0, W, H)
main_hole = gmsh.model.occ.addDisk(hole_x, hole_y, 0, hole_r, hole_r)

top_pin = gmsh.model.occ.addDisk(load_hole_x, load_hole_y, 0, load_hole_r, load_hole_r)
bot_pin = gmsh.model.occ.addDisk(load_hole_x, -load_hole_y, 0, load_hole_r, load_hole_r)

# --- Build the V-Notch ---
p1 = gmsh.model.occ.addPoint(0, notch_w / 2, 0)
p2 = gmsh.model.occ.addPoint(W - dist_opp_edge - v_depth, notch_w / 2, 0)
p3 = gmsh.model.occ.addPoint(W - dist_opp_edge, 0, 0)
p4 = gmsh.model.occ.addPoint(W - dist_opp_edge - v_depth, -notch_w / 2, 0)
p5 = gmsh.model.occ.addPoint(0, -notch_w / 2, 0)

l1 = gmsh.model.occ.addLine(p1, p2)
l2 = gmsh.model.occ.addLine(p2, p3)
l3 = gmsh.model.occ.addLine(p3, p4)
l4 = gmsh.model.occ.addLine(p4, p5)
l5 = gmsh.model.occ.addLine(p5, p1)

cl = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4, l5])
notch = gmsh.model.occ.addPlaneSurface([cl])

# --- BOOLEAN OPERATIONS ---
# Step A: Cut OUT the main hole and V-notch (empty space)
cut_out, _ = gmsh.model.occ.cut([(2, plate)], [(2, main_hole), (2, notch)])
plate_tag = cut_out[0][1]

# Step B: FRAGMENT the plate with the pins (creates shared boundaries, keeps all surfaces)
gmsh.model.occ.fragment([(2, plate_tag)], [(2, top_pin), (2, bot_pin)])

# Step C: Add exact center points for your constraints
pt_top = gmsh.model.occ.addPoint(load_hole_x, load_hole_y, 0)
pt_bot = gmsh.model.occ.addPoint(load_hole_x, -load_hole_y, 0)

gmsh.model.occ.synchronize()

# --- DYNAMIC SORTING & EMBEDDING ---
surfaces = gmsh.model.getEntities(2)
plate_surfs = []
pin_surfs = []

# Dynamically sort surfaces by looking at their center of mass
for dim, tag in surfaces:
    com = gmsh.model.occ.getCenterOfMass(dim, tag)
    # Check if it's the top pin
    if abs(com[0] - load_hole_x) < 1e-3 and abs(com[1] - load_hole_y) < 1e-3:
        pin_surfs.append(tag)
        gmsh.model.mesh.embed(0, [pt_top], 2, tag)  # Force node at center
    # Check if it's the bottom pin
    elif abs(com[0] - load_hole_x) < 1e-3 and abs(com[1] - (-load_hole_y)) < 1e-3:
        pin_surfs.append(tag)
        gmsh.model.mesh.embed(0, [pt_bot], 2, tag)  # Force node at center
    # Otherwise, it's the plate
    else:
        plate_surfs.append(tag)

# --- CREATE PHYSICAL GROUPS ---
# 0D Points (Tag 100 for Top, Tag 200 for Bottom)
gmsh.model.addPhysicalGroup(0, [pt_top], tag=100, name="Center_Top")
gmsh.model.addPhysicalGroup(0, [pt_bot], tag=200, name="Center_Bot")

# 2D Surfaces (Tag 1 for Plate, Tag 2 for Pins)
gmsh.model.addPhysicalGroup(2, plate_surfs, tag=1, name="Plate_Elements")
gmsh.model.addPhysicalGroup(2, pin_surfs, tag=2, name="Pin_Elements")


# 4. Set up Quad-Dominant Meshing
for dim, tag in surfaces:
    gmsh.model.mesh.setRecombine(dim, tag)

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
gmsh.fltk.run()
gmsh.finalize()
