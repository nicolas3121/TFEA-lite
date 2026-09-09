import gmsh

# 1. Initialize Gmsh
gmsh.initialize()
gmsh.model.add("Plate_With_Hole")

# 2. Define Parameters (in mm)
L = 125.0
W = 30.0

# Hole parameters
hole_r = 5.2
center_x = L / 2.0  # Center line at 62.5 (formerly notch_x)
hole_x = center_x - 9.3
hole_y = 14.8

# Loading positions
s_dist = 50.0
r_dist = 25.0

# Mesh sizes
lc_coarse = 1
lc_fine = 0.2

# 3. Build Main Geometry
rect = gmsh.model.occ.addRectangle(0, 0, 0, L, W)
hole = gmsh.model.occ.addDisk(hole_x, hole_y, 0, hole_r, hole_r)

# Perform Cut: Plate - Hole (Notch removed here)
out_tags, _ = gmsh.model.occ.cut([(2, rect)], [(2, hole)])
surf_tag = out_tags[0][1]

# 4. Add Loading Points
p_top1 = gmsh.model.occ.addPoint(center_x - r_dist, W, 0)
p_top2 = gmsh.model.occ.addPoint(center_x + r_dist, W, 0)
p_bot1 = gmsh.model.occ.addPoint(center_x - s_dist, 0, 0)
p_bot2 = gmsh.model.occ.addPoint(center_x + s_dist, 0, 0)

# 5. FRAGMENT: Embed the points into the boundaries
gmsh.model.occ.fragment(
    [(2, surf_tag)], [(0, p_top1), (0, p_top2), (0, p_bot1), (0, p_bot2)]
)
gmsh.model.occ.synchronize()

# Fetch the exact node tags after the fragment operation
pt1 = gmsh.model.getEntitiesInBoundingBox(
    center_x - r_dist - 0.1, W - 0.1, -0.1, center_x - r_dist + 0.1, W + 0.1, 0.1, 0
)
pt2 = gmsh.model.getEntitiesInBoundingBox(
    center_x + r_dist - 0.1, W - 0.1, -0.1, center_x + r_dist + 0.1, W + 0.1, 0.1, 0
)
pb1 = gmsh.model.getEntitiesInBoundingBox(
    center_x - s_dist - 0.1, -0.1, -0.1, center_x - s_dist + 0.1, 0.1, 0.1, 0
)
pb2 = gmsh.model.getEntitiesInBoundingBox(
    center_x + s_dist - 0.1, -0.1, -0.1, center_x + s_dist + 0.1, 0.1, 0.1, 0
)

# 6. Physical Groups for BCs (EXPLICIT TAGS ADDED)
# We give the boundary nodes tags 10 and 11 so they don't interfere with the element tags
gmsh.model.addPhysicalGroup(0, [pt1[0][1], pt2[0][1]], tag=10, name="Load_Points")
gmsh.model.addPhysicalGroup(0, [pb1[0][1], pb2[0][1]], tag=11, name="Support_Points")

# Get updated surface tags after fragment operation
surfaces = gmsh.model.getEntities(2)

# EXPLICITLY set tag=1 so your meshio snippet extracts "1" for the Plate elements
gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], tag=1, name="Plate")

# 7. Mesh Refinement (Box Field)
gmsh.model.mesh.field.add("Box", 1)
gmsh.model.mesh.field.setNumber(1, "VIn", lc_fine)
gmsh.model.mesh.field.setNumber(1, "VOut", lc_coarse)
gmsh.model.mesh.field.setNumber(1, "XMin", center_x - 20)
gmsh.model.mesh.field.setNumber(1, "XMax", center_x + 10)
gmsh.model.mesh.field.setNumber(1, "YMin", 0)
gmsh.model.mesh.field.setNumber(1, "YMax", 25)
gmsh.model.mesh.field.setAsBackgroundMesh(1)

# 8. Mesh Generation Options
gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads
gmsh.model.mesh.setRecombine(2, surfaces[0][1])  # Recombine to quads
# gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)  # 3 = Blossom-Quad

gmsh.option.setNumber("Mesh.SurfaceEdges", 1)
gmsh.option.setNumber("Mesh.SurfaceFaces", 1)

# 9. Generate and Save
gmsh.model.mesh.generate(2)
gmsh.write("modified_sen_hole_2d.msh")

# Launch GUI
gmsh.fltk.run()
gmsh.finalize()
