import csv

import numpy as np
from geomdl import knotvector
from scipy.interpolate import BSpline

from .XFEModel import XFEModel


class GrowthController:
    def __init__(
        self,
        nodes,
        elements,
        materials,
        reals,
        tip_enrichment,
        geometrical_range,
        corrected,
        h,
        embedded,
        dof_per_node,
        elem_dict,
        dcm,
        growth_direction_fn,
        da,
        control_points,
        k,
        bc_fn,
        force_fn,
        plot_fn,
    ):
        self.nodes = nodes
        self.elements = elements
        self.materials = materials
        self.reals = reals
        self.tip_enrichment = tip_enrichment
        self.geometrical_range = geometrical_range
        self.corrected = corrected
        self.h = h
        self.embedded = embedded
        self.dof_per_node = dof_per_node
        self.elem_dict = elem_dict
        self.dcm = dcm
        self.growth_direction_fn = growth_direction_fn
        self.da = da
        self.control_points = control_points
        self.k = k
        self.bc_fn = bc_fn
        self.force_fn = force_fn
        self.plot_fn = plot_fn
        self.K_I_tip_1 = []
        self.K_II_tip_1 = []
        self.K_I_tip_2 = []
        self.K_II_tip_2 = []
        self.control_points_history = [self.control_points]

    def _crack_growth(self, model, bspline):
        # 1. Primary Tip (s = 1)
        tip = bspline(1)

        # Calculate SIFs and get the true local tip coordinate system
        K_I, K_II, T_local = self.dcm.cal_sif(
            model.level_sets[0], model, model.cut_info, 1.0
        )

        # Extract the true tangent vector from the level set gradients
        # (Assuming T_local[0] is the t vector and T_local[1] is the n vector)
        t_tip_true = T_local[0]
        t_unit = t_tip_true / np.linalg.norm(t_tip_true)  # Normalize for safety

        self.K_I_tip_1.append(K_I)
        self.K_II_tip_1.append(K_II)

        theta = self.growth_direction_fn(K_I, K_II)
        print("K_I", K_I, "K_II", K_II)
        print("angle", np.degrees(theta))

        # Rotate the TRUE tip tangent by the calculated MTS angle
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        new_direction = rotation_matrix @ t_unit

        new_tip = tip + self.da * new_direction
        self.control_points.append(new_tip)

        # 2. Embedded Tip (s = 0)
        if self.embedded:
            tip = model.level_sets[0].bspline(0)

            # Catch T_local here as well!
            K_I, K_II, T_local = self.dcm.cal_sif(
                model.level_sets[0], model, model.cut_info, 0.0
            )

            # Use the true tangent for the second tip
            # (Standard DCM definitions ensure T_local[0] points OUTWARD into uncracked material)
            t_tip_true = T_local[0]
            t_unit = t_tip_true / np.linalg.norm(t_tip_true)

            self.K_I_tip_2.append(K_I)
            self.K_II_tip_2.append(K_II)

            theta = -self.growth_direction_fn(K_I, K_II)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            new_direction = rotation_matrix @ t_unit

            new_tip = tip + self.da * new_direction
            self.control_points.insert(0, new_tip)

        self.control_points_history.append(list(self.control_points))

    # def _crack_growth(self, model, bspline):
    #     tip = bspline(1)
    #     t_tip = bspline(1, nu=1)
    #     t_unit = t_tip / np.linalg.norm(t_tip)
    #     K_I, K_II, T_local = self.dcm.cal_sif(
    #         model.level_sets[0], model, model.cut_info, 1.0
    #     )
    #     self.K_I_tip_1.append(K_I)
    #     self.K_II_tip_1.append(K_II)
    #     theta = self.growth_direction_fn(K_I, K_II)
    #     print("K_I", K_I, "K_II", K_II)
    #     print("angle", np.degrees(theta))
    #     cos_t, sin_t = np.cos(theta), np.sin(theta)
    #     rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    #     new_direction = rotation_matrix @ t_unit
    #     new_tip = tip + self.da * new_direction
    #     self.control_points.append(new_tip)
    #     if self.embedded:
    #         tip = model.level_sets[0].bspline(0)
    #         t_tip = -model.level_sets[0].dbspline(0)
    #         t_unit = t_tip / np.linalg.norm(t_tip)
    #         K_I, K_II = self.dcm.cal_sif(
    #             model.level_sets[0], model, model.cut_info, 0.0
    #         )
    #         self.K_I_tip_2.append(K_I)
    #         self.K_II_tip_2.append(K_II)
    #         theta = -self.growth_direction_fn(K_I, K_II)
    #         cos_t, sin_t = np.cos(theta), np.sin(theta)
    #         rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    #         new_direction = rotation_matrix @ t_unit
    #         new_tip = tip + self.da * new_direction
    #         self.control_points.insert(0, new_tip)
    #
    #     self.control_points_history.append(list(self.control_points))

    def save_results(self, output_prefix="crack_growth_results"):
        cp_filename = f"{output_prefix}_splines.csv"
        with open(cp_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "point_index", "x", "y"])
            for iter_idx, cp_list in enumerate(self.control_points_history):
                for pt_idx, pt in enumerate(cp_list):
                    writer.writerow([iter_idx, pt_idx, pt[0], pt[1]])
        print(f"Saved control points to {cp_filename}")

        sif_filename = f"{output_prefix}_sifs.csv"
        with open(sif_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            if self.embedded:
                writer.writerow(
                    ["iteration", "K_I_tip_1", "K_II_tip_1", "K_I_tip_2", "K_II_tip_2"]
                )
                for i in range(len(self.K_I_tip_1)):
                    k1_2 = self.K_I_tip_2[i] if i < len(self.K_I_tip_2) else ""
                    k2_2 = self.K_II_tip_2[i] if i < len(self.K_II_tip_2) else ""
                    writer.writerow(
                        [i, self.K_I_tip_1[i], self.K_II_tip_1[i], k1_2, k2_2]
                    )
            else:
                writer.writerow(["iteration", "K_I_tip_1", "K_II_tip_1"])
                for i in range(len(self.K_I_tip_1)):
                    writer.writerow([i, self.K_I_tip_1[i], self.K_II_tip_1[i]])
        print(f"Saved SIF history to {sif_filename}")

    def run(self, max_iter=100):
        n = len(self.control_points)
        k = 2
        knots = knotvector.generate(k, n)
        bspline = BSpline(knots, np.array(self.control_points), k)

        for i in range(max_iter):
            model = XFEModel(
                self.nodes.copy(),
                self.elements,
                self.materials,
                self.reals,
                self.tip_enrichment,
                self.geometrical_range,
                self.corrected,
            )

            model.insert_crack_spline(
                bspline, embedded=self.embedded, h=self.h, snapping_tolerance=0.15
            )

            model.gen_list_dof(dof_per_node=self.dof_per_node)
            model.cal_global_matrices(self.elem_dict)

            self.bc_fn(model)
            self.force_fn(model)

            model.solve_static()

            if self.plot_fn is not None:
                self.plot_fn(model, bspline, i)

            try:
                self._crack_growth(model, bspline)
                n = len(self.control_points)
                k = 2
                knots = knotvector.generate(k, n)
                bspline = BSpline(knots, np.array(self.control_points), k)
            except ValueError as e:
                print(f"\nCrack growth failed at iteration {i}: {e}")
                self.save_results("failed_growth")
                break
        else:
            print(f"\nCrack growth completed {max_iter} iterations.")
            self.save_results("successful_growth")
