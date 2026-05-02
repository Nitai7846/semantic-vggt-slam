#!/usr/bin/env python3
"""
Step 1 — RANSAC Floor Plane Detection
======================================
Fits a plane to the fused point cloud using RANSAC and saves the
floor-plane parameters and per-voxel signed distances for downstream
occupancy-grid construction and path planning.

Usage
-----
    python step_1_floor.py \\
        --fused  data/fused_semantic_pointcloud.npz \\
        --output data/navigation/floor_plane.npz

Author : Nitai Shah
Course : ECEN 689 — Texas A&M University, Spring 2026
"""

import os
import argparse
import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description="RANSAC floor plane fitting")
    parser.add_argument("--fused",  required=True, help="Fused semantic .npz")
    parser.add_argument("--output", required=True, help="Output floor_plane.npz")
    parser.add_argument("--distance_threshold", type=float, default=0.02)
    parser.add_argument("--iterations",         type=int,   default=1000)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print("[LOAD] Fused point cloud …")
    data      = np.load(args.fused)
    positions = data["positions"]
    colors    = data["colors"]
    print(f"  Voxels: {len(positions):,}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    print("\n[RANSAC] Fitting floor plane …")
    plane_model, inlier_indices = pcd.segment_plane(
        distance_threshold=args.distance_threshold,
        ransac_n=3,
        num_iterations=args.iterations,
    )
    a, b, c, d = plane_model
    normal     = np.array([a, b, c])
    norm_mag   = np.linalg.norm(normal)
    unit_norm  = normal / norm_mag

    print(f"  Plane: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(f"  Inliers: {len(inlier_indices):,} / {len(positions):,}")

    signed_distances = (positions @ normal + d) / norm_mag
    print(f"  Height range: [{signed_distances.min():.3f}, {signed_distances.max():.3f}] m")

    np.savez(args.output,
             plane_model=np.array(plane_model),
             unit_normal=unit_norm,
             signed_distances=signed_distances)
    print(f"\n[SAVE] {args.output}")

    if args.visualize:
        vc = np.full((len(positions), 3), 0.85)
        vc[signed_distances > 0.05] = colors[signed_distances > 0.05].astype(np.float64) / 255.0
        inlier_mask = np.zeros(len(positions), dtype=bool)
        inlier_mask[inlier_indices] = True
        vc[inlier_mask] = [0.0, 0.9, 0.2]
        vc[signed_distances < -0.05] = [0.9, 0.1, 0.1]
        vis = o3d.geometry.PointCloud()
        vis.points = o3d.utility.Vector3dVector(positions)
        vis.colors = o3d.utility.Vector3dVector(vc)
        o3d.visualization.draw_geometries([vis], window_name="Floor Plane", width=1400, height=800)


if __name__ == "__main__":
    main()
