#!/usr/bin/env python3
"""
Step 2 — 2D Occupancy Grid Construction
========================================
Projects floor-inlier voxels onto the XZ plane to build a binary
occupancy grid (1 = traversable, 0 = occupied/unknown).

Usage
-----
    python step_2_create_grid.py \\
        --fused       data/fused_semantic_pointcloud.npz \\
        --floor_plane data/navigation/floor_plane.npz \\
        --output_dir  data/navigation/

Author : Nitai Shah
Course : ECEN 689 — Texas A&M University, Spring 2026
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Build 2D occupancy grid")
    parser.add_argument("--fused",       required=True)
    parser.add_argument("--floor_plane", required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--resolution",  type=float, default=0.05,
                        help="Grid cell size in metres (default 0.05)")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    positions = np.load(args.fused)["positions"]
    sd        = np.load(args.floor_plane)["signed_distances"]
    floor_idx = np.where(np.abs(sd) <= 0.05)[0]
    print(f"[LOAD] {len(positions):,} voxels, {len(floor_idx):,} floor inliers")

    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
    n_cols = int(np.ceil((x_max - x_min) / args.resolution)) + 1
    n_rows = int(np.ceil((z_max - z_min) / args.resolution)) + 1

    grid = np.zeros((n_rows, n_cols), dtype=np.uint8)
    fc   = np.clip(np.floor((positions[floor_idx, 0] - x_min) / args.resolution).astype(int), 0, n_cols-1)
    fr   = np.clip(np.floor((positions[floor_idx, 2] - z_min) / args.resolution).astype(int), 0, n_rows-1)
    grid[fr, fc] = 1

    n_free = int(grid.sum())
    print(f"[GRID] {n_rows}×{n_cols}  free={n_free:,} ({100*n_free/(n_rows*n_cols):.1f}%)")

    np.savez(os.path.join(args.output_dir, "occupancy_grid.npz"),
             grid=grid, x_min=np.float32(x_min), z_min=np.float32(z_min),
             resolution=np.float32(args.resolution),
             n_rows=np.int32(n_rows), n_cols=np.int32(n_cols))

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(np.where(grid, 255, 0).astype(np.uint8), cmap="gray", origin="lower",
              extent=[x_min, x_min + n_cols*args.resolution, z_min, z_min + n_rows*args.resolution])
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.set_title(f"Occupancy Grid  (res={args.resolution}m, free={n_free:,})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "occupancy_grid.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {args.output_dir}/occupancy_grid.{{npz,png}}")


if __name__ == "__main__":
    main()
