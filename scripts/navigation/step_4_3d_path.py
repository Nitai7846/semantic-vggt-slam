#!/usr/bin/env python3
"""
Step 4 — 3D Path Visualisation
===============================
Lifts the 2D A* path onto the reconstructed 3D floor surface and
renders the full scene with highlighted semantic clusters and
navigation paths in Open3D.

Usage
-----
    python step_4_3d_path.py \\
        --fused       data/fused_semantic_pointcloud.npz \\
        --floor_plane data/navigation/floor_plane.npz \\
        --astar_path  data/navigation/astar_path.npz

Author : Nitai Shah
Course : ECEN 689 — Texas A&M University, Spring 2026
"""

import argparse
import numpy as np
import open3d as o3d


def main():
    p = argparse.ArgumentParser(description="3D path visualisation")
    p.add_argument("--fused",       required=True)
    p.add_argument("--floor_plane", required=True)
    p.add_argument("--astar_path",  required=True)
    p.add_argument("--sphere_radius", type=float, default=0.03)
    args = p.parse_args()

    data      = np.load(args.fused)
    positions = data["positions"]
    colors    = data["colors"]

    floor  = np.load(args.floor_plane)
    sd     = floor["signed_distances"]
    fmask  = np.abs(sd) <= 0.05
    floor_pos = positions[fmask]
    floor_xz  = floor_pos[:, [0, 2]]

    astar  = np.load(args.astar_path)
    path_xz = astar["path_xz"]

    # Lift 2D → 3D via nearest floor inlier
    path_3d = np.zeros((len(path_xz), 3))
    for i, (px, pz) in enumerate(path_xz):
        d = np.sqrt((floor_xz[:, 0]-px)**2 + (floor_xz[:, 1]-pz)**2)
        path_3d[i] = [px, floor_pos[np.argmin(d), 1] + 0.02, pz]

    # Point cloud
    vc = colors.astype(np.float64) / 255.0
    vc[~fmask] = vc[~fmask] * 0.6 + 0.4 * np.array([0.85, 0.85, 0.85])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(vc)

    # Path line
    lines = [[i, i+1] for i in range(len(path_3d)-1)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(path_3d)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([[0.2, 0.5, 1.0]] * len(lines))

    # Spheres
    geom = [pcd, ls, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)]
    step = max(1, len(path_3d) // 20)
    for pt in path_3d[::step]:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=args.sphere_radius)
        s.translate(pt); s.paint_uniform_color([0.2, 0.5, 1.0]); s.compute_vertex_normals()
        geom.append(s)
    for pt, col in [(path_3d[0], [0.0,0.9,0.2]), (path_3d[-1], [0.9,0.1,0.1])]:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.06)
        s.translate(pt); s.paint_uniform_color(col); s.compute_vertex_normals()
        geom.append(s)

    print(f"[VIS] Path: {len(path_3d)} waypoints")
    o3d.visualization.draw_geometries(geom, window_name="3D Path", width=1400, height=800)


if __name__ == "__main__":
    main()
