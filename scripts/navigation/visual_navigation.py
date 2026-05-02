#!/usr/bin/env python3
"""
Language-Guided Visual Navigation (All-in-One)
===============================================
End-to-end pipeline: takes a natural-language query (e.g., "find the
refrigerator") and produces a 3D navigation path from the robot's
start position to the queried object in the semantic map.

Combines: CLIP 3D query → DBSCAN clustering → guide-based selection →
multi-instance detection → floor-plane occupancy grid → A* path
planning → 3D visualisation with Open3D.

Usage
-----
    python visual_navigation.py \\
        --fused       data/fused_semantic_pointcloud.npz \\
        --pt_dir      data/clip_features/ \\
        --kf_dir      data/keyframes/ \\
        --npz_dir     data/pointclouds/ \\
        --floor_plane data/navigation/floor_plane.npz \\
        --grid        data/navigation/occupancy_grid.npz \\
        --query       "refrigerator"

Author : Nitai Shah
Course : ECEN 689 — Texas A&M University, Spring 2026
"""

import os
import sys
import glob
import heapq
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import clip
import open3d as o3d
from PIL import Image
from scipy.ndimage import label
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Constants ─────────────────────────────────────────────────────────────────
N_PATCHES_H, N_PATCHES_W = 21, 37
VOXEL_SIZE = 0.02

INSTANCE_COLORS = [
    [0.0, 0.85, 0.20],   # green  — guide-selected
    [0.0, 0.70, 1.00],   # cyan
    [1.0, 0.50, 0.00],   # orange
    [0.90, 0.20, 0.80],  # purple
    [1.0, 0.90, 0.00],   # yellow
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sort_key(path):
    try:    return float(os.path.splitext(os.path.basename(path))[0])
    except: return float("inf")

def _largest_cc(sim_map, top_frac=0.15):
    thresh = np.percentile(sim_map, (1 - top_frac) * 100)
    lbl, n = label(sim_map >= thresh)
    return max((int(np.sum(lbl == i)) for i in range(1, n + 1)), default=0)

def _voxel_keys(pos, vs):
    return set(map(tuple, np.floor(pos / vs).astype(np.int32)))

def _w2c(x, z, xm, zm, r, nr, nc):
    return (int(np.clip(np.floor((z-zm)/r), 0, nr-1)),
            int(np.clip(np.floor((x-xm)/r), 0, nc-1)))

def _c2w(row, col, xm, zm, r):
    return (xm + (col+0.5)*r, zm + (row+0.5)*r)

def _snap(row, col, grid):
    if grid[row, col] == 1:
        return (row, col)
    vis, q = set(), deque([(row, col)])
    while q:
        r, c = q.popleft()
        if (r, c) in vis: continue
        vis.add((r, c))
        if grid[r, c] == 1: return (r, c)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                q.append((nr, nc))
    return (row, col)

def _astar(grid, start, goal):
    heap, came, g = [(0.0, start)], {}, {start: 0.0}
    d = np.sqrt(2)
    while heap:
        _, cur = heapq.heappop(heap)
        if cur == goal:
            p = []
            while cur in came: p.append(cur); cur = came[cur]
            p.append(start); return p[::-1]
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = cur[0]+dr, cur[1]+dc
            if not (0<=nr<grid.shape[0] and 0<=nc<grid.shape[1]): continue
            if grid[nr, nc]==0: continue
            tg = g[cur] + (d if dr and dc else 1.0)
            nb = (nr, nc)
            if tg < g.get(nb, float("inf")):
                came[nb] = cur; g[nb] = tg
                heapq.heappush(heap, (tg + np.sqrt((nr-goal[0])**2+(nc-goal[1])**2), nb))
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Language-guided visual navigation")
    ap.add_argument("--fused",       required=True)
    ap.add_argument("--pt_dir",      required=True)
    ap.add_argument("--kf_dir",      required=True)
    ap.add_argument("--npz_dir",     required=True)
    ap.add_argument("--floor_plane", required=True)
    ap.add_argument("--grid",        required=True)
    ap.add_argument("--query",       required=True)
    ap.add_argument("--start",       nargs=2, type=float, default=[0.0, 0.0])
    ap.add_argument("--output_dir",  default="output")
    ap.add_argument("--top_percent", type=float, default=1)
    ap.add_argument("--dbscan_eps",  type=float, default=0.13)
    ap.add_argument("--dbscan_min",  type=int,   default=5)
    ap.add_argument("--merge_dist",  type=float, default=0.4)
    ap.add_argument("--no_visualize", action="store_true")
    args = ap.parse_args()

    query      = args.query
    slug       = query.replace(" ", "_").lower()
    debug_dir  = os.path.join(args.output_dir, slug)
    hm_dir     = os.path.join(debug_dir, "heatmaps")
    os.makedirs(hm_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print("[LOAD] Fused point cloud …")
    d = np.load(args.fused)
    positions, features, colors = d["positions"], d["features"], d["colors"]

    print("[LOAD] Floor & occupancy …")
    fd    = np.load(args.floor_plane)
    sd    = fd["signed_distances"]
    fmask = np.abs(sd) <= 0.05
    fp    = positions[fmask]
    fxz   = fp[:, [0, 2]]

    occ   = np.load(args.grid)
    grid  = occ["grid"]
    xm, zm = float(occ["x_min"]), float(occ["z_min"])
    res    = float(occ["resolution"])
    nr, nc = int(occ["n_rows"]), int(occ["n_cols"])

    # ── CLIP query ────────────────────────────────────────────────────────
    print(f'[CLIP] Encoding: "{query}"')
    model, _ = clip.load("ViT-L/14", device="cpu"); model.eval()
    with torch.no_grad():
        tf = F.normalize(model.encode_text(clip.tokenize([query])), dim=-1)[0].numpy()

    # ── 2D scoring → guide frames ─────────────────────────────────────────
    pt_files = sorted(glob.glob(os.path.join(args.pt_dir, "*_clip.pt")), key=_sort_key)
    fr = []
    for pf in pt_files:
        fid   = os.path.basename(pf).replace("_clip.pt", "")
        feats = torch.load(pf, map_location="cpu", weights_only=False).float().numpy().reshape(-1, 768)
        feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8)
        sm    = -(feats @ tf).reshape(N_PATCHES_H, N_PATCHES_W)
        fr.append(dict(frame_id=fid, frame_num=float(fid), max_sim=float(sm.max()),
                       sim_map=sm, connected_patches=_largest_cc(sm)))
    fr.sort(key=lambda r: r["max_sim"], reverse=True)

    top  = fr[:20]
    nums = np.array([r["frame_num"] for r in top]).reshape(-1, 1)
    tdb  = DBSCAN(eps=30, min_samples=2).fit(nums)
    gids = set(r["frame_id"] for r, l in zip(top, tdb.labels_) if l >= 0)

    npz_files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")), key=_sort_key)
    gkeys = set()
    for nf in npz_files:
        fid = os.path.splitext(os.path.basename(nf))[0]
        if fid not in gids: continue
        dd = np.load(nf)
        pts = dd["pointcloud"].reshape(-1, 3)[dd["mask"].reshape(-1)]
        if len(pts): gkeys |= _voxel_keys(pts, VOXEL_SIZE)

    # ── 3D query + clustering ─────────────────────────────────────────────
    sim   = -(features @ tf)
    thr   = np.percentile(sim, 100 - args.top_percent)
    m3d   = sim >= thr
    hi    = positions[m3d]
    gidx  = np.where(m3d)[0]
    db    = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min).fit(hi)
    labs  = db.labels_
    ncl   = len(set(labs) - {-1})
    print(f'[3D] "{query}" → {m3d.sum():,} pts, {ncl} cluster(s)')

    # ── Guide-overlap selection + merge ───────────────────────────────────
    best_lbl, best_ov = -1, -1
    for lbl in range(ncl):
        m  = labs == lbl
        ov = len(_voxel_keys(hi[m], VOXEL_SIZE) & gkeys)
        if ov > best_ov: best_ov, best_lbl = ov, lbl

    if best_lbl >= 0:
        wc = hi[labs == best_lbl].mean(axis=0)
        merged = {l for l in range(ncl)
                  if np.linalg.norm(hi[labs==l].mean(axis=0) - wc) <= args.merge_dist}
        cmask   = np.isin(labs, list(merged))
        c_gidx  = gidx[cmask]
        centroid = hi[cmask].mean(axis=0)
    else:
        c_gidx = np.empty(0, dtype=int)
        centroid = np.zeros(3)

    # ── A* navigation ─────────────────────────────────────────────────────
    goal_xz = (float(centroid[0]), float(centroid[2]))
    sc = _snap(*_w2c(*args.start, xm, zm, res, nr, nc), grid)
    gc = _snap(*_w2c(*goal_xz,    xm, zm, res, nr, nc), grid)
    path_cells = _astar(grid, sc, gc)

    if path_cells:
        pxz = np.array([_c2w(r, c, xm, zm, res) for r, c in path_cells])
        p3d = np.zeros((len(pxz), 3))
        for i, (px, pz) in enumerate(pxz):
            dd = np.sqrt((fxz[:,0]-px)**2 + (fxz[:,1]-pz)**2)
            p3d[i] = [px, fp[np.argmin(dd), 1] + 0.02, pz]
        print(f"[NAV] Path: {len(path_cells)} steps, {len(path_cells)*res:.2f} m")
    else:
        p3d = None
        print("[NAV] No path found")

    # ── Save 2D map ───────────────────────────────────────────────────────
    img = np.zeros((nr, nc, 3), dtype=np.float32)
    img[grid==1] = 1.0; img[grid==0] = 0.1
    if path_cells:
        for r, c in path_cells: img[r, c] = INSTANCE_COLORS[0]
    img[sc[0], sc[1]] = [0.0, 0.9, 0.2]
    img[gc[0], gc[1]] = [0.9, 0.1, 0.1]
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.imshow(img, origin="lower",
              extent=[xm, xm+nc*res, zm, zm+nr*res], interpolation="nearest")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.set_title(f'"{query}" — Navigation')
    ax.legend(handles=[
        mpatches.Patch(color="white",       label="Free"),
        mpatches.Patch(color=(0.1,0.1,0.1), label="Occupied"),
        mpatches.Patch(color=(0,0.9,0.2),   label="Start"),
        mpatches.Patch(color=(0.9,0.1,0.1), label="Goal"),
    ], loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir, f"{slug}_navigation_2d.png"), dpi=150, bbox_inches="tight")
    plt.close()

    if args.no_visualize:
        print("[DONE] Headless — skipping viewer.")
        return

    # ── 3D visualisation ─────────────────────────────────────────────────
    vc = colors.astype(np.float64) / 255.0
    vc[~fmask] = vc[~fmask]*0.6 + 0.4*np.array([0.85, 0.85, 0.85])
    if len(c_gidx): vc[c_gidx] = INSTANCE_COLORS[0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(vc)
    geom = [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)]

    if p3d is not None:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(p3d)
        ls.lines  = o3d.utility.Vector2iVector([[i,i+1] for i in range(len(p3d)-1)])
        ls.colors = o3d.utility.Vector3dVector([INSTANCE_COLORS[0]]*(len(p3d)-1))
        geom.append(ls)
        step = max(1, len(p3d)//15)
        for pt in p3d[::step]:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            s.translate(pt); s.paint_uniform_color(INSTANCE_COLORS[0]); s.compute_vertex_normals()
            geom.append(s)
        for pt, col in [(p3d[0],[0,0.9,0.2]), (p3d[-1],[0.9,0.1,0.1])]:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.06)
            s.translate(pt); s.paint_uniform_color(col); s.compute_vertex_normals()
            geom.append(s)

    o3d.visualization.draw_geometries(
        geom, window_name=f'"{query}" — Navigation', width=1400, height=800)


if __name__ == "__main__":
    main()
