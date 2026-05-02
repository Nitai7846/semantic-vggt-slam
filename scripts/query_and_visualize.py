#!/usr/bin/env python3
"""
Open-Vocabulary 3D Query & Visualisation
=========================================
Given a fused semantic point cloud and a natural-language query, this
script locates the queried object(s) in 3D using the pipeline:

    text → CLIP text encoder → dot-product similarity in 3D →
    top-K% thresholding → DBSCAN spatial clustering →
    2D-heatmap-guided cluster selection → multi-instance detection →
    Open3D visualisation

Key technical detail
--------------------
CLIP features extracted at VGGT resolution produce *inverted* cosine
similarities (positional-embedding interpolation shifts the feature
distribution).  Fix: ``similarity = -(features @ text_feat)``, then
standard top-K% thresholding works correctly.

Usage
-----
    python query_and_visualize.py \\
        --fused   data/fused_semantic_pointcloud.npz \\
        --pt_dir  data/clip_features/ \\
        --kf_dir  data/keyframes/ \\
        --npz_dir data/pointclouds/ \\
        --query   "monitor"

Author : Nitai Shah
Course : ECEN 689 — Texas A&M University, Spring 2026
"""

import os
import glob
import argparse

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


# ── Constants ─────────────────────────────────────────────────────────────────
N_PATCHES_H, N_PATCHES_W = 21, 37
VOXEL_SIZE = 0.02


def sort_key(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:    return float(stem)
    except: return float("inf")


def largest_connected_region(sim_map, top_frac=0.15):
    thresh = np.percentile(sim_map, (1 - top_frac) * 100)
    lbl, n = label(sim_map >= thresh)
    return max((int(np.sum(lbl == i)) for i in range(1, n + 1)), default=0)


def positions_to_voxel_keys(pos, vs):
    idx = np.floor(pos / vs).astype(np.int32)
    return set(map(tuple, idx))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Open-vocabulary 3D semantic query")
    p.add_argument("--fused",     required=True, help="Fused .npz point cloud")
    p.add_argument("--pt_dir",    required=True, help="Per-frame CLIP .pt dir")
    p.add_argument("--kf_dir",    required=True, help="Keyframe images dir")
    p.add_argument("--npz_dir",   required=True, help="VGGT-SLAM per-frame NPZ dir")
    p.add_argument("--query",     required=True, help='Text query, e.g. "monitor"')
    p.add_argument("--output_dir", default="output", help="Debug output directory")

    # Tuning knobs
    p.add_argument("--top_percent",     type=float, default=1)
    p.add_argument("--dbscan_eps",      type=float, default=0.13)
    p.add_argument("--dbscan_min",      type=int,   default=5)
    p.add_argument("--merge_dist",      type=float, default=0.4)
    p.add_argument("--top_heatmap_frames", type=int, default=20)
    p.add_argument("--no_visualize",    action="store_true",
                   help="Skip Open3D viewer (headless mode)")
    args = p.parse_args()

    query      = args.query
    query_slug = query.replace(" ", "_").lower()
    debug_dir  = os.path.join(args.output_dir, query_slug)
    hm_dir     = os.path.join(debug_dir, "heatmaps")
    os.makedirs(hm_dir, exist_ok=True)

    # ── Load fused point cloud ────────────────────────────────────────────
    print("[LOAD] Fused point cloud …")
    data      = np.load(args.fused)
    positions = data["positions"]
    features  = data["features"]
    colors    = data["colors"]
    print(f"  {len(positions):,} voxels, features {features.shape}")

    # ── CLIP text encoding ────────────────────────────────────────────────
    print(f'[CLIP] Encoding query: "{query}"')
    model, _ = clip.load("ViT-L/14", device="cpu")
    model.eval()
    with torch.no_grad():
        text_feat = F.normalize(
            model.encode_text(clip.tokenize([query])), dim=-1
        )[0].numpy()

    # ── Score all 2D frames ───────────────────────────────────────────────
    pt_files = sorted(glob.glob(os.path.join(args.pt_dir, "*_clip.pt")), key=sort_key)
    print(f"\n[2D] Scoring {len(pt_files)} keyframes …")

    frame_results = []
    for pt_path in pt_files:
        fid   = os.path.basename(pt_path).replace("_clip.pt", "")
        feats = torch.load(pt_path, map_location="cpu", weights_only=False).float().numpy()
        flat  = feats.reshape(-1, 768)
        flat  = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-8)
        sim   = -(flat @ text_feat)                         # inverted similarity
        sm    = sim.reshape(N_PATCHES_H, N_PATCHES_W)
        frame_results.append(dict(
            frame_id=fid, frame_num=float(fid),
            max_sim=float(sm.max()), mean_sim=float(sm.mean()),
            connected_patches=largest_connected_region(sm),
            sim_map=sm, is_guide=False, contributed=False,
        ))
    frame_results.sort(key=lambda r: r["max_sim"], reverse=True)

    # ── Temporal DBSCAN on top frames → guide set ─────────────────────────
    top = frame_results[:args.top_heatmap_frames]
    nums = np.array([r["frame_num"] for r in top]).reshape(-1, 1)
    tdb  = DBSCAN(eps=30, min_samples=2).fit(nums)
    guide_ids = set(
        r["frame_id"] for r, l in zip(top, tdb.labels_) if l >= 0
    )
    for r in frame_results:
        r["is_guide"] = r["frame_id"] in guide_ids
    print(f"  Guide frames: {sorted(int(float(f)) for f in guide_ids)}")

    # ── Build guide voxel keys ────────────────────────────────────────────
    npz_files  = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")), key=sort_key)
    guide_keys = set()
    for nf in npz_files:
        fid = os.path.splitext(os.path.basename(nf))[0]
        if fid not in guide_ids:
            continue
        d = np.load(nf)
        pts = d["pointcloud"].reshape(-1, 3)[d["mask"].reshape(-1)]
        if len(pts):
            guide_keys |= positions_to_voxel_keys(pts, VOXEL_SIZE)
    print(f"  Guide voxels: {len(guide_keys):,}")

    # ── 3D query ──────────────────────────────────────────────────────────
    similarity = -(features @ text_feat)
    thresh     = np.percentile(similarity, 100 - args.top_percent)
    mask_3d    = similarity >= thresh
    print(f"\n[3D] '{query}' → {mask_3d.sum():,} / {len(positions):,} highlighted")

    # ── DBSCAN clustering ─────────────────────────────────────────────────
    hi_pos  = positions[mask_3d]
    g_idx   = np.where(mask_3d)[0]
    db      = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min).fit(hi_pos)
    labels  = db.labels_
    n_clust = len(set(labels) - {-1})
    print(f"[CLUSTER] {n_clust} cluster(s), {(labels==-1).sum()} outliers")

    # ── Guide-overlap cluster selection ───────────────────────────────────
    best_lbl, best_ov = -1, -1
    for lbl in range(n_clust):
        m   = labels == lbl
        ov  = len(positions_to_voxel_keys(hi_pos[m], VOXEL_SIZE) & guide_keys)
        sz  = int(m.sum())
        tag = ""
        if ov > best_ov:
            best_ov, best_lbl = ov, lbl
            tag = " ← best"
        print(f"  lbl={lbl:>3}  size={sz:>5}  guide_overlap={ov:>5}{tag}")

    # ── Merge nearby fragments ────────────────────────────────────────────
    if best_lbl >= 0:
        wc     = hi_pos[labels == best_lbl].mean(axis=0)
        merged = set()
        for lbl in range(n_clust):
            c = hi_pos[labels == lbl].mean(axis=0)
            if np.linalg.norm(c - wc) <= args.merge_dist:
                merged.add(lbl)
        cmask  = np.isin(labels, list(merged))
        c_gidx = g_idx[cmask]
        print(f"[MERGE] {len(merged)} fragment(s) → {cmask.sum()} pts")
    else:
        c_gidx = np.empty(0, dtype=int)
        wc     = np.zeros(3)

    # ── Save 2D heatmaps ─────────────────────────────────────────────────
    all_max = [r["max_sim"] for r in frame_results]
    ft      = np.percentile(all_max, 80)
    hm_results = [r for r in frame_results
                  if r["max_sim"] >= ft and r["connected_patches"] >= 4]
    n_save  = min(20, len(hm_results))

    if hm_results:
        gmin = min(r["sim_map"].min() for r in hm_results[:n_save])
        gmax = max(r["sim_map"].max() for r in hm_results[:n_save])
        print(f"\n[HEATMAP] Saving {n_save} heatmaps → {hm_dir}")
        for rank, r in enumerate(hm_results[:n_save]):
            kf = os.path.join(args.kf_dir, f"{r['frame_id']}.jpg")
            if not os.path.exists(kf):
                kf = os.path.join(args.kf_dir, f"{r['frame_id']}.png")
            if not os.path.exists(kf):
                continue
            img  = Image.open(kf).convert("RGB")
            sn   = (r["sim_map"] - gmin) / (gmax - gmin + 1e-8)
            up   = np.array(Image.fromarray((sn*255).astype(np.uint8)
                           ).resize(img.size, Image.BILINEAR)) / 255.0
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'Rank {rank+1} | Frame {r["frame_id"]} | "{query}"', fontsize=11)
            axes[0].imshow(img);           axes[0].set_title("Keyframe"); axes[0].axis("off")
            im = axes[1].imshow(r["sim_map"], cmap="hot", vmin=gmin, vmax=gmax)
            axes[1].set_title("Similarity"); axes[1].axis("off"); plt.colorbar(im, ax=axes[1])
            axes[2].imshow(img); axes[2].imshow(up, cmap="hot", alpha=0.6)
            axes[2].set_title("Overlay"); axes[2].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(hm_dir, f"rank{rank+1:02d}_{r['frame_id']}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()

    # ── 3D visualisation ─────────────────────────────────────────────────
    if args.no_visualize:
        print("[DONE] Headless mode — skipping Open3D viewer.")
        return

    print("\n[3D] Building visualisation …")
    vc = colors.astype(np.float64) / 255.0
    if len(c_gidx):
        vc[c_gidx] = [0.0, 0.85, 0.2]     # highlight in green

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(vc)

    geom = [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)]
    if len(c_gidx):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        s.translate(wc); s.paint_uniform_color([0.9, 0.1, 0.1]); s.compute_vertex_normals()
        geom.append(s)

    o3d.visualization.draw_geometries(
        geom, window_name=f'"{query}" — Semantic Query', width=1400, height=800
    )


if __name__ == "__main__":
    main()
