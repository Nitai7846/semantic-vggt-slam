#!/usr/bin/env python3
"""
Incremental Semantic Lifting & Voxel Fusion
============================================
For each keyframe, lifts dense CLIP features from 2D into the VGGT-SLAM
3D point cloud, then fuses all frames into a single voxel-averaged
semantic point cloud.

Pipeline per frame
------------------
1. Load VGGT-SLAM point cloud NPZ  →  (294, 518, 3) positions + mask + colours
2. Load dense CLIP features         →  (21, 37, 768)
3. Nearest-neighbour upsample       →  (294, 518, 768)
4. Index with mask → per-valid-pixel feature vector
5. Insert into an incremental voxel grid (running sums)

After all frames:
    Average position / feature / colour per voxel, L2-normalise features,
    and save as a single compressed `.npz` file.

Memory
------
Naive accumulation of all raw points (~20 M for the Office sequence)
kills the kernel. This script keeps memory proportional to the number
of *unique* voxels (~35 k at 2 cm resolution, ~100 MB).

Usage
-----
    python lift_and_fuse.py \\
        --npz_dir  data/pointclouds/ \\
        --pt_dir   data/clip_features/ \\
        --output   data/fused_semantic_pointcloud.npz

Author : Nitai Shah
Course : ECEN 689 — Texas A&M University, Spring 2026
"""

import os
import glob
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F


FEAT_DIM = 768


# ── Incremental voxel grid ───────────────────────────────────────────────────

class IncrementalVoxelGrid:
    """Running-sum voxel grid.  One frame at a time; discard raw points."""

    def __init__(self, voxel_size: float = 0.02):
        self.voxel_size   = voxel_size
        self.position_sum = {}
        self.feature_sum  = {}
        self.color_sum    = {}
        self.counts       = {}

    # ── per-frame insertion ──────────────────────────────────────────────
    def add_frame(self, positions, features, colors):
        """
        Parameters
        ----------
        positions : (N, 3) float32
        features  : (N, 768) float32
        colors    : (N, 3) uint8
        """
        keys = np.floor(positions / self.voxel_size).astype(np.int32)
        for i in range(len(positions)):
            k = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
            if k in self.counts:
                self.position_sum[k] += positions[i]
                self.feature_sum[k]  += features[i]
                self.color_sum[k]    += colors[i].astype(np.float64)
                self.counts[k]       += 1
            else:
                self.position_sum[k] = positions[i].copy()
                self.feature_sum[k]  = features[i].copy()
                self.color_sum[k]    = colors[i].astype(np.float64).copy()
                self.counts[k]       = 1

    # ── finalise ─────────────────────────────────────────────────────────
    def finalize(self):
        n = len(self.counts)
        positions = np.zeros((n, 3),        dtype=np.float32)
        features  = np.zeros((n, FEAT_DIM), dtype=np.float32)
        colors    = np.zeros((n, 3),        dtype=np.float64)
        counts    = np.zeros(n,             dtype=np.int32)

        for idx, k in enumerate(self.counts):
            c = self.counts[k]
            positions[idx] = self.position_sum[k] / c
            features[idx]  = self.feature_sum[k]  / c
            colors[idx]    = self.color_sum[k]     / c
            counts[idx]    = c

        norms    = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.maximum(norms, 1e-8)

        return positions, features, colors.astype(np.uint8), counts

    def num_voxels(self):
        return len(self.counts)


# ── Lift a single frame ──────────────────────────────────────────────────────

def lift_single_frame(npz_path, pt_path):
    """
    Returns
    -------
    positions : (M, 3)   float32  — valid 3D points in world frame
    features  : (M, 768) float32  — L2-normalised CLIP features
    colors    : (M, 3)   uint8
    """
    data       = np.load(npz_path)
    pointcloud = data["pointcloud"]   # (294, 518, 3)
    mask       = data["mask"]         # (294, 518)
    colors     = data["colors"]       # (294, 518, 3)

    feats = torch.load(pt_path, map_location="cpu").float()          # (21, 37, 768)
    feats = feats.permute(2, 0, 1).unsqueeze(0)
    feats = F.interpolate(feats, scale_factor=14, mode="nearest")    # (1, 768, 294, 518)
    feats = feats[0].permute(1, 2, 0)                                # (294, 518, 768)

    flat_mask = mask.reshape(-1)
    pos  = pointcloud.reshape(-1, 3)[flat_mask]
    feat = feats.reshape(-1, 768).numpy()[flat_mask]
    col  = colors.reshape(-1, 3)[flat_mask]

    norms = np.linalg.norm(feat, axis=1, keepdims=True)
    feat  = feat / np.maximum(norms, 1e-8)

    return pos.astype(np.float32), feat.astype(np.float32), col


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Semantic lifting & voxel fusion")
    parser.add_argument("--npz_dir",    required=True, help="VGGT-SLAM per-frame NPZ dir")
    parser.add_argument("--pt_dir",     required=True, help="Dense CLIP .pt feature dir")
    parser.add_argument("--output",     required=True, help="Output .npz path")
    parser.add_argument("--voxel_size", type=float, default=0.02,
                        help="Voxel edge length in metres (default: 0.02)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Match NPZ ↔ PT pairs
    npz_files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    pairs = []
    for npz_path in npz_files:
        frame_id = os.path.splitext(os.path.basename(npz_path))[0]
        pt_path  = os.path.join(args.pt_dir, f"{frame_id}_clip.pt")
        if os.path.exists(pt_path):
            pairs.append((npz_path, pt_path, frame_id))
        else:
            print(f"  [WARN] No .pt for frame {frame_id} — skipping")

    print(f"[CONFIG] Matched pairs : {len(pairs)}")
    print(f"[CONFIG] Voxel size    : {args.voxel_size} m\n")

    grid = IncrementalVoxelGrid(voxel_size=args.voxel_size)
    total_pts, t0 = 0, time.time()

    for idx, (npz_path, pt_path, fid) in enumerate(pairs):
        pos, feat, col = lift_single_frame(npz_path, pt_path)
        grid.add_frame(pos, feat, col)
        total_pts += len(pos)

        if (idx + 1) % 20 == 0 or idx == 0 or idx == len(pairs) - 1:
            print(f"  [{idx+1:4d}/{len(pairs)}] frame {fid:>8s} | "
                  f"{len(pos):6,} pts | voxels: {grid.num_voxels():,} | "
                  f"{time.time()-t0:.1f}s")

    dt = time.time() - t0
    print(f"\n[DONE] {len(pairs)} frames — {total_pts:,} raw pts → "
          f"{grid.num_voxels():,} voxels ({total_pts/grid.num_voxels():.0f}× compression)")

    positions, features, colors, counts = grid.finalize()
    print(f"[STATS] Observations per voxel — min: {counts.min()}, max: {counts.max()}, "
          f"median: {np.median(counts):.0f}")

    np.savez_compressed(
        args.output,
        positions=positions, features=features,
        colors=colors, counts=counts,
        voxel_size=args.voxel_size,
    )
    mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"[SAVE] {args.output}  ({len(positions):,} voxels, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
