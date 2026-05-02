#!/usr/bin/env python3
"""
Dense CLIP Feature Extraction at VGGT Resolution
=================================================
Extracts dense (21 × 37 × 768) CLIP ViT-L/14 features for every keyframe,
matched to VGGT-SLAM's internal resolution (518 × 294 for 1920×1080 input).

Key insight: CLIP's default 224×224 preprocessing center-crops the input,
discarding spatial alignment with VGGT's point map. Instead, we resize to
VGGT's resolution (518×294) and bilinearly interpolate CLIP's positional
embeddings from (16×16) → (37×21). Each output patch covers exactly a
14×14 pixel block, producing a 1:1 alignment with VGGT's point map.

Usage
-----
    python extract_dense_clip.py \\
        --input_dir  data/keyframes/ \\
        --output_dir data/clip_features/

Output
------
    One `<frame_name>_clip.pt` per image, each containing a
    (21, 37, 768) float32 tensor of L2-normalised dense CLIP features.

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
import clip
from PIL import Image
from torchvision import transforms as TF


# ── Resolution constants (for 1920×1080 input via VGGT) ──────────────────────
TARGET_W    = 518
TARGET_H    = 294
PATCH_SIZE  = 14
N_PATCHES_H = TARGET_H // PATCH_SIZE   # 21
N_PATCHES_W = TARGET_W // PATCH_SIZE   # 37

# CLIP normalisation statistics
NORMALIZE = TF.Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
)


# ── Positional-embedding interpolation ────────────────────────────────────────

def interpolate_pos_embed(pos_embed, n_patches_h, n_patches_w, orig_size=16):
    """Bilinearly interpolate CLIP's fixed positional embeddings to a new grid."""
    cls_pos   = pos_embed[0:1, :]
    patch_pos = pos_embed[1:, :]
    dim       = patch_pos.shape[-1]

    patch_pos = patch_pos.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(
        patch_pos.float(),
        size=(n_patches_h, n_patches_w),
        mode="bilinear",
        align_corners=False,
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(-1, dim).to(cls_pos.dtype)
    return torch.cat([cls_pos, patch_pos], dim=0)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_image(image_path):
    """Load an image and resize to VGGT resolution. Returns (1, 3, 294, 518)."""
    img    = Image.open(image_path).convert("RGB")
    tensor = TF.ToTensor()(img.resize((TARGET_W, TARGET_H), Image.BICUBIC))
    return NORMALIZE(tensor).unsqueeze(0)


# ── Dense feature extraction ─────────────────────────────────────────────────

def extract_dense_features(model, image_tensor, pos_embed_interp, device):
    """
    Forward pass through CLIP ViT-L/14 at VGGT resolution.

    Returns
    -------
    dense_features : Tensor (21, 37, 768)
        L2-normalised patch-level CLIP features.
    """
    with torch.no_grad():
        v = model.visual

        x = v.conv1(image_tensor.float())                            # (1, 1024, 21, 37)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (1, 777, 1024)

        cls_token = v.class_embedding.to(x.dtype).unsqueeze(0).unsqueeze(0)
        x = torch.cat([cls_token.expand(x.shape[0], -1, -1), x], dim=1)

        x = v.ln_pre(x + pos_embed_interp.to(x.dtype))
        x = v.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)

        patch_tokens = v.ln_post(x[:, 1:, :]) @ v.proj              # (1, 777, 768)
        patch_tokens = F.normalize(patch_tokens, dim=-1)

    return patch_tokens[0].reshape(N_PATCHES_H, N_PATCHES_W, -1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dense CLIP feature extraction at VGGT resolution"
    )
    parser.add_argument("--input_dir",  required=True, help="Keyframe images (.jpg/.png)")
    parser.add_argument("--output_dir", required=True, help="Output directory for .pt files")
    parser.add_argument("--device",     default=None,  help="cuda / cpu (auto-detected)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = sorted(
        glob.glob(os.path.join(args.input_dir, "*.jpg"))
        + glob.glob(os.path.join(args.input_dir, "*.png"))
    )
    if not image_paths:
        print(f"[ERROR] No images found in {args.input_dir}")
        return

    print(f"[CONFIG] Device     : {device}")
    print(f"[CONFIG] Images     : {len(image_paths)}")
    print(f"[CONFIG] VGGT res   : {TARGET_W}×{TARGET_H}")
    print(f"[CONFIG] Patch grid : {N_PATCHES_W}×{N_PATCHES_H}  →  output (21, 37, 768)\n")

    # Load CLIP once
    print("[CLIP] Loading ViT-L/14 …")
    model, _ = clip.load("ViT-L/14", device=device)
    model.eval()

    pos_embed_interp = interpolate_pos_embed(
        model.visual.positional_embedding,
        N_PATCHES_H, N_PATCHES_W,
        orig_size=model.visual.input_resolution // PATCH_SIZE,
    ).to(device)
    print(f"[POS]  Interpolated to ({N_PATCHES_H}, {N_PATCHES_W})  shape {pos_embed_interp.shape}\n")

    t0, skipped = time.time(), 0
    for idx, path in enumerate(image_paths):
        stem = os.path.splitext(os.path.basename(path))[0]
        out  = os.path.join(args.output_dir, f"{stem}_clip.pt")
        if os.path.exists(out):
            skipped += 1
            continue

        tensor = preprocess_image(path).to(device)
        feats  = extract_dense_features(model, tensor, pos_embed_interp, device)
        torch.save(feats.cpu(), out)

        if (idx + 1) % 10 == 0 or idx == 0:
            dt = time.time() - t0
            print(f"  [{idx+1:4d}/{len(image_paths)}]  {stem}  |  {dt:.1f}s")

    dt = time.time() - t0
    done = len(image_paths) - skipped
    print(f"\n[DONE] {done} images in {dt:.1f}s  ({dt/max(done,1):.2f}s/img)")
    if skipped:
        print(f"[DONE] Skipped {skipped} (already existed)")


if __name__ == "__main__":
    main()
