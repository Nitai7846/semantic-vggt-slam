#!/usr/bin/env python3
"""
Step 3 — A* Path Planning on the Occupancy Grid
================================================
Runs A* search on the 2D occupancy grid from a start to a goal position.

Usage
-----
    python step_3_a_star.py \\
        --grid       data/navigation/occupancy_grid.npz \\
        --start      0.0 0.0 \\
        --goal       0.5 2.5 \\
        --output_dir data/navigation/

Author : Nitai Shah
Course : ECEN 689 — Texas A&M University, Spring 2026
"""

import os
import heapq
import argparse
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Helpers ───────────────────────────────────────────────────────────────────

def world_to_cell(x, z, x_min, z_min, res, n_rows, n_cols):
    col = int(np.clip(np.floor((x - x_min) / res), 0, n_cols - 1))
    row = int(np.clip(np.floor((z - z_min) / res), 0, n_rows - 1))
    return (row, col)


def cell_to_world(row, col, x_min, z_min, res):
    return (x_min + (col + 0.5) * res, z_min + (row + 0.5) * res)


def snap_to_free(row, col, grid):
    if grid[row, col] == 1:
        return (row, col)
    visited, queue = set(), deque([(row, col, 0)])
    while queue:
        r, c, d = queue.popleft()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        if grid[r, c] == 1:
            return (r, c)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                queue.append((nr, nc, d + 1))
    return (row, col)


def astar(grid, start, goal):
    heap = [(0.0, start)]
    came, g = {}, {start: 0.0}
    diag = np.sqrt(2)
    while heap:
        _, cur = heapq.heappop(heap)
        if cur == goal:
            path = []
            while cur in came:
                path.append(cur)
                cur = came[cur]
            path.append(start)
            return path[::-1]
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = cur[0]+dr, cur[1]+dc
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                continue
            if grid[nr, nc] == 0:
                continue
            cost = diag if (dr and dc) else 1.0
            tg   = g[cur] + cost
            nb   = (nr, nc)
            if tg < g.get(nb, float("inf")):
                came[nb] = cur
                g[nb]    = tg
                h = np.sqrt((nr-goal[0])**2 + (nc-goal[1])**2)
                heapq.heappush(heap, (tg + h, nb))
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="A* path planning")
    p.add_argument("--grid",       required=True, help="occupancy_grid.npz")
    p.add_argument("--start",      nargs=2, type=float, default=[0.0, 0.0],
                   metavar=("X", "Z"), help="Start position in world XZ")
    p.add_argument("--goal",       nargs=2, type=float, required=True,
                   metavar=("X", "Z"), help="Goal position in world XZ")
    p.add_argument("--output_dir", default="output")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    occ    = np.load(args.grid)
    grid   = occ["grid"]
    x_min, z_min = float(occ["x_min"]), float(occ["z_min"])
    res    = float(occ["resolution"])
    n_rows, n_cols = int(occ["n_rows"]), int(occ["n_cols"])

    sc = snap_to_free(*world_to_cell(*args.start, x_min, z_min, res, n_rows, n_cols), grid)
    gc = snap_to_free(*world_to_cell(*args.goal,  x_min, z_min, res, n_rows, n_cols), grid)

    print(f"[A*] Start {tuple(args.start)} → cell {sc}")
    print(f"[A*] Goal  {tuple(args.goal)}  → cell {gc}")
    path = astar(grid, sc, gc)

    if path is None:
        print("[A*] No path found.")
        return

    length_m = len(path) * res
    print(f"[A*] Path: {len(path)} steps, {length_m:.2f} m")

    path_xz = np.array([cell_to_world(r, c, x_min, z_min, res) for r, c in path])
    np.savez(os.path.join(args.output_dir, "astar_path.npz"),
             path_xz=path_xz, start_xz=np.array(args.start), goal_xz=np.array(args.goal))

    # Visualise
    img = np.zeros((n_rows, n_cols, 3), dtype=np.float32)
    img[grid == 1] = 1.0
    img[grid == 0] = 0.1
    for r, c in path:
        img[r, c] = [0.2, 0.5, 1.0]
    img[sc[0], sc[1]] = [0.0, 0.9, 0.2]
    img[gc[0], gc[1]] = [0.9, 0.1, 0.1]

    x_max = x_min + n_cols * res
    z_max = z_min + n_rows * res
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img, origin="lower", extent=[x_min, x_max, z_min, z_max], interpolation="nearest")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.set_title(f"A* Path  ({len(path)} steps, {length_m:.2f} m)")
    ax.legend(handles=[
        mpatches.Patch(color="white",         label="Free"),
        mpatches.Patch(color=(0.1,0.1,0.1),   label="Occupied"),
        mpatches.Patch(color=(0.2,0.5,1.0),   label="Path"),
        mpatches.Patch(color=(0.0,0.9,0.2),   label="Start"),
        mpatches.Patch(color=(0.9,0.1,0.1),   label="Goal"),
    ], loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "astar_path.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {args.output_dir}/astar_path.{{npz,png}}")


if __name__ == "__main__":
    main()
