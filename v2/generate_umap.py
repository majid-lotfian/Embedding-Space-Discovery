"""
generate_umap.py
================
Run this SEPARATELY after CH_ssl_embedder.py finishes.
No PyTorch — no CUDA/numba conflict.

Usage:
    python v2/generate_umap.py --emb_dir ./output
"""

import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap as umap_module

COLORS = ["#4477AA", "#EE6677", "#228833"]
LABELS = {0: "no-CH", 1: "CH-T", 2: "CH-C"}
SEED = 42

def plot_2d(Z, y, title, path):
    plt.figure(figsize=(6, 5))
    for lab, col in zip([0, 1, 2], COLORS):
        idx = y == lab
        if idx.any():
            plt.scatter(Z[idx, 0], Z[idx, 1], s=15, alpha=0.75,
                        label=LABELS[lab], color=col)
    plt.title(title, fontsize=11); plt.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", required=True)
    args = parser.parse_args()

    dim_dirs = sorted([
        d for d in os.listdir(args.emb_dir)
        if d.startswith("repr_dim_") and os.path.isdir(os.path.join(args.emb_dir, d))
    ])

    for dim_dir in dim_dirs:
        path = os.path.join(args.emb_dir, dim_dir)
        emb_file = os.path.join(path, "embeddings.npy")
        lbl_file = os.path.join(path, "labels_CH3.npy")

        if not os.path.exists(emb_file):
            print(f"Skipping {dim_dir} — embeddings.npy not found.", flush=True)
            continue

        E = np.load(emb_file)
        y = np.load(lbl_file)
        d = dim_dir.replace("repr_dim_", "")

        print(f"UMAP for dim={d} ({len(E)} samples)...", flush=True)
        Z = umap_module.UMAP(n_components=2, n_neighbors=min(15, len(E) - 1),
                             min_dist=0.1, random_state=SEED, n_jobs=1).fit_transform(E)
        out_path = os.path.join(path, "umap_ch3.png")
        plot_2d(Z, y, f"UMAP (dim={d})", out_path)
        print(f"  Saved: {out_path}", flush=True)

    print("Done.", flush=True)

if __name__ == "__main__":
    main()
