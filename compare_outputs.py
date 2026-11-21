#!/usr/bin/env python
"""
Side-by-side viewer: input T1 | each model's fake T2 | real T2 | |error| per model.

Picks slices from the fixed eval subset: by default the best, median and worst
for the first run (by masked SSIM from its per-slice CSV), plus a few random ones.

Usage:
  python compare_outputs.py --runs cut_v1=/scratch/$USER/mri_runs/results/cut_v1/test_latest/images
  python compare_outputs.py --runs cut_v1=<dir> cyclegan_v1=<dir> --files BraTS2021_00019_s061.png
"""
import argparse
import csv
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load01(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def find_pred(pred_dir, name):
    stem = os.path.splitext(name)[0]
    for p in (os.path.join(pred_dir, name),
              os.path.join(pred_dir, "fake_B", name),
              os.path.join(pred_dir, f"{stem}_fake_B.png")):
        if os.path.exists(p):
            return p
    return None


ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="name=pred_dir pairs")
ap.add_argument("--data", default=os.path.expandvars("/scratch/$USER/brats/brats_t1t2"))
ap.add_argument("--files", nargs="*", help="specific slice filenames to show")
ap.add_argument("--results_dir", default="results")
ap.add_argument("--out", default="compare.png")
args = ap.parse_args()

runs = [r.split("=", 1) for r in args.runs]

if args.files:
    files = args.files
else:
    per_slice = os.path.join(args.results_dir, f"{runs[0][0]}_eval_per_slice.csv")
    if os.path.exists(per_slice):
        rows = sorted(csv.DictReader(open(per_slice)), key=lambda r: float(r["ssim"]))
        picks = [rows[-1], rows[len(rows) // 2], rows[0]]          # best, median, worst
        files = [r["file"] for r in picks]
        labels = ["best", "median", "worst"]
    else:
        files, labels = [], []
    pool = json.load(open(os.path.join(args.data, "split.json")))["eval_subset"]["files"]
    extra = [f for f in random.Random(0).sample(pool, 6) if f not in files][: 6 - len(files)]
    files += extra

cols = 2 + 2 * len(runs)
fig, ax = plt.subplots(len(files), cols, figsize=(2.4 * cols, 2.5 * len(files)))
ax = np.atleast_2d(ax)
for r, name in enumerate(files):
    t1 = load01(os.path.join(args.data, "testA", name))
    t2 = load01(os.path.join(args.data, "testB", name))
    mask = load01(os.path.join(args.data, "testMask", name)) > 0.5
    panels = [("input T1", t1, "gray", 1)]
    for run, d in runs:
        p = find_pred(d, name)
        fake = load01(p) if p else np.zeros_like(t2)
        panels.append((f"{run} fake T2", fake, "gray", 1))
    panels.append(("real T2", t2, "gray", 1))
    for run, d in runs:
        p = find_pred(d, name)
        fake = load01(p) if p else np.zeros_like(t2)
        err = np.abs(fake - t2) * mask
        panels.append((f"{run} |error| (MAE {err[mask].mean():.3f})", err, "inferno", 0.5))
    for c, (title, img, cmap, vmax) in enumerate(panels):
        ax[r, c].imshow(img, cmap=cmap, vmin=0, vmax=vmax)
        ax[r, c].set_title(title, fontsize=7)
        ax[r, c].axis("off")
    ax[r, 0].set_title(f"{name}\ninput T1", fontsize=7)

plt.tight_layout()
plt.savefig(args.out, dpi=100)
print("saved", args.out, "showing:", ", ".join(files))
