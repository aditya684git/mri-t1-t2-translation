#!/usr/bin/env python
"""
Is a model's output spatially shifted relative to the real T2?

For each slice in the eval subset, slide the prediction by dx, dy in [-R, R] pixels
and compute brain-masked SSIM. If the best average SSIM is at (0, 0) the outputs are
aligned; if it peaks elsewhere, the model shifts images and SSIM is unfairly penalised.

Usage:
  python shift_check.py --runs cut_v1=<pred_dir> cyclegan_v1=<pred_dir>
  python shift_check.py --target T1 --runs syndiff_dir1=<pred_dir>   # T2 -> T1 outputs, against the real T1
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def load01(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float64) / 255.0


def find_pred(d, name):
    stem = os.path.splitext(name)[0]
    for p in (os.path.join(d, name), os.path.join(d, "fake_B", name), os.path.join(d, f"{stem}_fake_B.png")):
        if os.path.exists(p):
            return p


ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="name=pred_dir pairs")
ap.add_argument("--data", default=os.path.expandvars("/scratch/$USER/brats/brats_t1t2"))
ap.add_argument("--radius", type=int, default=2)
ap.add_argument("--n", type=int, default=50, help="eval slices to use (speed)")
ap.add_argument("--target", choices=["T2", "T1"], default="T2",
                help="real image to compare with: T2 (testB) for T1->T2 outputs, T1 (testA) for T2->T1 outputs")
args = ap.parse_args()

files = json.load(open(os.path.join(args.data, "split.json")))["eval_subset"]["files"][: args.n]
R = args.radius
shifts = [(dy, dx) for dy in range(-R, R + 1) for dx in range(-R, R + 1)]

for run, d in (r.split("=", 1) for r in args.runs):
    scores = np.zeros(len(shifts))
    for name in files:
        gt = load01(os.path.join(args.data, {"T2": "testB", "T1": "testA"}[args.target], name))
        mask = load01(os.path.join(args.data, "testMask", name)) > 0.5
        pred = load01(find_pred(d, name))
        for i, (dy, dx) in enumerate(shifts):
            _, smap = structural_similarity(gt, np.roll(pred, (dy, dx), axis=(0, 1)), data_range=1.0, full=True)
            scores[i] += smap[mask].mean()
    scores /= len(files)
    grid = scores.reshape(2 * R + 1, 2 * R + 1)
    best = shifts[int(np.argmax(scores))]
    print(f"\n{run} vs real {args.target}: brain SSIM by shift (rows dy, cols dx, {-R}..{R}), {len(files)} slices")
    for dy, row in zip(range(-R, R + 1), grid):
        print(f"  dy={dy:+d}  " + "  ".join(f"{v:.3f}" for v in row))
    print(f"  best shift (dy, dx) = {best}  SSIM {scores.max():.3f}  vs unshifted {grid[R, R]:.3f}")
