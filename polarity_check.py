#!/usr/bin/env python
"""
Does a model's FINE detail follow the T1 input or the real T2?

For each eval slice we keep only fine detail (image minus a Gaussian-blurred copy),
then measure the Pearson correlation, inside the brain, between the model's fine detail and
  - the input T1's fine detail
  - the real T2's fine detail
A good T1->T2 translator should correlate positively with T2. A model that kept T1's
local contrast (e.g. dark fluid lines between folds) will correlate positively with T1
and negatively with T2.

Usage:
  python polarity_check.py --runs identity=/scratch/$USER/brats/brats_t1t2/testA cut_v1=<dir> cyclegan_v1=<dir>
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def load01(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float64) / 255.0


def find_pred(d, name):
    stem = os.path.splitext(name)[0]
    for p in (os.path.join(d, name), os.path.join(d, "fake_B", name), os.path.join(d, f"{stem}_fake_B.png")):
        if os.path.exists(p):
            return p


def detail(img, sigma):
    return img - gaussian_filter(img, sigma)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="name=pred_dir pairs")
ap.add_argument("--data", default=os.path.expandvars("/scratch/$USER/brats/brats_t1t2"))
ap.add_argument("--sigma", type=float, default=2.0, help="blur size in pixels; detail = finer than this")
args = ap.parse_args()

files = json.load(open(os.path.join(args.data, "split.json")))["eval_subset"]["files"]
print(f"Fine-detail correlation inside the brain (blur sigma {args.sigma}px), {len(files)} eval slices")
print(f"{'run':<14}{'with input T1':>15}{'with real T2':>15}")

ref = []
for name in files:
    t1 = load01(os.path.join(args.data, "testA", name))
    t2 = load01(os.path.join(args.data, "testB", name))
    m = gaussian_filter((load01(os.path.join(args.data, "testMask", name)) > 0.5).astype(float), 1) > 0.99  # skip brain edge
    ref.append((name, detail(t1, args.sigma), detail(t2, args.sigma), m))

print(f"{'(T1 vs T2)':<14}{'':>15}{np.mean([corr(a[m], b[m]) for _, a, b, m in ref]):>15.3f}")
for run, d in (r.split("=", 1) for r in args.runs):
    c1, c2 = [], []
    for name, d1, d2, m in ref:
        dp = detail(load01(find_pred(d, name)), args.sigma)
        c1.append(corr(dp[m], d1[m]))
        c2.append(corr(dp[m], d2[m]))
    print(f"{run:<14}{np.mean(c1):>15.3f}{np.mean(c2):>15.3f}")
