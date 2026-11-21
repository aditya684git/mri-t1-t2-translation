#!/usr/bin/env python
"""
Per eval patient: median brain brightness of the real T2 vs a model's fake T2, and the MAE.
If some patients' real T2 is much darker than usual, their MAE should be much higher.

Usage:
  python intensity_check.py --pred_dir /scratch/$USER/mri_runs/results/cut_v1/test_latest/images
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image


def load01(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float64) / 255.0


def find_pred(d, name):
    stem = os.path.splitext(name)[0]
    for p in (os.path.join(d, name), os.path.join(d, "fake_B", name), os.path.join(d, f"{stem}_fake_B.png")):
        if os.path.exists(p):
            return p


ap = argparse.ArgumentParser()
ap.add_argument("--pred_dir", required=True)
ap.add_argument("--data", default=os.path.expandvars("/scratch/$USER/brats/brats_t1t2"))
args = ap.parse_args()

files = json.load(open(os.path.join(args.data, "split.json")))["eval_subset"]["files"]
acc = defaultdict(lambda: {"real": [], "fake": [], "err": []})
for n in files:
    m = load01(os.path.join(args.data, "testMask", n)) > 0.5
    real = load01(os.path.join(args.data, "testB", n))[m]
    fake = load01(find_pred(args.pred_dir, n))[m]
    a = acc[n.rsplit("_s", 1)[0]]
    a["real"].append(real)
    a["fake"].append(fake)
    a["err"].append(np.abs(fake - real))

print(f"{'patient':<18}{'real T2 median':>15}{'fake T2 median':>15}{'MAE':>8}")
rows = []
for p, a in acc.items():
    r, f, e = (np.concatenate(a[k]) for k in ("real", "fake", "err"))
    rows.append((p, np.median(r), np.median(f), e.mean()))
for p, r, f, e in sorted(rows, key=lambda x: x[1]):
    print(f"{p:<18}{r:15.3f}{f:15.3f}{e:8.3f}")
real_med = np.array([r[1] for r in rows]); mae = np.array([r[3] for r in rows])
print(f"\nCorrelation between real-T2 brightness and MAE: {np.corrcoef(real_med, mae)[0, 1]:+.2f}")
print("(strongly negative = darker real T2 -> bigger error, i.e. the scaling problem)")
