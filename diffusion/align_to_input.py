#!/usr/bin/env python
"""
Undo a whole-pixel spatial shift in a model's outputs, using only the input T1.

Some SynDiff runs shift their outputs by a few pixels (up to ~8 px, a different amount in each run). For
each output slice this finds the integer shift (dy, dx), both within +-radius, that best aligns the output's
brain silhouette with the silhouette of its input T1 (testA, same filename), by maximising Dice. It applies
that shift (zero fill, no wrap-around) and saves the result under the same relative path in --out.

Only testA is read, never testB or testMask, so this is legitimate test-time post-processing: it uses
nothing the model was not given.

Silhouette = pixels above --thresh (0-1 scale), holes filled. The filling matters: fluid is dark in T1 and
bright in T2, so without it the T1 ventricles would be holes the output does not have.

Shift convention as in shift_check.py: (dy, dx) is the shift applied to the output, +dy down, +dx right.

Reads every layout evaluate.py reads, and keeps it in --out:
  <pred_dir>/<name>.png (diffusion)   <pred_dir>/fake_B/<name>.png (CUT)   <pred_dir>/<stem>_fake_B.png (CycleGAN)
Outputs: the shifted slices as grayscale PNGs, and shifts.csv (one row per slice, Dice before/after).

Usage:
  python diffusion/align_to_input.py --pred_dir <outputs> --out <outputs>_aligned --data /scratch/$USER/brats/brats_t1t2
  python evaluate.py --run <run>_aligned --pred_dir <outputs>_aligned --data /scratch/$USER/brats/brats_t1t2
"""
import argparse
import collections
import csv
import os

import numpy as np
from PIL import Image
from scipy import ndimage


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"))


def find_pred(pred_dir, name):
    stem = os.path.splitext(name)[0]
    for p in (os.path.join(pred_dir, name),
              os.path.join(pred_dir, "fake_B", name),
              os.path.join(pred_dir, f"{stem}_fake_B.png")):
        if os.path.exists(p):
            return p
    return None


def silhouette(img, thresh):
    return ndimage.binary_fill_holes(img > thresh * 255)


def shift(img, dy, dx):
    """img moved by dy rows down and dx columns right; uncovered pixels are 0."""
    h, w = img.shape
    out = np.zeros_like(img)
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        img[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def dice(a, b):
    total = a.sum() + b.sum()
    return 2.0 * np.logical_and(a, b).sum() / total if total else 0.0


def best_shift(pred_mask, ref_mask, radius):
    """(dy, dx, Dice) maximising Dice; ties go to the smallest shift."""
    best = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            key = (dice(shift(pred_mask, dy, dx), ref_mask), -(abs(dy) + abs(dx)))
            if best is None or key > best[0]:
                best = (key, dy, dx)
    return best[1], best[2], best[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="a model's generated T2 slices (any evaluate.py layout)")
    ap.add_argument("--out", required=True, help="new folder for the shifted slices")
    ap.add_argument("--data", default=os.path.expandvars("/scratch/$USER/brats/brats_t1t2"),
                    help="dataset folder; only its testA (input T1) is read")
    ap.add_argument("--radius", type=int, default=16, help="largest shift tried, in pixels, per axis")
    ap.add_argument("--thresh", type=float, default=0.05, help="brain = above this (0-1 scale), holes filled")
    ap.add_argument("--apply", choices=["slice", "median"], default="slice",
                    help="slice: each slice's own shift; median: the run's median shift on every slice")
    args = ap.parse_args()
    if os.path.abspath(args.out) == os.path.abspath(args.pred_dir):
        raise SystemExit("--out must be a new folder, not --pred_dir")

    test_a = os.path.join(args.data, "testA")
    rows = []
    for name in sorted(f for f in os.listdir(test_a) if f.endswith(".png")):
        p = find_pred(args.pred_dir, name)
        if p is None:
            continue
        pred, t1 = load_gray(p), load_gray(os.path.join(test_a, name))
        if pred.shape != t1.shape:
            raise SystemExit(f"Size mismatch for {name}: output {pred.shape} vs input T1 {t1.shape}")
        pred_mask, ref_mask = silhouette(pred, args.thresh), silhouette(t1, args.thresh)
        dy, dx, d = best_shift(pred_mask, ref_mask, args.radius)
        rows.append({"file": name, "path": os.path.relpath(p, args.pred_dir), "dy": dy, "dx": dx,
                     "dice_before": round(dice(pred_mask, ref_mask), 4), "dice_after": round(d, 4)})
        if len(rows) % 500 == 0:
            print(f"  {len(rows)} slices", flush=True)
    if not rows:
        raise SystemExit(f"No outputs in {args.pred_dir} match a testA filename")

    dys, dxs = np.array([r["dy"] for r in rows]), np.array([r["dx"] for r in rows])
    med = (int(np.round(np.median(dys))), int(np.round(np.median(dxs))))
    for r in rows:
        dy, dx = med if args.apply == "median" else (r["dy"], r["dx"])
        r["applied_dy"], r["applied_dx"] = dy, dx
        dst = os.path.join(args.out, r["path"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        Image.fromarray(shift(load_gray(os.path.join(args.pred_dir, r["path"])), dy, dx)).save(dst)

    with open(os.path.join(args.out, "shifts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    q = lambda v: "[" + ", ".join(f"{x:+.0f}" for x in np.percentile(v, [25, 75])) + "]"
    counts = collections.Counter(zip(dys.tolist(), dxs.tolist())).most_common()
    print(f"\n{args.pred_dir}: {len(rows)} slices, shift that best aligns each output with its input T1")
    print(f"  median (dy, dx) = ({med[0]:+d}, {med[1]:+d})   IQR dy {q(dys)}  dx {q(dxs)}   "
          f"range dy [{dys.min():+d}, {dys.max():+d}]  dx [{dxs.min():+d}, {dxs.max():+d}]")
    print("  slices per shift: " + "  ".join(f"({dy:+d},{dx:+d}) {n}" for (dy, dx), n in counts[:12])
          + (f"  ... {len(counts) - 12} other shifts" if len(counts) > 12 else ""))
    print(f"  slices per dy: {dict(sorted(collections.Counter(dys.tolist()).items()))}")
    print(f"  slices per dx: {dict(sorted(collections.Counter(dxs.tolist()).items()))}")
    print(f"  silhouette Dice with input T1: {np.mean([r['dice_before'] for r in rows]):.3f} before, "
          f"{np.mean([r['dice_after'] for r in rows]):.3f} after")
    edge = int(np.sum((np.abs(dys) == args.radius) | (np.abs(dxs) == args.radius)))
    if edge:
        print(f"  WARNING: {edge} slices at the search limit +-{args.radius}; the true shift may be larger")
    applied = "each slice's own shift" if args.apply == "slice" else f"the median shift {med} on every slice"
    print(f"  applied {applied} -> {args.out} (shifts.csv has every slice)")


if __name__ == "__main__":
    main()
