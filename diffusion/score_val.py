#!/usr/bin/env python
"""
Score T1 -> T2 outputs on the validation patients, to choose the loss type, weight and epoch without
touching the 200 eval slices.

Validation slices: rows["val"] of a make_mat.py folder's manifest.json, i.e. every slice of 5 test-split
patients outside the eval subset. They are the same 5 patients for n500, n2000 and n7514 (make_mat.py draws
them with the same seed after the same random draws). Their PNGs sit in the dataset's testA/testB/testMask.

Per run:
  raw scores    brain-masked SSIM, PSNR and MAE against the real T2, computed by evaluate.py's own score(),
                with no alignment
  shift         align_to_input.py's input-based shift for each slice: the output's silhouette against its
                input T1, never testB or testMask. Reported as the median (dy, dx) and the share of slices
                within 1 px of (0, 0) on both axes
  aligned SSIM  brain SSIM after undoing each slice's shift: what the shift costs. --select uses the
                baseline's aligned SSIM as its quality floor
Writes <out_dir>/<run>_val_per_slice.csv and adds or replaces the run's row in <out_dir>/val_results.csv.

--select applies the README's selection rule (section 7) to every row of val_results.csv. Row names are
<exp>_e<epoch>; the data size is the n<number> part of <exp>, and the baseline run for it is syndiff_<data>.
A candidate must
  1. be epoch 1 or later
  2. have a median shift within 1 px of (0, 0) on both axes
  3. have raw SSIM >= the floor: the best aligned SSIM of any epoch >= 1 of the same-size baseline
     (no baseline rows for that size: no floor, so nothing of that size passes)
The passing candidate with the highest raw SSIM is selected (SSIM within 0.005 is a tie, broken by the lower
MAE). If none passes, it says so and lists the best row of each run instead of choosing.

Usage:
  python diffusion/score_val.py --run sc_n2000_mind_0.25_e30 --pred_dir <val outputs> \
      --syndata /scratch/$USER/syndiff_data/n2000 --data /scratch/$USER/brats/brats_t1t2
  python diffusion/score_val.py --select
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))  # the repo root, for evaluate.py
from align_to_input import best_shift, find_pred, load_gray, shift, silhouette  # noqa: E402

FIELDS = ["run", "n_slices", "n_missing", "ssim", "ssim_std", "psnr", "mae", "shift_dy", "shift_dx",
          "within_1px", "ssim_aligned"]


def score_run(args):
    from evaluate import score  # needs scikit-image
    names = json.load(open(os.path.join(args.syndata, "manifest.json")))["rows"]["val"]
    rows, missing = [], 0
    for name in names:
        p = find_pred(args.pred_dir, name)
        if p is None:
            missing += 1
            continue
        pred = load_gray(p)
        t1 = load_gray(os.path.join(args.data, "testA", name))
        gt = load_gray(os.path.join(args.data, "testB", name)) / 255.0
        mask = load_gray(os.path.join(args.data, "testMask", name)) > 127
        dy, dx, _ = best_shift(silhouette(pred, args.thresh), silhouette(t1, args.thresh), args.radius)
        raw = score(pred / 255.0, gt, mask)
        aligned = score(shift(pred, dy, dx) / 255.0, gt, mask)
        rows.append({"file": name, "ssim": raw["ssim"], "psnr": raw["psnr"], "mae": raw["mae"],
                     "dy": dy, "dx": dx, "ssim_aligned": aligned["ssim"]})
    if not rows:
        raise SystemExit(f"No validation outputs found in {args.pred_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{args.run}_val_per_slice.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    col = lambda k: np.array([r[k] for r in rows], dtype=float)
    dys, dxs = col("dy"), col("dx")
    summary = {"run": args.run, "n_slices": len(rows), "n_missing": missing,
               "ssim": round(col("ssim").mean(), 4), "ssim_std": round(col("ssim").std(), 4),
               "psnr": round(col("psnr").mean(), 2), "mae": round(col("mae").mean(), 4),
               "shift_dy": int(np.round(np.median(dys))), "shift_dx": int(np.round(np.median(dxs))),
               "within_1px": round(float(np.mean((np.abs(dys) <= 1) & (np.abs(dxs) <= 1))), 3),
               "ssim_aligned": round(col("ssim_aligned").mean(), 4)}

    table = os.path.join(args.out_dir, "val_results.csv")
    old = [r for r in csv.DictReader(open(table))] if os.path.exists(table) else []
    with open(table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows([r for r in old if r["run"] != args.run] + [summary])

    print(f"{args.run}: {len(rows)} validation slices ({missing} missing)")
    print(f"  raw      SSIM {summary['ssim']:.4f} +- {summary['ssim_std']:.4f}   PSNR {summary['psnr']:.2f} dB   "
          f"MAE {summary['mae']:.4f}")
    print(f"  shift    median (dy, dx) = ({summary['shift_dy']:+d}, {summary['shift_dx']:+d}), "
          f"{100 * summary['within_1px']:.0f}% of slices within 1 px of (0, 0)  [input-based, align_to_input.py]")
    print(f"  aligned  SSIM {summary['ssim_aligned']:.4f}  (after undoing each slice's shift; a baseline's sets --select's floor)")
    print(f"  -> {table}")


def parse_run(run):
    """'sc_n500_mind_0.25_e30' -> ('sc_n500_mind_0.25', 30, 'n500'); None for names not <exp>_e<epoch>."""
    m = re.fullmatch(r"(.+)_e(\d+)", run)
    if not m:
        return None
    exp, epoch = m.group(1), int(m.group(2))
    size = next((t for t in exp.split("_") if re.fullmatch(r"n\d+", t)), None)
    return exp, epoch, size


def baseline_floors(rows):
    """{data size: (best aligned SSIM over the baseline's epochs >= 1, that row's run)}."""
    floors = {}
    for r in rows:
        p = parse_run(r["run"])
        if p and p[2] and p[0] == f"syndiff_{p[2]}" and p[1] >= 1:
            v = float(r["ssim_aligned"])
            if p[2] not in floors or v > floors[p[2]][0]:
                floors[p[2]] = (v, r["run"])
    return floors


def why_not(r, floors):
    """None if r is a candidate under the selection rule, else the first reason it is not."""
    p = parse_run(r["run"])
    if p is None or p[2] is None:
        return "name not <exp>_e<epoch> with a data size n<N>"
    if p[1] == 0:
        return "epoch 0"
    if abs(int(r["shift_dy"])) > 1 or abs(int(r["shift_dx"])) > 1:
        return "median shift > 1 px"
    if p[2] not in floors:
        return f"no syndiff_{p[2]} baseline rows, so no floor"
    floor, src = floors[p[2]]
    if float(r["ssim"]) < floor:
        return f"raw SSIM below the floor {floor:.4f} ({src} aligned)"
    return None


def select(table):
    rows = list(csv.DictReader(open(table)))
    if not rows:
        raise SystemExit(f"{table} is empty")
    floors = baseline_floors(rows)
    # highest SSIM first; SSIM rounded to 0.005 so near-ties go to the lower MAE
    key = lambda r: (-round(float(r["ssim"]) / 0.005), float(r["mae"]))
    line = lambda r, note: (f"{r['run']:40s} {float(r['ssim']):7.4f} {float(r['psnr']):6.2f} {float(r['mae']):7.4f} "
                            f"({int(r['shift_dy']):+d},{int(r['shift_dx']):+d}){'':3s} {float(r['within_1px']):6.2f} "
                            f"{float(r['ssim_aligned']):8.4f}{note}")
    header = f"{'run':40s} {'SSIM':>7s} {'PSNR':>6s} {'MAE':>7s} {'shift':>9s} {'<=1px':>6s} {'aligned':>8s}"

    for size, (v, src) in sorted(floors.items()):
        print(f"quality floor for {size}: raw SSIM >= {v:.4f}, the best aligned SSIM of the baseline ({src})")
    print()
    print(header)
    for r in sorted(rows, key=lambda r: (why_not(r, floors) is not None, key(r))):
        reason = why_not(r, floors)
        print(line(r, "" if reason is None else f"   (excluded: {reason})"))

    passing = sorted((r for r in rows if why_not(r, floors) is None), key=key)
    if passing:
        print(f"\nSelected by the validation rule: {passing[0]['run']}")
        return
    print("\nNo candidate passes; nothing is selected. Best row of each run (highest raw SSIM, epoch >= 1):")
    best = {}
    for r in sorted(rows, key=key):
        p = parse_run(r["run"])
        if p is None or p[1] >= 1:
            best.setdefault(p[0] if p else r["run"], r)
    print(header)
    for exp, r in sorted(best.items(), key=lambda kv: key(kv[1])):
        print(line(r, f"   ({why_not(r, floors)})"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="row name, e.g. sc_n2000_mind_0.25_e30")
    ap.add_argument("--pred_dir", help="T1 -> T2 outputs for the validation slices (infer_syndiff.py --phase val)")
    ap.add_argument("--syndata", help="make_mat.py folder whose manifest.json lists the validation slices")
    ap.add_argument("--data", default=os.path.expandvars("/scratch/$USER/brats/brats_t1t2"), help="dataset with testA/testB/testMask")
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(HERE), "results", "syndiff_val"))
    ap.add_argument("--radius", type=int, default=16, help="align_to_input.py's search radius")
    ap.add_argument("--thresh", type=float, default=0.05, help="align_to_input.py's silhouette threshold")
    ap.add_argument("--select", action="store_true", help="rank val_results.csv by the selection rule and stop")
    args = ap.parse_args()
    if args.select:
        return select(os.path.join(args.out_dir, "val_results.csv"))
    if not (args.run and args.pred_dir and args.syndata):
        ap.error("--run, --pred_dir and --syndata are required (or --select)")
    score_run(args)


if __name__ == "__main__":
    main()
