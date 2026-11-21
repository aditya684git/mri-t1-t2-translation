#!/usr/bin/env python
"""
Score generated T2 slices against the real T2, inside the brain mask only.

Works for any model as long as its outputs keep our test filenames. It finds
each prediction in any of these layouts:
  <pred_dir>/<name>.png                (Aditya's diffusion outputs, if saved this way)
  <pred_dir>/fake_B/<name>.png         (CUT repo)
  <pred_dir>/<name>_fake_B.png         (CycleGAN repo)

Metrics (images scaled to 0-1, computed only where testMask is brain):
  SSIM  structural similarity: full SSIM map, averaged over brain pixels
  PSNR  10*log10(1 / MSE over brain pixels)
  MAE   mean absolute error over brain pixels
Also reported, for comparison with papers only: whole-image SSIM and PSNR
(background included, which inflates both).

Outputs:
  <out_dir>/<run>_<subset>_per_slice.csv   one row per slice (sort it to find failure cases)
  <out_dir>/results.csv                    one summary row per run+subset (the results table)

Usage:
  python evaluate.py --run cut_v1 --pred_dir /scratch/$USER/mri_runs/results/cut_v1/test_latest/images
  python evaluate.py --run cut_v1 --pred_dir ... --subset all       # every test slice, not just the 200
  python evaluate.py --run identity --pred_dir /scratch/$USER/brats/brats_t1t2/testA   # "do nothing" baseline
"""
import argparse
import csv
import json
import os

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


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


def score(pred, gt, mask):
    err = pred[mask] - gt[mask]
    mse = np.mean(err ** 2)
    _, ssim_map = structural_similarity(gt, pred, data_range=1.0, full=True)
    mse_full = np.mean((pred - gt) ** 2)
    return {
        "ssim": float(ssim_map[mask].mean()),
        "psnr": float(10 * np.log10(1.0 / mse)) if mse > 1e-10 else 100.0,  # cap: perfect match
        "mae": float(np.mean(np.abs(err))),
        # whole-image versions (background included), only for comparing with papers
        "ssim_full": float(ssim_map.mean()),
        "psnr_full": float(10 * np.log10(1.0 / mse_full)) if mse_full > 1e-10 else 100.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="name for this row of the results table, e.g. cut_v1")
    ap.add_argument("--pred_dir", required=True, help="folder holding the model's generated T2 slices")
    ap.add_argument("--data", default=os.path.expandvars("/scratch/$USER/brats/brats_t1t2"))
    ap.add_argument("--subset", choices=["eval", "all"], default="eval",
                    help="eval = the fixed 200 slices in split.json; all = every test slice")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    split = json.load(open(os.path.join(args.data, "split.json")))
    if args.subset == "eval":
        names = split["eval_subset"]["files"]
    else:
        names = sorted(os.listdir(os.path.join(args.data, "testB")))

    rows, missing = [], []
    for i, name in enumerate(names, 1):
        p = find_pred(args.pred_dir, name)
        if p is None:
            missing.append(name)
            continue
        gt = load01(os.path.join(args.data, "testB", name))
        mask = load01(os.path.join(args.data, "testMask", name)) > 0.5
        pred = load01(p)
        if pred.shape != gt.shape:
            raise SystemExit(f"Size mismatch for {name}: prediction {pred.shape} vs real T2 {gt.shape}")
        rows.append({"file": name, "patient": name.rsplit("_s", 1)[0],
                     "brain_pixels": int(mask.sum()), **score(pred, gt, mask)})
        if i % 500 == 0:
            print(f"  {i}/{len(names)} scored")

    if missing:
        print(f"WARNING: {len(missing)} of {len(names)} predictions not found, e.g. {missing[:3]}")
    if not rows:
        raise SystemExit("Nothing scored. Check --pred_dir.")

    os.makedirs(args.out_dir, exist_ok=True)

    per_slice = os.path.join(args.out_dir, f"{args.run}_{args.subset}_per_slice.csv")
    with open(per_slice, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {"run": args.run, "subset": args.subset, "n_slices": len(rows),
               "n_patients": len({r["patient"] for r in rows}), "n_missing": len(missing)}
    for m in ("ssim", "psnr", "mae", "ssim_full", "psnr_full"):
        v = np.array([r[m] for r in rows])
        summary[f"{m}_mean"] = round(float(v.mean()), 4)
        summary[f"{m}_std"] = round(float(v.std()), 4)

    # results table: replace any earlier row for the same run+subset, keep the rest
    table = os.path.join(args.out_dir, "results.csv")
    old = []
    if os.path.exists(table):
        old = [r for r in csv.DictReader(open(table))
               if not (r["run"] == args.run and r["subset"] == args.subset)]
    with open(table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        w.writeheader()
        w.writerows(old + [summary])

    print(f"\n{args.run} ({args.subset}): {len(rows)} slices from {summary['n_patients']} patients")
    print(f"  SSIM {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}   (higher is better, max 1)")
    print(f"  PSNR {summary['psnr_mean']:.2f} ± {summary['psnr_std']:.2f} dB (higher is better)")
    print(f"  MAE  {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}   (lower is better, 0-1 scale)")
    print(f"  whole image (background included, for comparing with papers): "
          f"SSIM {summary['ssim_full_mean']:.4f}, PSNR {summary['psnr_full_mean']:.2f} dB")
    worst = sorted(rows, key=lambda r: r["ssim"])[:3]
    print("  Lowest-SSIM slices:", ", ".join(f"{r['file']} ({r['ssim']:.3f})" for r in worst))
    print(f"\nPer-slice scores: {per_slice}\nResults table:    {table}")


if __name__ == "__main__":
    main()
