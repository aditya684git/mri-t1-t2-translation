"""Per-patient paired comparison of two runs on the eval subset.

Averages each run's per-slice scores (from evaluate.py's *_eval_per_slice.csv) per patient, counts the
patients where run B improves on run A, and runs a paired two-sided Wilcoxon signed-rank test on the
per-patient means, for brain SSIM, PSNR and MAE (for MAE, lower is better).

Usage (from the repo root):
  python scripts/wilcoxon_per_patient.py
  python scripts/wilcoxon_per_patient.py --a results/cyclegan_v1_eval_per_slice.csv \
      --b results/diffusion_sc_v1_eval_per_slice.csv
"""
import argparse
import csv
import os
from collections import defaultdict

from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = [("ssim", "higher"), ("psnr", "higher"), ("mae", "lower")]


def per_patient_means(path):
    vals = defaultdict(lambda: defaultdict(list))
    files = set()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            files.add(r["file"])
            for m, _ in METRICS:
                vals[r["patient"]][m].append(float(r[m]))
    means = {p: {m: sum(v) / len(v) for m, v in d.items()} for p, d in vals.items()}
    return means, files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default=os.path.join(ROOT, "results", "cyclegan_v1_eval_per_slice.csv"),
                    help="baseline run (default: CycleGAN)")
    ap.add_argument("--b", default=os.path.join(ROOT, "results", "diffusion_sc_v1_eval_per_slice.csv"),
                    help="compared run (default: SynDiff + silhouette structure loss)")
    args = ap.parse_args()

    a, files_a = per_patient_means(args.a)
    b, files_b = per_patient_means(args.b)
    if files_a != files_b:
        raise SystemExit(f"the two CSVs score different slices ({len(files_a ^ files_b)} differ)")
    patients = sorted(a)
    n_slices = len(files_a)

    print(f"A: {os.path.relpath(args.a, ROOT)}")
    print(f"B: {os.path.relpath(args.b, ROOT)}")
    print(f"{len(patients)} patients, {n_slices} slices; per-patient means of brain metrics")
    print("Paired two-sided Wilcoxon signed-rank test (scipy.stats.wilcoxon) on the per-patient means\n")
    print(f"{'metric':<6} {'mean A':>8} {'mean B':>8} {'B better':>9} {'W':>6} {'p':>12}")
    for m, better in METRICS:
        xa = [a[p][m] for p in patients]
        xb = [b[p][m] for p in patients]
        improved = sum((y > x) if better == "higher" else (y < x) for x, y in zip(xa, xb))
        res = wilcoxon(xb, xa, alternative="two-sided")
        print(f"{m:<6} {sum(xa) / len(xa):8.4f} {sum(xb) / len(xb):8.4f} "
              f"{improved:>5}/{len(patients):<3} {res.statistic:6.1f} {res.pvalue:12.3g}")
    print("\nB better: patients where B's mean is higher (SSIM, PSNR) or lower (MAE) than A's.")


if __name__ == "__main__":
    main()
