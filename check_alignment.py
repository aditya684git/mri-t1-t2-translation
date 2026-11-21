#!/usr/bin/env python
"""
Check that BraTS T1 and T2 scans are co-registered (voxel-for-voxel aligned).

For N random patients it:
  1. compares image shape and affine matrix of T1 vs T2,
  2. computes the Dice overlap of the two brain masks (voxels > 0;
     the scans are skull-stripped, so background is exactly 0),
  3. saves a figure per patient row: T1 | T2 | T1 + T2 brain outline | checkerboard.

Usage:
  python check_alignment.py --root /scratch/$USER/brats/extracted --n 5 --out alignment_check.png
"""
import argparse
import glob
import os
import random
import re

import matplotlib
matplotlib.use("Agg")  # no display on the cluster; just save to file
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def find_patients(root):
    """Return {patient_id: (t1_path, t2_path)} for every patient that has both."""
    patients = {}
    pattern = re.compile(r"(.+)_t1\.nii(\.gz)?$")
    for t1 in glob.glob(os.path.join(root, "**", "*_t1.nii*"), recursive=True):
        m = pattern.match(os.path.basename(t1))
        if not m:
            continue
        pid = m.group(1)
        folder = os.path.dirname(t1)
        t2 = [p for p in glob.glob(os.path.join(folder, pid + "_t2.nii*"))]
        if t2:
            patients[pid] = (t1, t2[0])
    return dict(sorted(patients.items()))


def to_display(sl):
    """Rotate so the brain is upright and scale to 0-1 for plotting."""
    sl = np.rot90(sl)
    brain = sl[sl > 0]
    if brain.size == 0:
        return sl
    lo, hi = np.percentile(brain, [0.5, 99.5])
    return np.clip((sl - lo) / (hi - lo + 1e-8), 0, 1) * (sl > 0)


def checkerboard(a, b, tile=30):
    yy, xx = np.indices(a.shape)
    mask = ((yy // tile) + (xx // tile)) % 2 == 0
    return np.where(mask, a, b)


def dice(m1, m2):
    return 2.0 * np.logical_and(m1, m2).sum() / (m1.sum() + m2.sum() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing the extracted BraTS patient folders")
    ap.add_argument("--n", type=int, default=5, help="number of patients to check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="alignment_check.png")
    args = ap.parse_args()

    patients = find_patients(args.root)
    print(f"Found {len(patients)} patients with both T1 and T2.")
    if not patients:
        raise SystemExit("No *_t1.nii(.gz) / *_t2.nii(.gz) pairs found under --root.")

    pids = random.Random(args.seed).sample(list(patients), min(args.n, len(patients)))

    fig, axes = plt.subplots(len(pids), 4, figsize=(12, 3 * len(pids)))
    axes = np.atleast_2d(axes)
    all_ok = True

    print(f"\n{'patient':<22}{'shape match':<13}{'affine match':<14}{'Dice':>6}")
    for row, pid in enumerate(pids):
        t1_img, t2_img = nib.load(patients[pid][0]), nib.load(patients[pid][1])
        t1 = np.asanyarray(t1_img.dataobj).astype(np.float32)
        t2 = np.asanyarray(t2_img.dataobj).astype(np.float32)

        same_shape = t1.shape == t2.shape
        same_affine = np.allclose(t1_img.affine, t2_img.affine, atol=1e-3)
        d = dice(t1 > 0, t2 > 0) if same_shape else float("nan")
        ok = same_shape and same_affine and d > 0.9
        all_ok &= ok
        print(f"{pid:<22}{str(same_shape):<13}{str(same_affine):<14}{d:6.3f}  {'OK' if ok else 'CHECK'}")

        # pick the axial slice with the most brain in it
        z = int(np.argmax((t1 > 0).sum(axis=(0, 1))))
        a, b = to_display(t1[:, :, z]), to_display(t2[:, :, z])

        axes[row, 0].imshow(a, cmap="gray"); axes[row, 0].set_title(f"{pid}\nT1, slice {z}", fontsize=8)
        axes[row, 1].imshow(b, cmap="gray"); axes[row, 1].set_title("T2", fontsize=8)
        axes[row, 2].imshow(a, cmap="gray")
        axes[row, 2].contour(np.rot90(t2[:, :, z] > 0), levels=[0.5], colors="r", linewidths=0.8)
        axes[row, 2].set_title("T1 + T2 brain outline (red)", fontsize=8)
        axes[row, 3].imshow(checkerboard(a, b), cmap="gray"); axes[row, 3].set_title(f"checkerboard, Dice {d:.3f}", fontsize=8)
        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"\nSaved figure to {args.out}")
    print("\nVERDICT:", "T1 and T2 look co-registered." if all_ok
          else "At least one patient needs a closer look (see CHECK rows and the figure).")


if __name__ == "__main__":
    main()
