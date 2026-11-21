#!/usr/bin/env python
"""
Turn BraTS 2021 T1/T2 volumes into the PNG folders that CUT, CycleGAN and the
diffusion model all train and test on.

Output layout (under --out):
  trainA/    T1 slices from training half 1
  trainB/    T2 slices from training half 2   (different patients -> truly unpaired)
  testA/     T1 slices from the test patients
  testB/     T2 slices from the SAME test patients, SAME filenames (ground truth)
  testMask/  brain mask for every test slice, SAME filenames (for masked metrics)
  split.json patient lists, settings, slice counts, and the fixed eval subset

Each step:
  1. Pick --n_patients at random (fixed seed), split by PATIENT into test and train,
     then split train into two halves (A and B).
  2. Load each volume with nibabel.
  3. Normalise per volume: clip to the 0.5th-99.5th percentile of BRAIN voxels
     (background excluded, it is all zeros), scale to 0-1, keep background at 0.
  4. Keep axial slices whose brain area is >= --min_frac of the largest brain slice,
     then take every --stride-th one.
  5. Rotate upright, pad 240x240 -> 256x256, save as 8-bit grayscale PNG.

Usage:
  python preprocess.py --root /scratch/$USER/brats/extracted --out /scratch/$USER/brats/brats_t1t2
"""
import argparse
import glob
import json
import os
import random
import re
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from PIL import Image

PAD = 8  # 240 + 2*8 = 256


def find_patients(root):
    patients = {}
    pattern = re.compile(r"(.+)_t1\.nii(\.gz)?$")
    for t1 in glob.glob(os.path.join(root, "**", "*_t1.nii*"), recursive=True):
        m = pattern.match(os.path.basename(t1))
        if not m:
            continue
        pid = m.group(1)
        t2 = glob.glob(os.path.join(os.path.dirname(t1), pid + "_t2.nii*"))
        if t2:
            patients[pid] = (t1, t2[0])
    return dict(sorted(patients.items()))


def load(path):
    return np.asanyarray(nib.load(path).dataobj).astype(np.float32)


def normalise(vol, mask):
    lo, hi = np.percentile(vol[mask], [0.5, 99.5])
    out = np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1)
    out[~mask] = 0
    return out


def select_slices(mask, min_frac, stride):
    area = mask.sum(axis=(0, 1))
    keep = np.where(area >= min_frac * area.max())[0]
    return keep[::stride].tolist()


def to_png(sl):
    """2D slice (240x240, values 0-1 or bool) -> upright, padded 256x256 uint8 image."""
    sl = np.rot90(sl)
    sl = np.pad(sl, PAD, mode="constant", constant_values=0)
    return Image.fromarray((sl * 255).round().astype(np.uint8), mode="L")


def process(job):
    """job = (pid, t1_path, t2_path, role, out_dir, min_frac, stride). Returns (pid, slice list)."""
    pid, t1_path, t2_path, role, out, min_frac, stride = job
    t1, t2 = load(t1_path), load(t2_path)
    mask = (t1 > 0) | (t2 > 0)
    zs = select_slices(mask, min_frac, stride)
    t1n, t2n = normalise(t1, mask), normalise(t2, mask)

    for z in zs:
        name = f"{pid}_s{z:03d}.png"
        if role == "A":
            to_png(t1n[:, :, z]).save(os.path.join(out, "trainA", name))
        elif role == "B":
            to_png(t2n[:, :, z]).save(os.path.join(out, "trainB", name))
        else:  # test: both contrasts + mask, identical filenames
            to_png(t1n[:, :, z]).save(os.path.join(out, "testA", name))
            to_png(t2n[:, :, z]).save(os.path.join(out, "testB", name))
            to_png(mask[:, :, z].astype(np.float32)).save(os.path.join(out, "testMask", name))
    return pid, zs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with the extracted BraTS patient folders")
    ap.add_argument("--out", required=True, help="output dataset folder (gets trainA/trainB/testA/testB)")
    ap.add_argument("--n_patients", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=80)
    ap.add_argument("--min_frac", type=float, default=0.15, help="keep slices with brain area >= this fraction of the max")
    ap.add_argument("--stride", type=int, default=2, help="take every Nth kept slice")
    ap.add_argument("--eval_patients", type=int, default=20, help="test patients in the fixed eval subset")
    ap.add_argument("--eval_slices", type=int, default=10, help="slices per patient in the eval subset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    args = ap.parse_args()

    # ---- 1. patient-level split ------------------------------------------------
    patients = find_patients(args.root)
    print(f"Found {len(patients)} patients with T1 and T2.")
    if len(patients) < args.n_patients:
        raise SystemExit(f"Asked for {args.n_patients} patients but only {len(patients)} found.")

    rng = random.Random(args.seed)
    chosen = rng.sample(sorted(patients), args.n_patients)
    test = sorted(chosen[: args.n_test])
    train = chosen[args.n_test:]
    half = len(train) // 2
    train_a, train_b = sorted(train[:half]), sorted(train[half:])
    print(f"Split: {len(train_a)} trainA (T1) | {len(train_b)} trainB (T2) | {len(test)} test patients")

    for sub in ["trainA", "trainB", "testA", "testB", "testMask"]:
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    # ---- 2-5. slice, normalise, save (parallel over patients) ------------------
    jobs = [(p, *patients[p], role, args.out, args.min_frac, args.stride)
            for role, group in [("A", train_a), ("B", train_b), ("test", test)] for p in group]
    print(f"Processing {len(jobs)} patients with {args.workers} workers...")
    slices = {}
    with Pool(args.workers) as pool:
        for i, (pid, zs) in enumerate(pool.imap_unordered(process, jobs), 1):
            slices[pid] = zs
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} patients done")

    # ---- fixed evaluation subset (same images for all three models) -----------
    eval_pids = sorted(random.Random(args.seed + 1).sample(test, min(args.eval_patients, len(test))))
    eval_files = []
    for p in eval_pids:
        zs = slices[p]
        idx = np.linspace(0, len(zs) - 1, min(args.eval_slices, len(zs))).round().astype(int)
        eval_files += [f"{p}_s{zs[i]:03d}.png" for i in sorted(set(idx))]

    counts = {sub: len(os.listdir(os.path.join(args.out, sub)))
              for sub in ["trainA", "trainB", "testA", "testB", "testMask"]}
    split = {
        "dataset": "BraTS 2021 Task 1 (Kaggle mirror dschettler8845/brats-2021-task1), T1 and T2 only",
        "settings": {k: v for k, v in vars(args).items() if k not in ("root", "out", "workers")},
        "image": "axial slices, rot90, padded 240->256, 8-bit grayscale PNG, per-volume 0.5-99.5 pct of brain voxels -> 0-1",
        "patients": {"trainA": train_a, "trainB": train_b, "test": test},
        "slice_counts": counts,
        "eval_subset": {"patients": eval_pids, "files": eval_files},
    }
    with open(os.path.join(args.out, "split.json"), "w") as f:
        json.dump(split, f, indent=2)

    print("\nSlices written:")
    for k, v in counts.items():
        print(f"  {k:<9} {v}")
    print(f"Eval subset: {len(eval_files)} slices from {len(eval_pids)} patients")
    print(f"Split saved to {os.path.join(args.out, 'split.json')}")


if __name__ == "__main__":
    main()
