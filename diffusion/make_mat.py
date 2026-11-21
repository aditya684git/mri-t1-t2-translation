#!/usr/bin/env python
"""
Convert our preprocessed PNG dataset into the six .mat files SynDiff reads:
  data_{train,val,test}_{T1,T2}.mat   +  test_files.json  +  manifest.json

Format copied from SynDiff_sample_data/T1.mat (checked with h5py): an HDF5 file with one dataset
"data_fs", float32, shape (N, W, H), contiguous, no compression, values in [0, 1]. SynDiff's
LoadDataSet transposes each slice ((0, 2, 1), MATLAB column-major order), so we store every image
transposed and the loader hands back the upright image, identical to our PNG.

Splits (all from split.json; nothing new is invented):
  train  N random slices from trainA (T1) and N from trainB (T2), drawn and shuffled independently
         with a fixed seed. Equal lengths because SynDiff pairs the two files row by row; the rows
         are different patients, so training stays unpaired.
  val    every slice of a few test-split patients that are NOT in the eval subset, T1/T2 paired.
  test   exactly the eval_subset files, T1 from testA and T2 from testB, in split.json order.
         test_files.json lists the filenames in row order (infer_syndiff.py names outputs from it).

Usage:
  python make_mat.py --data /scratch/$USER/brats/brats_t1t2 --out /scratch/$USER/syndiff_data/n500 --n_train 500
  python make_mat.py --data ... --out /scratch/$USER/syndiff_data/n7514        # all: min(trainA, trainB)
"""
import argparse
import json
import os

import h5py
import numpy as np
from PIL import Image

SIZE = 256


def patient_of(name):
    return name.rsplit("_s", 1)[0]


def list_pngs(folder):
    return sorted(f for f in os.listdir(folder) if f.endswith(".png"))


def load01(path):
    img = Image.open(path)
    if img.mode != "L" or img.size != (SIZE, SIZE):
        raise SystemExit(f"{path}: expected {SIZE}x{SIZE} grayscale, got {img.mode} {img.size}")
    return np.asarray(img, dtype=np.float32) / 255.0


def write_mat(path, png_paths):
    if not png_paths:
        raise SystemExit(f"No slices for {path}")
    with h5py.File(path, "w") as f:
        d = f.create_dataset("data_fs", shape=(len(png_paths), SIZE, SIZE), dtype=np.float32)
        for i, p in enumerate(png_paths):
            d[i] = load01(p).T  # transposed, see docstring


def describe(path):
    with h5py.File(path, "r") as f:
        d = f["data_fs"]
        lo, hi = np.inf, -np.inf
        for s in range(0, d.shape[0], 500):  # in blocks: the full train file is ~2 GB
            block = d[s:s + 500]
            lo, hi = min(lo, float(block.min())), max(hi, float(block.max()))
        return d.shape, d.dtype, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="our dataset: trainA trainB testA testB split.json")
    ap.add_argument("--out", required=True, help="output folder, e.g. /scratch/$USER/syndiff_data/n500")
    ap.add_argument("--n_train", type=int, default=None, help="slices per contrast (default: all, min of the two)")
    ap.add_argument("--n_val_patients", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    split = json.load(open(os.path.join(args.data, "split.json")))
    rng = np.random.default_rng(args.seed)
    folder = lambda name: os.path.join(args.data, name)

    # train: independent random draws, equal length
    a, b = list_pngs(folder("trainA")), list_pngs(folder("trainB"))
    n = args.n_train or min(len(a), len(b))
    if n > min(len(a), len(b)):
        raise SystemExit(f"--n_train {n} but trainA has {len(a)} and trainB {len(b)} slices")
    train_t1 = [a[i] for i in rng.permutation(len(a))[:n]]
    train_t2 = [b[i] for i in rng.permutation(len(b))[:n]]

    # test: the fixed eval subset, in split.json order
    test_files = split["eval_subset"]["files"]

    # val: other test-split patients, never the eval-subset ones
    eval_patients = set(split["eval_subset"]["patients"]) | {patient_of(f) for f in test_files}
    candidates = sorted(p for p in split["patients"]["test"] if p not in eval_patients)
    if len(candidates) < args.n_val_patients:
        raise SystemExit(f"Only {len(candidates)} test patients outside the eval subset")
    val_patients = sorted(rng.choice(candidates, args.n_val_patients, replace=False).tolist())
    val_files = [f for f in list_pngs(folder("testA")) if patient_of(f) in val_patients]

    for name in test_files + val_files:
        for sub in ("testA", "testB"):
            if not os.path.exists(os.path.join(folder(sub), name)):
                raise SystemExit(f"Missing {sub}/{name}")

    jobs = {
        "data_train_T1.mat": [os.path.join(folder("trainA"), f) for f in train_t1],
        "data_train_T2.mat": [os.path.join(folder("trainB"), f) for f in train_t2],
        "data_val_T1.mat": [os.path.join(folder("testA"), f) for f in val_files],
        "data_val_T2.mat": [os.path.join(folder("testB"), f) for f in val_files],
        "data_test_T1.mat": [os.path.join(folder("testA"), f) for f in test_files],
        "data_test_T2.mat": [os.path.join(folder("testB"), f) for f in test_files],
    }

    os.makedirs(args.out, exist_ok=True)
    existing = [f for f in jobs if os.path.exists(os.path.join(args.out, f))]
    if existing and not args.overwrite:
        raise SystemExit(f"{args.out} already has {existing}; pass --overwrite to replace")

    for fname, paths in jobs.items():
        print(f"writing {fname} ({len(paths)} slices)", flush=True)
        write_mat(os.path.join(args.out, fname), paths)

    json.dump(test_files, open(os.path.join(args.out, "test_files.json"), "w"), indent=1)
    manifest = {"source": os.path.abspath(args.data), "seed": args.seed, "n_train": n,
                "val_patients": val_patients,
                "rows": {"train_T1": train_t1, "train_T2": train_t2, "val": val_files, "test": test_files}}
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)

    print(f"\n{args.out}  (val patients: {', '.join(val_patients)})")
    for fname in jobs:
        shape, dtype, lo, hi = describe(os.path.join(args.out, fname))
        print(f"  {fname:18s} {shape[0]:5d} slices  shape {shape}  {dtype}  min {lo:.4f}  max {hi:.4f}")
    print(f"  test_files.json    {len(test_files)} names in row order")


if __name__ == "__main__":
    main()
