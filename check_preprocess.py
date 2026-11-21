#!/usr/bin/env python
"""Show 4 random slices from each output folder so we can eyeball the preprocessing.

The three test rows (testA, testB, testMask) use the SAME four filenames, so each
column should show one slice as T1, as T2, and as its brain mask.

Usage:
  python check_preprocess.py --data /scratch/$USER/brats/brats_t1t2 --out preprocess_check.png
"""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="preprocess_check.png")
args = ap.parse_args()

folders = ["trainA", "trainB", "testA", "testB", "testMask"]
test_names = random.Random(0).sample(sorted(os.listdir(os.path.join(args.data, "testA"))), 4)

fig, ax = plt.subplots(len(folders), 4, figsize=(10, 13))
for r, f in enumerate(folders):
    if f.startswith("test"):
        names = test_names
    else:
        names = random.Random(r).sample(sorted(os.listdir(os.path.join(args.data, f))), 4)
    for c, n in enumerate(names):
        ax[r, c].imshow(Image.open(os.path.join(args.data, f, n)), cmap="gray", vmin=0, vmax=255)
        ax[r, c].set_title(f"{f}\n{n}", fontsize=7)
        ax[r, c].axis("off")

plt.tight_layout()
plt.savefig(args.out, dpi=90)
print("saved", args.out)
