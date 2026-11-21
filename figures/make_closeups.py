#!/usr/bin/env python
"""
Close-up comparison figures, one per slice, for the slides. Run on Palmetto, where the outputs are.

Top row:    input T1 | CUT | CycleGAN | SynDiff original | SynDiff + silhouette loss | real T2   (all cropped)
Bottom row: the full T1 with the crop box | |output - real T2| for each of the four models, inside the brain
            mask, on one shared scale (each titled with that slice's brain SSIM from results/*_per_slice.csv)

Output folders default to the ones in diffusion/README.md; model outputs are found in any evaluate.py layout.
The crop is --box Y0 X0 H W for every slice; a slice can carry its own box as NAME:Y0,X0,H,W.

Usage (from the repo root):
  python figures/make_closeups.py
  python figures/make_closeups.py --files BraTS2021_00271_s084:70,40,110,110 BraTS2021_00359_s082 --box 60 60 136 136
Outputs: figures/slides/closeup_<slice>.png (300 dpi) and .svg
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from make_slide_figures import COLOUR, LABEL  # same colours, labels and font sizes as the other figures  # noqa: E402

DEFAULT_FILES = ["BraTS2021_00271_s084", "BraTS2021_00359_s082", "BraTS2021_01509_s117"]  # tumour, ventricles, top slice
RUNS = ["cut_v1", "cyclegan_v1", "diffusion_v1", "diffusion_sc_v1"]
SHORT = {r: LABEL[r].replace("SynDiff ", "SynDiff\n").replace("SynDiff\n+ ", "SynDiff +\n") for r in RUNS}  # two lines


def load01(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def find_pred(pred_dir, name):
    stem = os.path.splitext(name)[0]
    for p in (os.path.join(pred_dir, name), os.path.join(pred_dir, "fake_B", name),
              os.path.join(pred_dir, f"{stem}_fake_B.png")):
        if os.path.exists(p):
            return p
    raise SystemExit(f"No output for {name} in {pred_dir}")


def slice_ssim(run):
    """{filename: brain SSIM} from the repo's per-slice scores, if present."""
    path = os.path.join(ROOT, "results", f"{run}_eval_per_slice.csv")
    if not os.path.exists(path):
        return {}
    return {r["file"]: float(r["ssim"]) for r in csv.DictReader(open(path))}


def parse_file(spec, default_box):
    name, _, box = spec.partition(":")
    name = name if name.endswith(".png") else name + ".png"
    return name, (tuple(int(v) for v in box.split(",")) if box else tuple(default_box))


def figure(name, box, dirs, data, emax, ssims, out_dir):
    y0, x0, h, w = box
    crop = lambda img: img[y0:y0 + h, x0:x0 + w]
    t1, t2 = load01(os.path.join(data, "testA", name)), load01(os.path.join(data, "testB", name))
    mask = load01(os.path.join(data, "testMask", name)) > 0.5
    preds = {run: load01(find_pred(dirs[run], name)) for run in RUNS}
    if not (0 <= y0 and 0 <= x0 and y0 + h <= t1.shape[0] and x0 + w <= t1.shape[1]):
        raise SystemExit(f"--box {box} does not fit inside {t1.shape} for {name}")

    fig, axes = plt.subplots(2, 6, figsize=(16, 6.6), gridspec_kw={"hspace": 0.28, "wspace": 0.04})
    top = [("input T1", "black", t1)] + [(SHORT[r], COLOUR[r], preds[r]) for r in RUNS] + [("real T2", "black", t2)]
    for ax, (title, colour, img) in zip(axes[0], top):
        ax.imshow(crop(img), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, color=colour, fontsize=16)
    ax = axes[1, 0]
    ax.imshow(t1, cmap="gray", vmin=0, vmax=1)
    ax.add_patch(Rectangle((x0 - .5, y0 - .5), w, h, fill=False, edgecolor="#F0E442", linewidth=2.5))
    ax.set_title("crop location", fontsize=16)
    for ax, run in zip(axes[1, 1:5], RUNS):
        err = np.where(mask, np.abs(preds[run] - t2), 0)
        im = ax.imshow(crop(err), cmap="magma", vmin=0, vmax=emax, interpolation="nearest")
        s = ssims[run].get(name)
        ax.set_title("|error|" + (f"\nslice SSIM {s:.2f}" if s is not None else ""), color=COLOUR[run], fontsize=16)
    axes[1, 5].axis("off")
    cax = axes[1, 5].inset_axes([0.15, 0.05, 0.12, 0.9])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("|output − real T2|", fontsize=16)
    cb.ax.tick_params(labelsize=14)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    stem = os.path.splitext(name)[0]
    fig.suptitle(f"{stem}   (crop rows {y0}–{y0 + h}, cols {x0}–{x0 + w}; error inside the brain mask)",
                 fontsize=18, y=1.0)
    for ext, kw in (("png", {"dpi": 300}), ("svg", {})):
        fig.savefig(os.path.join(out_dir, f"closeup_{stem}.{ext}"), bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)
    print(f"  closeup_{stem}.png / .svg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES, help="slice names, optionally NAME:Y0,X0,H,W")
    ap.add_argument("--box", nargs=4, type=int, default=[48, 48, 160, 160], metavar=("Y0", "X0", "H", "W"),
                    help="crop for slices without their own box")
    ap.add_argument("--data", default=os.path.expandvars("/scratch/$USER/brats/brats_t1t2"))
    ap.add_argument("--cut", default=os.path.expandvars("/scratch/$USER/mri_runs/results/cut_v1/test_latest/images"))
    ap.add_argument("--cyclegan", default=os.path.expandvars("/scratch/$USER/mri_runs/results/cyclegan_v1/test_latest/images"))
    ap.add_argument("--original", default=os.path.expandvars("/scratch/$USER/syndiff_outputs/diffusion_v1"))
    ap.add_argument("--fixed", default=os.path.expandvars("/scratch/$USER/syndiff_outputs/diffusion_sc_v1"))
    ap.add_argument("--emax", type=float, default=0.5, help="top of the error colour scale (0-1 intensity)")
    ap.add_argument("--out", default=os.path.join(HERE, "slides"))
    args = ap.parse_args()
    dirs = dict(zip(RUNS, (args.cut, args.cyclegan, args.original, args.fixed)))
    ssims = {run: slice_ssim(run) for run in RUNS}
    os.makedirs(args.out, exist_ok=True)
    for spec in args.files:
        name, box = parse_file(spec, args.box)
        figure(name, box, dirs, args.data, args.emax, ssims, args.out)


if __name__ == "__main__":
    main()
