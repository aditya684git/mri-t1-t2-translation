"""Cut the individual image panels out of three existing figures, for the blog's figure grids.

The source slices and model outputs live on the cluster, so the blog figures are rebuilt from the
figures already in this folder instead of being re-plotted:

  compare_cut_vs_cyclegan.png  (compare_outputs.py, 6 rows x 6 columns)   -> blog Figure 7
  final_four_models.png        (compare_outputs.py, 6 rows x 10 columns)  -> blog Figure 18
  alignment_check.png          (check_alignment.py, 5 rows x 4 columns)   -> blog Figure 2

Each panel is found as a connected block of non-white pixels, checked against the expected grid, and
saved with PIL's crop: no resizing, no re-plotting, pixel values unchanged. Titles and the white
margins between panels are left out; the blog draws its own labels.

compare_outputs.py draws the images with cmap gray (vmin 0, vmax 1) and the error maps
|output - real T2| inside the brain mask with cmap inferno (vmin 0, vmax 0.5), so the blog's colour bar
for the error maps is inferno over 0-0.5.

Usage (from the repo root):  python figures/make_blog_panels.py [--out docs/assets/panels]
"""
import argparse
import os

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ROWS6 = ["01598_s027", "00271_s084", "01509_s117", "01143_s121", "01648_s074", "01163_s094"]
FIGURES = {
    # name: (source file, row names, column names, rows kept for the blog)
    "fig7": ("compare_cut_vs_cyclegan.png", ROWS6,
             ["t1", "cut", "cyclegan", "t2", "cut_err", "cyclegan_err"],
             ["00271_s084", "01648_s074", "01509_s117", "01163_s094"]),
    "fig18": ("final_four_models.png", ROWS6,
              ["t1", "cut", "cyclegan", "sd", "sil", "t2", "cut_err", "cyclegan_err", "sd_err", "sil_err"],
              ["01598_s027", "01143_s121", "01648_s074", "01163_s094"]),
    "fig2": ("alignment_check.png", ["01204", "01277", "00117", "00759", "01463"],
             ["t1", "t2", "outline", "checker"],
             ["01204", "01277"]),
}


def panels(path, n_rows, n_cols):
    """Bounding boxes (x0, y0, x1, y1) of the image panels, as an n_rows x n_cols list of lists."""
    img = np.asarray(Image.open(path).convert("RGB")).astype(int)
    labels, _ = ndimage.label(img.min(axis=2) < 235)
    boxes = [(s[1].start, s[0].start, s[1].stop, s[0].stop) for s in ndimage.find_objects(labels)]
    # image panels are large and roughly square; titles and text are small or flat
    big = [b for b in boxes if (b[2] - b[0]) > 100 and (b[3] - b[1]) > 100]
    big.sort(key=lambda b: (b[1], b[0]))
    if len(big) != n_rows * n_cols:
        raise SystemExit(f"{path}: found {len(big)} panels, expected {n_rows} x {n_cols}")
    grid = [sorted(big[r * n_cols:(r + 1) * n_cols]) for r in range(n_rows)]
    for row in grid:  # every panel in a row must share its top edge (within a pixel or two)
        if max(b[1] for b in row) - min(b[1] for b in row) > 3:
            raise SystemExit(f"{path}: panel rows are not aligned: {row}")
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "assets", "panels"))
    args = ap.parse_args()
    for name, (src, rows, cols, keep) in FIGURES.items():
        path = os.path.join(HERE, src)
        grid = panels(path, len(rows), len(cols))
        img = Image.open(path).convert("RGB")
        out = os.path.join(args.out, name)
        os.makedirs(out, exist_ok=True)
        for r, row_name in enumerate(rows):
            if row_name not in keep:
                continue
            for c, col_name in enumerate(cols):
                box = grid[r][c]
                img.crop(box).save(os.path.join(out, f"{row_name}_{col_name}.png"), optimize=True)
        sizes = sorted({(b[2] - b[0], b[3] - b[1]) for row in grid for b in row})
        print(f"{name}: {src} -> {len(keep)} rows x {len(cols)} panels in {out}; panel sizes {sizes}")


if __name__ == "__main__":
    main()
