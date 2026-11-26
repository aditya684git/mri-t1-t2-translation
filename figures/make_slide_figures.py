#!/usr/bin/env python
"""
Presentation figures, all read from the repo's result files (no numbers typed in):

  main_results       brain SSIM / PSNR / MAE for the five headline models, 95% CI over the test patients
  shift_heatmaps     SSIM over (dy, dx) for the original and fixed T1 -> T2 diffusive generator
  generator_shifts   best shift of all four generators, original vs fixed
  shift_drift_n500   median input-based shift and raw val SSIM vs epoch, n500 baseline vs fixed
  data_size          eval SSIM vs number of training slices, original aligned vs fixed, CycleGAN reference
  per_patient        patient-mean SSIM, CycleGAN vs fixed SynDiff, one line per patient, Wilcoxon test
  loss_sweep         best raw val SSIM of each n500 run, baseline's aligned score as a reference

Inputs: results/results.csv, results/*_eval_per_slice.csv, results/shift_cause_n7514_e30.txt,
results/shift_cause_sc_n7514_e30.txt, results/syndiff_val/val_results.csv.
Outputs: figures/slides/<name>.png (300 dpi) and .svg.

Usage (from the repo root):  python figures/make_slide_figures.py
"""
import argparse
import csv
import os
import re
from decimal import Decimal, ROUND_HALF_UP

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# One colour per model in every chart: Okabe-Ito, colour-blind safe
OKABE = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2",
             vermillion="#D55E00", purple="#CC79A7", grey="#8C8C8C", black="#000000")
MODELS = [  # run in results.csv, label, colour
    ("cut_v1", "CUT", OKABE["orange"]),
    ("cyclegan_v1", "CycleGAN", OKABE["blue"]),
    ("diffusion_v1", "SynDiff original", OKABE["vermillion"]),
    ("diffusion_v1_aligned", "SynDiff original, aligned", OKABE["purple"]),
    ("diffusion_sc_v1", "SynDiff + silhouette loss", OKABE["green"]),
]
LABEL = {run: label for run, label, _ in MODELS}
COLOUR = {run: c for run, _, c in MODELS}
ORIGINAL, ALIGNED, FIXED, CYCLEGAN = (COLOUR["diffusion_v1"], COLOUR["diffusion_v1_aligned"],
                                      COLOUR["diffusion_sc_v1"], COLOUR["cyclegan_v1"])
METRICS = [("ssim", "Brain SSIM", "↑ better"), ("psnr", "Brain PSNR (dB)", "↑ better"), ("mae", "Brain MAE", "↓ better")]

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16,
    "legend.fontsize": 16, "figure.titlesize": 22, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "svg.fonttype": "none", "axes.titleweight": "bold",
})


# ---------------------------------------------------------------- reading the result files
def read_csv(path):
    return list(csv.DictReader(open(path)))


def results_table():
    return {r["run"]: r for r in read_csv(os.path.join(RESULTS, "results.csv")) if r["subset"] == "eval"}


def patient_means(run, metric):
    """{patient: mean of the metric over that patient's eval slices}"""
    per = {}
    for r in read_csv(os.path.join(RESULTS, f"{run}_eval_per_slice.csv")):
        per.setdefault(r["patient"], []).append(float(r[metric]))
    return {p: float(np.mean(v)) for p, v in per.items()}


def ci95(values):
    v = np.asarray(values, float)
    return stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def parse_shift_file(path):
    """shift_check.py output -> [{name, what, target, grid (dy x dx), radius, n_slices, best, ssim_best, ssim_0}]"""
    blocks, cur = [], None
    for line in open(path):
        m = re.match(r"(\S+) \((.+?)\) vs real (T\d): brain SSIM by shift .*?(-?\d+)\.\.(\d+)\), (\d+) slices", line)
        if m:
            cur = {"name": m.group(1), "what": m.group(2), "target": m.group(3), "radius": int(m.group(5)),
                   "n_slices": int(m.group(6)), "rows": []}
            blocks.append(cur)
            continue
        m = re.match(r"\s+dy=([+-]?\d+)\s+(.*)", line)
        if m and cur is not None:
            cur["rows"].append([float(x) for x in m.group(2).split()])
            continue
        m = re.match(r"\s+best shift \(dy, dx\) = \((-?\d+), (-?\d+)\)\s+SSIM ([\d.]+)\s+vs unshifted ([\d.]+)", line)
        if m and cur is not None:
            cur.update(best=(int(m.group(1)), int(m.group(2))), ssim_best=float(m.group(3)), ssim_0=float(m.group(4)))
            cur["grid"] = np.array(cur.pop("rows"))
    return blocks


def val_rows():
    """{exp: {epoch: row}} from val_results.csv"""
    out = {}
    for r in read_csv(os.path.join(RESULTS, "syndiff_val", "val_results.csv")):
        m = re.fullmatch(r"(.+)_e(\d+)", r["run"])
        if m:
            out.setdefault(m.group(1), {})[int(m.group(2))] = r
    return out


def save(fig, out_dir, name):
    for ext, kw in (("png", {"dpi": 300}), ("svg", {})):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  {name}.png / .svg")


def fmt_shift(s):
    return f"({s[0]:+d}, {s[1]:+d})".replace("+0", "0").replace("-", "−")


# ---------------------------------------------------------------- figures
def main_results(out_dir):
    table = results_table()
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4))
    for ax, (metric, title, better) in zip(axes, METRICS):
        heights, errs, labels = [], [], []
        dec = 2 if metric == "psnr" else 3
        for run, _, _ in MODELS:
            pm = patient_means(run, metric)
            text = table[run][f"{metric}_mean"]
            h = float(text)
            # round the CSV's decimal text half-up (0.1455 -> 0.146), as in the README tables;
            # formatting the float would give 0.145 because 0.1455 is stored as 0.14549...
            labels.append(str(Decimal(text).quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP)))
            # every patient has the same number of eval slices, so the slice mean equals the patient mean
            assert abs(h - np.mean(list(pm.values()))) < 2e-3, (run, metric)
            heights.append(h)
            errs.append(ci95(list(pm.values())))
        x = np.arange(len(MODELS))
        ax.bar(x, heights, yerr=errs, color=[c for *_, c in MODELS], width=0.75, capsize=6,
               error_kw={"elinewidth": 1.8, "ecolor": "#333333"})
        for xi, h, e, label in zip(x, heights, errs, labels):
            ax.text(xi, h + e + 0.015 * (ax.get_ylim()[1]), label, ha="center", va="bottom", fontsize=14)
        ax.set_title(title, pad=34)
        ax.text(0.5, 1.02, better, transform=ax.transAxes, ha="center", va="bottom", fontsize=17, color="#333333")
        ax.set_xticks([])
        top = max(h + e for h, e in zip(heights, errs))
        ax.set_ylim(0, top * 1.15)
    handles = [Rectangle((0, 0), 1, 1, color=c) for *_, c in MODELS]
    fig.legend(handles, [l for _, l, _ in MODELS], loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.04), handlelength=1.2, columnspacing=1.4)
    n = len(patient_means(MODELS[0][0], "ssim"))
    fig.suptitle(f"200 eval slices, {n} test patients; error bars: 95% CI over patient means", y=1.0,
                 fontsize=18, fontweight="normal")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    save(fig, out_dir, "main_results")


def shift_heatmaps(out_dir):
    orig = parse_shift_file(os.path.join(RESULTS, "shift_cause_n7514_e30.txt"))
    fixed = parse_shift_file(os.path.join(RESULTS, "shift_cause_sc_n7514_e30.txt"))
    pick = lambda blocks: next(b for b in blocks if b["name"] == "diffusive_dir2")
    panels = [(pick(orig), "SynDiff original", ORIGINAL), (pick(fixed), "SynDiff + silhouette loss", FIXED)]
    vmin = min(b["grid"].min() for b, _, _ in panels)
    vmax = max(b["grid"].max() for b, _, _ in panels)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    for ax, (b, title, colour) in zip(axes, panels):
        r = b["radius"]
        im = ax.imshow(b["grid"], cmap="viridis", vmin=vmin, vmax=vmax, extent=(-r - .5, r + .5, r + .5, -r - .5))
        ax.plot(0, 0, marker="+", color="white", markersize=26, markeredgewidth=3.5)
        dy, dx = b["best"]
        ax.plot(dx, dy, marker="o", markerfacecolor="none", markeredgecolor="#F0E442", markersize=24, markeredgewidth=3.5)
        ax.set_title(title, color=colour)
        ax.set_xlabel("dx (px, + = right)")
        ax.set_ylabel("dy (px, + = down)")
        ax.set_xticks(range(-r, r + 1, 5))
        ax.set_yticks(range(-r, r + 1, 5))
        for s in ax.spines.values():
            s.set_visible(False)
        note = (f"peak {fmt_shift(b['best'])}: SSIM {b['ssim_best']:.3f}\n"
                f"centre (0, 0): SSIM {b['ssim_0']:.3f}")
        ax.text(0.5, -0.2, note, transform=ax.transAxes, ha="center", va="top", fontsize=16)
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.03)
    cbar.set_label("brain SSIM vs real T2")
    fig.suptitle(f"T1 → T2 diffusive generator: SSIM when the output is moved by (dy, dx)\n"
                 f"○ peak    + centre (no move)    same colour scale in both panels",
                 fontsize=18, fontweight="normal", y=1.04)
    save(fig, out_dir, "shift_heatmaps")


def generator_shifts(out_dir):
    orig = {b["name"]: b for b in parse_shift_file(os.path.join(RESULTS, "shift_cause_n7514_e30.txt"))}
    fixed = {b["name"]: b for b in parse_shift_file(os.path.join(RESULTS, "shift_cause_sc_n7514_e30.txt"))}
    order = ["diffusive_dir2", "non_diffusive_dir2", "diffusive_dir1", "non_diffusive_dir1"]
    names = {"diffusive_dir2": "diffusive", "non_diffusive_dir2": "translation (cycle)",
             "diffusive_dir1": "diffusive", "non_diffusive_dir1": "translation (cycle)"}
    fig, ax = plt.subplots(figsize=(14, 6.2))
    ax.set_xlim(0, 3)
    ax.set_ylim(len(order) + 0.9, 0)
    ax.axis("off")
    ax.text(0.05, 0.45, "generator", fontsize=18, fontweight="bold", va="center")
    for col, (label, colour) in enumerate((("SynDiff original", ORIGINAL), ("SynDiff + silhouette loss", FIXED)), 1):
        ax.text(col + 0.5, 0.45, label, fontsize=18, fontweight="bold", ha="center", va="center", color=colour)
    cells = []  # (text artist, box) pairs, shrunk to fit below
    for i, key in enumerate(order, 1):
        o, f = orig[key], fixed[key]
        ax.text(0.05, i + 0.12, names[key], fontsize=18, va="center")
        ax.text(0.05, i + 0.48, f"{o['what'].replace('->', ' → ')}, vs real {o['target']}", fontsize=15,
                color="#555555", va="center")
        for col, (b, colour) in enumerate(((o, ORIGINAL), (f, FIXED)), 1):
            box = Rectangle((col + 0.04, i - 0.04), 0.92, 0.88, facecolor=colour, alpha=0.13,
                            edgecolor=colour, linewidth=2)
            ax.add_patch(box)
            t1 = ax.text(col + 0.5, i + 0.22, fmt_shift(b["best"]), fontsize=24, fontweight="bold", ha="center",
                         va="center", color=colour)
            sub = (f"SSIM {b['ssim_0']:.3f} → {b['ssim_best']:.3f} if moved back" if b["best"] != (0, 0)
                   else f"SSIM {b['ssim_0']:.3f} as output (best)")
            t2 = ax.text(col + 0.5, i + 0.6, sub, fontsize=14, ha="center", va="center", color="#333333")
            cells += [(t1, box), (t2, box)]
    n = orig[order[0]]["n_slices"]
    ax.set_title(f"Best alignment shift (dy, dx) of each generator, full data, epoch 30 ({n} eval slices)",
                 fontsize=19, pad=14)
    fit_text_in_boxes(fig, cells)
    save(fig, out_dir, "generator_shifts")


def fit_text_in_boxes(fig, cells, pad=0.08):
    """Shrink each text until it fits inside its box with `pad` (fraction of the box width) on each side."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    start = {id(t): t.get_fontsize() for t, _ in cells}
    for text, box in cells:
        limit = box.get_window_extent(renderer).width * (1 - 2 * pad)
        while text.get_window_extent(renderer).width > limit and text.get_fontsize() > 8:
            text.set_fontsize(text.get_fontsize() - 0.5)
    # texts that started at the same size end at the same size, so every cell looks alike
    for size in set(start.values()):
        group = [t for t, _ in cells if start[id(t)] == size]
        smallest = min(t.get_fontsize() for t in group)
        for t in group:
            t.set_fontsize(smallest)


def shift_drift_n500(out_dir):
    v = val_rows()
    runs = [("syndiff_n500", "SynDiff original", ORIGINAL), ("sc_n500_silhouette_0.5", "SynDiff + silhouette loss", FIXED)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
    for exp, label, colour in runs:
        ep = sorted(e for e in v[exp] if e >= 1)  # epoch 0: untrained, its shift search hits the limit
        a1.plot(ep, [int(v[exp][e]["shift_dx"]) for e in ep], "-o", color=colour, lw=3, ms=9, label=label)
        a2.plot(ep, [float(v[exp][e]["ssim"]) for e in ep], "-o", color=colour, lw=3, ms=9, label=label + ", raw")
    ep = sorted(e for e in v["syndiff_n500"] if e >= 1)
    a2.plot(ep, [float(v["syndiff_n500"][e]["ssim_aligned"]) for e in ep], "--s", color=ALIGNED, lw=2.5, ms=8,
            label="SynDiff original, aligned")
    a1.axhline(0, color="#999999", lw=1)
    a1.set_title("Median shift found from the input (dx)")
    a1.set_xlabel("epoch")
    a1.set_ylabel("dx (px)")
    a1.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    a2.axhline(0, color="#999999", lw=1)
    a2.set_title("Validation SSIM ↑")
    a2.set_xlabel("epoch")
    a2.set_ylabel("brain SSIM")
    for a in (a1, a2):
        a.set_xticks(ep)
    # one shared legend below both panels, so it covers no data
    handles, labels = a2.get_legend_handles_labels()
    labels = ["SynDiff original (raw output)", "SynDiff + silhouette loss (raw output)",
              "SynDiff original, aligned (right panel only)"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=14)
    n = v["syndiff_n500"][ep[0]]["n_slices"]
    fig.suptitle(f"500 training slices: the original drifts off-centre, the fix stays at 0 ({n} validation slices)",
                 fontsize=18, fontweight="normal")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, out_dir, "shift_drift_n500")


def data_size(out_dir):
    table = results_table()
    data = ["n500", "n2000", "n7514"]  # make_mat.py data names; the number is the training slices per contrast
    series = [("SynDiff original, aligned", ALIGNED, ["diffusion_n500_aligned", "diffusion_n2000_aligned", "diffusion_v1_aligned"]),
              ("SynDiff + silhouette loss", FIXED, ["diffusion_sc_n500", "diffusion_sc_n2000", "diffusion_sc_v1"])]
    x = [int(d[1:]) for d in data]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for label, colour, runs in series:
        y = [float(table[r]["ssim_mean"]) for r in runs]
        ax.plot(x, y, "-o", color=colour, lw=3.5, ms=12, label=label)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 12 if colour == FIXED else -26),
                        ha="center", fontsize=15, color=colour)
    cg = float(table["cyclegan_v1"]["ssim_mean"])
    ax.axhline(cg, color=CYCLEGAN, ls="--", lw=2.5)
    ax.text(x[0], cg + 0.008, f"CycleGAN (all data): {cg:.3f}", color=CYCLEGAN, fontsize=15, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:,}" for s in x])
    ax.minorticks_off()
    ax.set_xlim(x[0] / 1.4, x[-1] * 1.4)
    ax.set_xlabel("training slices per contrast")
    ax.set_ylabel("eval brain SSIM ↑")
    ax.set_title("More data helps, and the fix helps at every size")
    ax.legend(frameon=False, loc="lower right")
    lo = min(float(table[r]["ssim_mean"]) for _, _, runs in series for r in runs)
    ax.set_ylim(lo - 0.06, max(cg, *(float(table[r]["ssim_mean"]) for _, _, runs in series for r in runs)) + 0.05)
    save(fig, out_dir, "data_size")


def per_patient(out_dir):
    a, b = patient_means("cyclegan_v1", "ssim"), patient_means("diffusion_sc_v1", "ssim")
    pats = sorted(a)
    x, y = np.array([a[p] for p in pats]), np.array([b[p] for p in pats])
    up = int((y > x).sum())
    p = stats.wilcoxon(y, x).pvalue
    fig, ax = plt.subplots(figsize=(9, 7.5))
    for xi, yi in zip(x, y):
        c, ls = (FIXED, "-") if yi > xi else (OKABE["vermillion"], "--")
        ax.plot([0, 1], [xi, yi], ls, color=c, lw=2.2, marker="o", ms=8, alpha=0.9)
    ax.plot([0, 1], [x.mean(), y.mean()], color="black", lw=4, marker="D", ms=11, zorder=5)
    ax.text(1.06, y.mean(), f"mean\n{y.mean():.3f}", va="center", fontsize=15)
    ax.text(-0.06, x.mean(), f"mean\n{x.mean():.3f}", va="center", ha="right", fontsize=15)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CycleGAN", "SynDiff +\nsilhouette loss"])
    for t, c in zip(ax.get_xticklabels(), (CYCLEGAN, FIXED)):
        t.set_color(c)
        t.set_fontweight("bold")
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("patient-mean brain SSIM ↑")
    ax.plot([], [], "-", color=FIXED, lw=2.2, label="improved")
    ax.plot([], [], "--", color=OKABE["vermillion"], lw=2.2, label="worse")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    ptxt = f"{p:.5f}" if p >= 1e-4 else f"{p:.1e}"
    ax.set_title(f"{up}/{len(pats)} patients improved, Wilcoxon p = {ptxt}")
    save(fig, out_dir, "per_patient")
    return up, len(pats), p


def loss_sweep(out_dir):
    v = val_rows()
    runs = [("syndiff_n500", "baseline", ORIGINAL), ("sc_n500_silhouette_0.5", "silhouette\nλ = 0.5", FIXED),
            ("sc_n500_silhouette_2", "silhouette\nλ = 2", OKABE["grey"]), ("sc_n500_mind_0.25", "MIND\nλ = 0.25", OKABE["grey"]),
            ("sc_n500_mind_1", "MIND\nλ = 1", OKABE["grey"])]
    best, epochs = [], []
    for exp, _, _ in runs:
        rows = {e: r for e, r in v[exp].items() if e >= 1}
        e = max(rows, key=lambda k: float(rows[k]["ssim"]))
        best.append(float(rows[e]["ssim"]))
        epochs.append(e)
    floor_ep = max((e for e in v["syndiff_n500"] if e >= 1), key=lambda k: float(v["syndiff_n500"][k]["ssim_aligned"]))
    floor = float(v["syndiff_n500"][floor_ep]["ssim_aligned"])
    fig, ax = plt.subplots(figsize=(13, 6.5))
    xs = np.arange(len(runs))
    ax.bar(xs, best, color=[c for *_, c in runs], width=0.65)
    for xi, b, e in zip(xs, best, epochs):
        ax.text(xi, b + (0.012 if b >= 0 else -0.012), f"{b:.3f}".replace("-", "−") + f"\n(epoch {e})", ha="center",
                va="bottom" if b >= 0 else "top", fontsize=15)
    ax.axhline(0, color="#333333", lw=1.2)
    ax.axhline(floor, color=ALIGNED, ls="--", lw=2.5)
    ax.text(len(runs) - 0.55, floor + 0.012, f"baseline after alignment: {floor:.3f}", color=ALIGNED, ha="right",
            va="bottom", fontsize=15)
    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l, _ in runs])
    ax.set_ylabel("best raw validation SSIM ↑")
    ax.set_ylim(min(best) - 0.09, max(best + [floor]) + 0.1)
    ax.set_title("Loss sweep on 500 training slices\n"
                 "silhouette λ = 0.5 wins; MIND removes the shift but breaks the translation")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    save(fig, out_dir, "loss_sweep")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "slides"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"writing to {args.out}")
    main_results(args.out)
    shift_heatmaps(args.out)
    generator_shifts(args.out)
    shift_drift_n500(args.out)
    data_size(args.out)
    up, n, p = per_patient(args.out)
    loss_sweep(args.out)
    print(f"per-patient: {up}/{n} improved, Wilcoxon p = {p:.5f}")


if __name__ == "__main__":
    main()
