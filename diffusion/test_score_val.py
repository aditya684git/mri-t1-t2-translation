#!/usr/bin/env python
"""
Local check of score_val.py on fake validation slices, no real data needed (needs scikit-image).

Scoring, end to end: a fake dataset (testA, testB, testMask) and a fake make_mat.py folder whose
manifest.json lists 8 "validation" slices. The real T2 serves as the output of two runs:
  syndiff_n500_e30        every output moved by (-2, +8) -> shift (+2, -8), raw SSIM < 1, aligned SSIM 1
  sc_n500_mind_0.25_e30   outputs unchanged              -> shift (0, 0), raw SSIM 1
Checks the per-slice CSV, the rows of val_results.csv, that rerunning a run replaces its row, and that
--select excludes the shifted baseline and selects the unshifted run (raw 1 >= the baseline's aligned 1).

--select on hand-written tables:
  - an unshifted but undertrained baseline epoch (syndiff_n500_e5, raw SSIM -0.025) fails the quality floor
    (the best aligned SSIM of syndiff_n500, 0.290 at e30); epoch 0 never counts; a size with no baseline
    rows (n2000) has no floor, so its runs do not count; near-tied SSIMs go to the lower MAE
  - when nothing passes: "No candidate passes", no selection, and the best row (epoch >= 1) of each run

Usage:  python diffusion/test_score_val.py
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from align_to_input import shift  # noqa: E402
from score_val import FIELDS  # noqa: E402
from test_align import fake_pair  # noqa: E402

SCRIPT = os.path.join(HERE, "score_val.py")
BASE, SC = "syndiff_n500_e30", "sc_n500_mind_0.25_e30"


def run(*args, quiet=False):
    res = subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)
    if res.returncode:
        raise SystemExit(res.stdout + res.stderr)
    if not quiet:
        print(res.stdout.strip())
    return res.stdout


def write_table(folder, rows):
    """rows: (run, ssim, mae, dy, dx, ssim_aligned)"""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "val_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for run_name, ssim, mae, dy, dx, aligned in rows:
            w.writerow({"run": run_name, "n_slices": 300, "n_missing": 0, "ssim": ssim, "ssim_std": 0.05,
                        "psnr": 15.0, "mae": mae, "shift_dy": dy, "shift_dx": dx,
                        "within_1px": 1.0 if abs(dy) <= 1 and abs(dx) <= 1 else 0.0, "ssim_aligned": aligned})


def check_scoring(tmp):
    rng = np.random.default_rng(1)
    data, syn, out = (os.path.join(tmp, d) for d in ("brats", "syndata", "val"))
    for sub in ("testA", "testB", "testMask"):
        os.makedirs(os.path.join(data, sub))
    os.makedirs(syn)
    names = [f"BraTS2021_{900 + i // 4:05d}_s{60 + i:03d}.png" for i in range(8)]
    t2s = {}
    for name in names:
        t1, t2 = fake_pair(rng)
        Image.fromarray(t1).save(os.path.join(data, "testA", name))
        Image.fromarray(t2).save(os.path.join(data, "testB", name))
        Image.fromarray(((t1 > 0) * 255).astype(np.uint8)).save(os.path.join(data, "testMask", name))
        t2s[name] = t2
    json.dump({"rows": {"val": names}}, open(os.path.join(syn, "manifest.json"), "w"))

    for run_name, move in ((BASE, (-2, 8)), (SC, (0, 0))):
        pred = os.path.join(tmp, run_name)
        os.makedirs(pred)
        for name in names:
            Image.fromarray(shift(t2s[name], *move)).save(os.path.join(pred, name))
        print(f"\n=== {run_name}")
        run("--run", run_name, "--pred_dir", pred, "--syndata", syn, "--data", data, "--out_dir", out)

    rows = {r["run"]: r for r in csv.DictReader(open(os.path.join(out, "val_results.csv")))}
    s, a = rows[BASE], rows[SC]
    assert (int(s["shift_dy"]), int(s["shift_dx"])) == (2, -8) and float(s["within_1px"]) == 0, s
    assert float(s["ssim"]) < 0.9 and float(s["ssim_aligned"]) > 0.999, s
    assert (int(a["shift_dy"]), int(a["shift_dx"])) == (0, 0) and float(a["within_1px"]) == 1, a
    assert float(a["ssim"]) > 0.999 and float(a["mae"]) == 0, a
    assert int(s["n_slices"]) == int(a["n_slices"]) == len(names)
    per = list(csv.DictReader(open(os.path.join(out, f"{BASE}_val_per_slice.csv"))))
    assert [r["file"] for r in per] == names and all((r["dy"], r["dx"]) == ("2", "-8") for r in per)
    print(f"\nOK  shifted run: input-based shift (+2, -8) on every slice, raw SSIM {float(s['ssim']):.3f}, "
          f"aligned {float(s['ssim_aligned']):.3f}")
    print(f"OK  unshifted run: shift (0, 0), raw SSIM {float(a['ssim']):.3f}")

    run("--run", SC, "--pred_dir", os.path.join(tmp, SC), "--syndata", syn, "--data", data, "--out_dir", out,
        quiet=True)
    assert [r["run"] for r in csv.DictReader(open(os.path.join(out, "val_results.csv")))] == [BASE, SC]
    print("OK  rerunning a run replaces its row")

    print("\n=== --select")
    text = run("--select", "--out_dir", out)
    assert f"Selected by the validation rule: {SC}" in text
    assert f"{BASE}" in text and "excluded: median shift > 1 px" in text
    print("OK  --select excludes the shifted baseline and selects the unshifted run")


def check_select(tmp):
    folder = os.path.join(tmp, "select")
    rows = [  # run, raw SSIM, MAE, dy, dx, aligned SSIM
        ("syndiff_n500_e0", 0.500, 0.08, 0, 0, 0.50),     # epoch 0: never a candidate
        ("syndiff_n500_e5", -0.025, 0.30, 0, 0, -0.02),   # unshifted but undertrained: below the floor
        ("syndiff_n500_e30", 0.130, 0.17, 1, -7, 0.290),  # shifted; its aligned SSIM sets the n500 floor
        ("sc_n500_mind_0.25_e10", 0.270, 0.15, 0, 0, 0.27),   # unshifted, below the floor
        ("sc_n500_mind_0.25_e30", 0.310, 0.10, 0, 1, 0.31),   # passes
        ("sc_n500_mind_1_e30", 0.312, 0.09, 0, 0, 0.312),     # passes; tied with the one above, lower MAE
        ("sc_n2000_mind_0.25_e30", 0.900, 0.05, 0, 0, 0.90),  # no syndiff_n2000 rows: no floor
    ]
    write_table(folder, rows)
    print("\n=== --select, hand-written table")
    text = run("--select", "--out_dir", folder)
    assert "quality floor for n500: raw SSIM >= 0.2900" in text and "(syndiff_n500_e30)" in text
    assert "Selected by the validation rule: sc_n500_mind_1_e30" in text
    lines = {l.split()[0]: l for l in text.splitlines() if l.startswith(("syndiff_", "sc_"))}
    assert "epoch 0" in lines["syndiff_n500_e0"]
    assert "below the floor 0.2900" in lines["syndiff_n500_e5"]
    assert "below the floor" in lines["sc_n500_mind_0.25_e10"]
    assert "no syndiff_n2000 baseline" in lines["sc_n2000_mind_0.25_e30"]
    assert "excluded" not in lines["sc_n500_mind_0.25_e30"] and "excluded" not in lines["sc_n500_mind_1_e30"]
    print("OK  undertrained unshifted baseline e5 fails the floor 0.290; epoch 0 and a size without a baseline "
          "never count; the near-tie goes to the lower MAE")

    write_table(folder, [r for r in rows if r[0] not in ("sc_n500_mind_0.25_e30", "sc_n500_mind_1_e30")])
    print("\n=== --select, nothing passes")
    text = run("--select", "--out_dir", folder)
    assert "No candidate passes" in text and "Selected" not in text
    best = text.split("Best row of each run")[1].splitlines()[2:]
    best_runs = [l.split()[0] for l in best if l.strip()]
    assert sorted(best_runs) == ["sc_n2000_mind_0.25_e30", "sc_n500_mind_0.25_e10", "syndiff_n500_e30"], best_runs
    print("OK  nothing passes: no selection, and the best row (epoch >= 1) of each run is listed")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        check_scoring(tmp)
        check_select(tmp)
    print("\nAll validation-scoring checks passed.")


if __name__ == "__main__":
    main()
