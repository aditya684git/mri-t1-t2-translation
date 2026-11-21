#!/usr/bin/env python
"""
Local check of align_to_input.py on fake shifted outputs, no real data needed.

Fake slices: an ellipse "brain" with ventricles that are dark in the T1 and bright in the "T2" output, zero
background (plus faint noise in the outputs). The data folder holds ONLY testA, so the script cannot be
using testB or testMask. Runs, one per output layout evaluate.py reads:
  diffusion  <name>.png, every output moved by (-2, +8)        -> expect correction (+2, -8)
  cut        fake_B/<name>.png, not moved                      -> expect (0, 0)
  cyclegan   <stem>_fake_B.png saved as RGB, moved by (-1, +7) -> expect (+1, -7)
  jitter     <name>.png, a different random shift per slice    -> expect each slice's own correction,
                                                                  and --apply median uses the median
Checks the shifts found, the aligned images (equal to the unshifted output away from the borders),
the kept layouts and filenames, and that missing predictions are simply skipped.

Usage:  python diffusion/test_align.py
"""
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from align_to_input import shift  # noqa: E402

N = 12
INNER = (slice(20, -20), slice(20, -20))  # away from borders a shift uncovers


def fake_pair(rng):
    yy, xx = np.mgrid[:256, :256]
    cy, cx = 128 + rng.integers(-10, 11), 128 + rng.integers(-10, 11)
    ry, rx = rng.integers(70, 95), rng.integers(55, 75)
    brain = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
    vent = ((yy - cy + 10) / (ry * 0.3)) ** 2 + ((xx - cx + 6) / (rx * 0.18)) ** 2 <= 1
    texture = rng.normal(0, 8, (256, 256))
    t1 = np.where(brain, 150 + texture, 0)
    t1[vent] = 5                                                  # below the threshold: a hole in the T1 mask
    t2 = np.where(brain, 90 + texture, rng.uniform(0, 8, (256, 256)))  # faint background noise
    t2[vent] = 240                                                # bright fluid: no hole in the output
    return t1.clip(0, 255).astype(np.uint8), t2.clip(0, 255).astype(np.uint8)


def run(pred_dir, out, data, *extra):
    cmd = [sys.executable, os.path.join(HERE, "align_to_input.py"),
           "--pred_dir", pred_dir, "--out", out, "--data", data, *extra]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode:
        raise SystemExit(res.stdout + res.stderr)
    print(res.stdout.strip())
    return {r["file"]: r for r in csv.DictReader(open(os.path.join(out, "shifts.csv")))}


def main():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "brats_t1t2")
        os.makedirs(os.path.join(data, "testA"))
        names = [f"BraTS2021_{i:05d}_s{40 + 2 * i:03d}.png" for i in range(N)]
        outputs = {}
        for name in names:
            t1, t2 = fake_pair(rng)
            Image.fromarray(t1).save(os.path.join(data, "testA", name))
            outputs[name] = t2
        jitter = {name: (int(rng.integers(-5, 6)), int(rng.integers(-5, 6))) for name in names}

        layouts = {  # run: (move applied to the output, path of the output for a name, save as RGB)
            "diffusion": (lambda n: (-2, 8), lambda n: n, False),
            "cut": (lambda n: (0, 0), lambda n: os.path.join("fake_B", n), False),
            "cyclegan": (lambda n: (-1, 7), lambda n: os.path.splitext(n)[0] + "_fake_B.png", True),
            "jitter": (lambda n: jitter[n], lambda n: n, False),
        }
        for run_name, (move, rel, rgb) in layouts.items():
            pred_dir = os.path.join(tmp, run_name)
            for name in names[:-2] if run_name == "cut" else names:  # cut: 2 predictions missing
                img = Image.fromarray(shift(outputs[name], *move(name)))
                path = os.path.join(pred_dir, rel(name))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                (img.convert("RGB") if rgb else img).save(path)

            out = pred_dir + "_aligned"
            print(f"\n=== {run_name}")
            found = run(pred_dir, out, data)
            expected_names = names[:-2] if run_name == "cut" else names
            assert sorted(found) == sorted(expected_names), run_name
            for name in expected_names:
                dy, dx = move(name)
                r = found[name]
                assert (int(r["dy"]), int(r["dx"])) == (-dy, -dx), (run_name, name, r)
                aligned = np.asarray(Image.open(os.path.join(out, rel(name))))
                assert aligned.ndim == 2, "aligned outputs are saved as grayscale"
                assert np.array_equal(aligned[INNER], outputs[name][INNER]), (run_name, name)
            print(f"OK  {run_name}: every slice's shift recovered and undone, layout and filenames kept")

        # --apply median on the jittered run: every slice gets the run's median correction
        out = os.path.join(tmp, "jitter_median")
        print("\n=== jitter, --apply median")
        found = run(os.path.join(tmp, "jitter"), out, data, "--apply", "median")
        med = (int(np.round(np.median([-j[0] for j in jitter.values()]))),
               int(np.round(np.median([-j[1] for j in jitter.values()]))))
        assert all((int(r["applied_dy"]), int(r["applied_dx"])) == med for r in found.values())
        print(f"OK  --apply median applies {med} to every slice")

        # refuses to write into its own input folder
        res = subprocess.run([sys.executable, os.path.join(HERE, "align_to_input.py"), "--pred_dir",
                              os.path.join(tmp, "cut"), "--out", os.path.join(tmp, "cut"), "--data", data],
                             capture_output=True, text=True)
        assert res.returncode != 0 and "new folder" in res.stderr
        print("OK  --out equal to --pred_dir is refused")

    print("\nAll alignment checks passed.")


if __name__ == "__main__":
    main()
