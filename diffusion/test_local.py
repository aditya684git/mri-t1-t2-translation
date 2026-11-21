#!/usr/bin/env python
"""
Local check of make_mat.py on fake data, no GPU and no real data needed.

Builds ~20 random 256x256 grayscale PNGs per folder plus a fake split.json, runs make_mat.py, then:
  - compares the .mat layout with SynDiff_sample_data/T1.mat (dataset name, dtype, ndim, storage)
  - loads every split through SynDiff's own dataset.CreateDatasetSynthesis (unmodified)
  - checks each loaded row equals its source PNG pixel for pixel (so nothing is rotated or flipped),
    in the row order of test_files.json / manifest.json
  - checks val patients are test-split patients outside the eval subset, and train lengths are equal
  - checks infer_syndiff.to_uint8 turns a loaded row back into the exact original PNG

Usage:  python diffusion/test_local.py --syndiff ../SynDiff
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import h5py
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def fake_dataset(root, rng):
    def save(sub, name):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
        img = rng.integers(0, 256, (256, 256), dtype=np.uint8)
        img[:8], img[-8:], img[:, :8], img[:, -8:] = 0, 0, 0, 0  # zero padding, like ours
        img[20:60, 20:40] = 255                                   # asymmetric marker
        Image.fromarray(img).save(os.path.join(root, sub, name))

    pats = {"trainA": [f"BraTS2021_A{i}" for i in range(4)],
            "trainB": [f"BraTS2021_B{i}" for i in range(4)],
            "test": [f"BraTS2021_T{i:02d}" for i in range(10)]}
    for p in pats["trainA"]:
        for s in range(5):
            save("trainA", f"{p}_s{10 + 2 * s:03d}.png")
    for i, p in enumerate(pats["trainB"]):
        for s in range(5 if i else 4):  # 19 T2 slices vs 20 T1: unequal on purpose
            save("trainB", f"{p}_s{10 + 2 * s:03d}.png")
    for p in pats["test"]:
        for s in (40, 42):
            for sub in ("testA", "testB", "testMask"):
                save(sub, f"{p}_s{s:03d}.png")
    eval_patients = ["BraTS2021_T07", "BraTS2021_T02", "BraTS2021_T05"]  # not sorted: order must be kept
    eval_files = [f"{p}_s{s:03d}.png" for p in eval_patients for s in (42, 40)]
    json.dump({"patients": pats, "eval_subset": {"patients": eval_patients, "files": eval_files}},
              open(os.path.join(root, "split.json"), "w"))
    return pats, eval_patients, eval_files


def png_as_loaded(path):
    """What SynDiff's loader should return for this PNG: 0-1 float32, then (x - 0.5) / 0.5."""
    x = np.asarray(Image.open(path), dtype=np.float32) / 255.0
    return (x - 0.5) / 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syndiff", required=True, help="SynDiff clone (read only)")
    args = ap.parse_args()
    syndiff = os.path.abspath(args.syndiff)

    sys.path.insert(0, syndiff)
    from dataset import CreateDatasetSynthesis  # SynDiff's own loader
    spec = importlib.util.spec_from_file_location("infer_syndiff", os.path.join(HERE, "infer_syndiff.py"))
    infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infer)

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        src, out = os.path.join(tmp, "brats_t1t2"), os.path.join(tmp, "mat")
        pats, eval_patients, eval_files = fake_dataset(src, rng)
        make_mat = [sys.executable, os.path.join(HERE, "make_mat.py"), "--data", src, "--out", out]
        subprocess.run(make_mat, check=True)

        # 1. layout identical to SynDiff's sample data
        with h5py.File(os.path.join(syndiff, "SynDiff_sample_data", "T1.mat"), "r") as f:
            ref = f["data_fs"]
            ref_layout = (list(f.keys()), ref.dtype, ref.ndim, ref.chunks, ref.compression)
        for fname in sorted(os.listdir(out)):
            if fname.endswith(".mat"):
                with h5py.File(os.path.join(out, fname), "r") as f:
                    d = f["data_fs"]
                    layout = (list(f.keys()), d.dtype, d.ndim, d.chunks, d.compression)
                    assert layout == ref_layout, (fname, layout, ref_layout)
                    assert 0.0 <= d[()].min() and d[()].max() <= 1.0, fname
        print(f"OK  layout matches sample data: {ref_layout}")

        # 2. every split loads through SynDiff, with pixel-exact, upright rows in the recorded order
        names = json.load(open(os.path.join(out, "test_files.json")))
        manifest = json.load(open(os.path.join(out, "manifest.json")))
        assert names == eval_files, "test_files.json must keep split.json order"
        rows = manifest["rows"]
        for phase, t1_rows, t2_rows, t1_dir, t2_dir in (
                ("train", rows["train_T1"], rows["train_T2"], "trainA", "trainB"),
                ("val", rows["val"], rows["val"], "testA", "testB"),
                ("test", names, names, "testA", "testB")):
            ds = CreateDatasetSynthesis(phase, out, "T1", "T2")
            assert len(ds) == len(t1_rows) == len(t2_rows), phase
            for i, (x, y) in enumerate(ds):
                assert tuple(x.shape) == (1, 256, 256), (phase, x.shape)
                assert np.array_equal(x[0].numpy(), png_as_loaded(os.path.join(src, t1_dir, t1_rows[i]))), (phase, i)
                assert np.array_equal(y[0].numpy(), png_as_loaded(os.path.join(src, t2_dir, t2_rows[i]))), (phase, i)
            print(f"OK  {phase}: {len(ds)} rows load through SynDiff and equal their PNGs (upright, in order)")

        # 3. split rules
        assert len(rows["train_T1"]) == len(rows["train_T2"]) == 19
        val_patients = set(manifest["val_patients"])
        assert len(val_patients) == 5 and val_patients <= set(pats["test"]) and not val_patients & set(eval_patients)
        assert {f.rsplit("_s", 1)[0] for f in rows["val"]} == val_patients
        print(f"OK  train 19 = 19 (trainB had 19, trainA 20); val patients {sorted(val_patients)}")

        # 4. inference output mapping: [-1, 1] -> exact original PNG
        x, _ = CreateDatasetSynthesis("test", out, "T1", "T2")[0]
        back = infer.to_uint8(x[0].numpy())
        assert np.array_equal(back, np.asarray(Image.open(os.path.join(src, "testA", names[0])))), "to_uint8"
        print("OK  to_uint8 inverts the loader exactly")

        # 5. --n_train, and refusing to overwrite
        out2 = os.path.join(tmp, "mat10")
        subprocess.run(make_mat[:-1] + [out2, "--n_train", "10"], check=True, stdout=subprocess.DEVNULL)
        assert len(CreateDatasetSynthesis("train", out2, "T1", "T2")) == 10
        again = subprocess.run(make_mat, capture_output=True, text=True)
        assert again.returncode != 0 and "--overwrite" in again.stderr
        print("OK  --n_train 10 gives 10 rows; existing output is not overwritten without --overwrite")

    print("\nAll local checks passed.")


if __name__ == "__main__":
    main()
