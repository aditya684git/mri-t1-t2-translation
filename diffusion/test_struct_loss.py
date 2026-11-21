#!/usr/bin/env python
"""
Local check (CPU, no GPU, no data) of struct_loss.py and syndiff_train_sc.py.

Fake slices: an elliptical brain with white matter, grey matter and ventricles, zero background, with
T1-like and T2-like contrasts (the same anatomy, different tissue intensities, as in our data). On SynDiff's
-1..1 scale, for both losses:
  - an aligned output gives the loss's minimum: exactly 0 (mind) or the sigmoid-tail floor (silhouette)
    when the output is the source; also exactly 0 (mind) for a contrast-inverted copy
  - for a T2-like output of a T1 source: the loss is smallest at shift 0, grows with every step of a shift
    along either axis (0, 1, 2, 4, 8 px), and the same holds when the source is a generated image
  - gradients reach a shifted output and are finite
For --struct_loss none:
  - syndiff_train_sc.py minus its '# [sc]' lines is ../SynDiff/train.py byte for byte (line endings aside)
  - inside train_syndiff, every use of the added code is under `if args.struct_loss != 'none':`, and the
    default is 'none', so with none the training step runs exactly train.py's operations
  - run_syndiff_train.apply_patches (used by both training scripts) still applies all three patches

Usage:  python diffusion/test_struct_loss.py --syndiff ../SynDiff
"""
import argparse
import ast
import os
import sys

import numpy as np
import torch
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import struct_loss  # noqa: E402
from align_to_input import shift  # noqa: E402

SHIFTS = [0, 1, 2, 4, 8]


def fake_pair(seed=0):
    """(T1, T2) on 0-1, 256 x 256: same anatomy, different contrast, background exactly 0."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:256, :256]
    brain = ((yy - 128) / 95) ** 2 + ((xx - 125) / 75) ** 2 <= 1
    field = ndimage.gaussian_filter(rng.normal(size=(256, 256)), 6)
    wm = brain & (field > 0.002) & (((yy - 128) / 80) ** 2 + ((xx - 125) / 60) ** 2 <= 1)
    vent = ((yy - 118) / 25) ** 2 + ((xx - 118) / 8) ** 2 <= 1
    vent |= ((yy - 118) / 25) ** 2 + ((xx - 134) / 8) ** 2 <= 1
    t1, t2 = np.zeros((256, 256)), np.zeros((256, 256))
    t1[brain], t2[brain] = 0.55, 0.50   # grey matter
    t1[wm], t2[wm] = 0.80, 0.35         # white matter
    t1[vent], t2[vent] = 0.10, 0.95     # fluid
    tex = ndimage.gaussian_filter(rng.normal(0, 0.03, (256, 256)), 1)
    t1 = np.where(brain, ndimage.gaussian_filter(t1, 0.7) + tex, 0).clip(0, 1)
    t2 = np.where(brain, ndimage.gaussian_filter(t2, 0.7) + tex, 0).clip(0, 1)
    return t1, t2


def tensor(img01):
    return torch.tensor(img01 * 2 - 1, dtype=torch.float32)[None, None]  # SynDiff's -1..1 scale


def moved(img01, dy, dx):
    return tensor(shift(img01, dy, dx))


def check_losses():
    t1, t2 = fake_pair()
    src = tensor(t1)
    for kind, fn in struct_loss.LOSSES.items():
        same = fn(src, src, True).item()
        if kind == "mind":
            inverted = fn(-src, src, True).item()  # 1 - x on the 0-1 scale: MIND does not change
            assert same < 1e-6 and inverted < 1e-5, (same, inverted)
            print(f"OK  {kind}: output = source -> {same:.1e}; contrast-inverted source -> {inverted:.1e}")
        else:
            assert same < 0.01, same
            print(f"OK  {kind}: output = source -> {same:.4f} (the sigmoid tails; its minimum)")

        for src_real, source, label in ((True, src, "real T1 source"),
                                        (False, tensor(struct_loss.to01(src).numpy()[0, 0] * 0.98 + 0.004),
                                         "generated source (background not exactly 0)")):
            rows = {}
            for axis in ("dy", "dx"):
                rows[axis] = [fn(moved(t2, s if axis == "dy" else 0, s if axis == "dx" else 0), source,
                                 src_real).item() for s in SHIFTS]
                rows[axis + "-"] = [fn(moved(t2, -s if axis == "dy" else 0, -s if axis == "dx" else 0), source,
                                       src_real).item() for s in SHIFTS]
            for name, vals in rows.items():
                assert all(a < b for a, b in zip(vals, vals[1:])), (kind, label, name, vals)
            print(f"OK  {kind}, T2-like output vs {label}: grows with every shift step "
                  f"(0,1,2,4,8 px along dy: {', '.join(f'{v:.4f}' for v in rows['dy'])})")

        out = moved(t2, 3, -5).requires_grad_(True)
        fn(out, src, True).backward()
        g = out.grad
        assert torch.isfinite(g).all() and g.abs().sum() > 0
        print(f"OK  {kind}: finite, non-zero gradient on a shifted output ({int((g != 0).sum())} pixels)")

    # terms(): one loss per named pair, as syndiff_train_sc.py calls it
    pairs = {"a": (moved(t2, 0, 0), src, True), "b": (moved(t2, 0, 8), src, True)}
    for kind in struct_loss.LOSSES:
        t = struct_loss.terms(kind, pairs)
        assert list(t) == ["a", "b"] and t["a"] < t["b"]
    print("OK  terms(): one value per named pair")


class Ancestors(ast.NodeVisitor):
    def __init__(self):
        self.parents = {}

    def generic_visit(self, node):
        for child in ast.iter_child_nodes(node):
            self.parents[child] = node
        super().generic_visit(node)


def guarded(node, parents):
    """Is node inside `if args.struct_loss != 'none':`?"""
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If) and ast.unparse(node.test).replace('"', "'") == "args.struct_loss != 'none'":
            return True
    return False


def check_none(syndiff):
    sc_path = os.path.join(HERE, "syndiff_train_sc.py")
    sc = open(sc_path, newline="").read().replace("\r\n", "\n")
    orig = open(os.path.join(syndiff, "train.py"), newline="").read().replace("\r\n", "\n")
    stripped = "\n".join(l for l in sc.split("\n") if "# [sc]" not in l)
    assert stripped == orig, "syndiff_train_sc.py differs from train.py outside its '# [sc]' lines"
    added = sum("# [sc]" in l for l in sc.split("\n"))
    print(f"OK  syndiff_train_sc.py = SynDiff's train.py + {added} lines tagged '# [sc]', nothing else changed")

    tree = ast.parse(sc)
    anc = Ancestors()
    anc.visit(tree)
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_syndiff")
    uses = [n for n in ast.walk(train)
            if (isinstance(n, ast.Name) and n.id in ("struct_loss", "errG_struct"))
            or (isinstance(n, ast.Attribute) and n.attr in ("lambda_struct",))]
    assert uses and all(guarded(n, anc.parents) for n in uses), \
        [n.lineno for n in uses if not guarded(n, anc.parents)]
    default = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args
               and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--struct_loss"][0]
    assert {k.arg: ast.literal_eval(k.value) for k in default.keywords}["default"] == "none"
    print(f"OK  all {len(uses)} uses of the added loss in train_syndiff are under "
          "`if args.struct_loss != 'none':`; default is none")

    # the shared wrapper patches still apply
    sys.path.insert(0, os.path.abspath(syndiff))
    import run_syndiff_train
    from utils.EMA import EMA
    original = (torch.autograd.set_detect_anomaly, EMA.__init__,
                torch.utils.data.distributed.DistributedSampler.set_epoch)
    try:
        run_syndiff_train.apply_patches()
        assert torch.autograd.set_detect_anomaly is run_syndiff_train.no_anomaly
        assert EMA.__init__ is not original[1]
        assert torch.utils.data.distributed.DistributedSampler.set_epoch is not original[2]
    finally:
        torch.autograd.set_detect_anomaly, EMA.__init__ = original[:2]
        torch.utils.data.distributed.DistributedSampler.set_epoch = original[2]
    print("OK  run_syndiff_train.apply_patches(): anomaly detection off, EMA fix and epoch hook applied")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syndiff", required=True, help="SynDiff clone (read only)")
    args = ap.parse_args()
    print(f"torch {torch.__version__}\n")
    check_losses()
    print()
    check_none(args.syndiff)
    print("\nAll struct-loss checks passed.")


if __name__ == "__main__":
    main()
