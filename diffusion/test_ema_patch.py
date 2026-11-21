#!/usr/bin/env python
"""
Local check (CPU, no GPU) of run_syndiff_train.py's fix for SynDiff's EMA optimizer on new PyTorch.

Wraps Adam on a tiny model in SynDiff's own EMA class (utils/EMA.py, unmodified), with a CosineAnnealingLR,
as train.py does; trains a few steps; then checks, through patch_ema:
  - state_dict() works and holds an EMA copy of every parameter (which differs from the parameter)
  - after torch.save / torch.load, load_state_dict() into a fresh EMA restores the EMA weights, Adam's
    moments and the learning rate, and hands them to the wrapped Adam (which EMA.step uses)
  - one more step on the original and the restored optimizer gives identical parameters and EMA weights
  - swap_parameters_with_ema (used when train.py saves checkpoints) swaps in the restored EMA weights

Usage:  python diffusion/test_ema_patch.py --syndiff ../SynDiff
"""
import argparse
import importlib.util
import io
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def build(EMA, seed):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    opt = EMA(torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.5, 0.9)), ema_decay=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10, eta_min=1e-5)
    return model, opt, sched


def train_step(model, opt, sched, x, y):
    model.zero_grad()  # train.py zeroes gradients through the networks too
    torch.nn.functional.mse_loss(model(x), y).backward()
    opt.step()
    sched.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syndiff", required=True, help="SynDiff clone (read only)")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.syndiff))
    from utils.EMA import EMA  # SynDiff's own class
    spec = importlib.util.spec_from_file_location("run_syndiff_train", os.path.join(HERE, "run_syndiff_train.py"))
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    print(f"torch {torch.__version__}")

    # without the patch (information only: whether this torch shows the bug)
    try:
        build(EMA, 0)[1].state_dict()
        print("--  unpatched EMA.state_dict() works on this torch")
    except AttributeError as e:
        print(f"--  unpatched EMA.state_dict() fails here too: AttributeError: {e}")

    added = wrapper.patch_ema(EMA)
    print(f"--  patch adds {added}")

    x, y = torch.randn(16, 4), torch.randn(16, 2)
    model, opt, sched = build(EMA, 0)
    for _ in range(5):
        train_step(model, opt, sched, x, y)

    # 1. state_dict holds the EMA weights
    sd = opt.state_dict()
    params = list(model.parameters())
    assert len(sd["state"]) == len(params) and all("ema" in s for s in sd["state"].values())
    assert all(not torch.equal(opt.optimizer.state[p]["ema"], p.data) for p in params), "EMA should lag the weights"
    print(f"OK  state_dict(): EMA weights for all {len(params)} parameters")

    # 2. save, load into a fresh EMA (different initial weights), as train.py's resume does
    buf = io.BytesIO()
    torch.save({"model": model.state_dict(), "opt": sd, "sched": sched.state_dict()}, buf)
    buf.seek(0)
    ck = torch.load(buf, weights_only=True)  # the default on torch >= 2.6
    model2, opt2, sched2 = build(EMA, 1)
    model2.load_state_dict(ck["model"])
    opt2.load_state_dict(ck["opt"])
    sched2.load_state_dict(ck["sched"])
    params2 = list(model2.parameters())
    assert opt2.optimizer.state is opt2.state and opt2.optimizer.param_groups is opt2.param_groups
    for p, p2 in zip(params, params2):
        for key in ("ema", "exp_avg", "exp_avg_sq"):
            assert torch.equal(opt.optimizer.state[p][key], opt2.optimizer.state[p2][key]), key
    assert opt2.optimizer.param_groups[0]["lr"] == opt.optimizer.param_groups[0]["lr"]
    print("OK  load_state_dict(): EMA weights, Adam moments and lr restored in the wrapped Adam")

    # 3. training continues identically
    train_step(model, opt, sched, x, y)
    train_step(model2, opt2, sched2, x, y)
    for p, p2 in zip(params, params2):
        assert torch.equal(p, p2) and torch.equal(opt.optimizer.state[p]["ema"], opt2.optimizer.state[p2]["ema"])
    print("OK  one more step after resuming matches the uninterrupted run exactly")

    # 4. the checkpoint swap uses the restored EMA weights
    emas = [opt2.optimizer.state[p]["ema"].clone() for p in params2]
    opt2.swap_parameters_with_ema(store_params_in_ema=True)
    assert all(torch.equal(p.data, e) for p, e in zip(params2, emas))
    print("OK  swap_parameters_with_ema() puts the EMA weights in the model")

    print("\nAll EMA checks passed.")


if __name__ == "__main__":
    main()
