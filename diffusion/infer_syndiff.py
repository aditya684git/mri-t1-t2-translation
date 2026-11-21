#!/usr/bin/env python
"""
Inference with a trained SynDiff model, saved as PNGs that evaluate.py / shift_check.py can score.

--direction 2 (default): T1 -> T2, with the real T1 as the source (gen_diffusive_2, as before)
--direction 1:           T2 -> T1, with the real T2 as the source (gen_diffusive_1); score these with
                         `shift_check.py --target T1` against the real T1 (testA), not with evaluate.py
--generator diffusive (default): the diffusion generator, sampled as in SynDiff's test.py
--generator non_diffusive:       the cycle's translation network for that direction instead
                                 (gen_non_diffusive_1to2 for 2, gen_non_diffusive_2to1 for 1), one forward
                                 pass; for finding where a shift comes from
--phase test (default): the 200 eval slices (data_test_*.mat), named from test_files.json
--phase val:            the validation patients (data_val_*.mat), named from manifest.json rows["val"]

Runs the sampling of SynDiff's test.py by importing test.py's own functions from the unmodified clone:
  load_checkpoint, get_time_schedule, Posterior_Coefficients, sample_from_model
and SynDiff's dataset.CreateDatasetSynthesis to read the .mat files. Two deliberate differences:
  - no CenterCrop((256, 152)): that crop fits SynDiff's 256x152 sample data and would cut our slices
  - outputs go from [-1, 1] straight to 0-255 (test.py divides each image by its own max instead)
Each output is a 256x256 grayscale PNG named like the testA/testB slice it came from.

Needs a GPU: importing SynDiff's generator compiles its CUDA ops (nvcc + ninja) on first use.

Usage:
  python infer_syndiff.py --exp_dir /scratch/$USER/syndiff_runs/syndiff_n7514 --epoch 50 \
      --data /scratch/$USER/syndiff_data/n7514 --out /scratch/$USER/syndiff_outputs/diffusion_v1
  ... --direction 1 --out /scratch/$USER/syndiff_outputs/diffusion_v1_dir1        # T2 -> T1
  ... --phase val --out /scratch/$USER/syndiff_val_outputs/syndiff_n7514_e30       # validation patients
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
from PIL import Image

# Network and diffusion settings; they must match training. Values set by SynDiff's README train/test
# commands (syndiff_train.sbatch passes the same ones):
README_SETTINGS = dict(image_size=256, num_channels=2, num_channels_dae=64, ch_mult=[1, 1, 2, 2, 4, 4],
                       num_timesteps=4, num_res_blocks=2, embedding_type="positional", z_emb_dim=256)
# Everything else NCSNpp and the sampler read: argparse defaults, identical in SynDiff's train.py and test.py.
CODE_DEFAULTS = dict(nz=100, t_emb_dim=256, n_mlp=3, attn_resolutions=(16,), dropout=0.,
                     resamp_with_conv=True, conditional=True, fir=True, fir_kernel=[1, 3, 3, 1],
                     skip_rescale=True, resblock_type="biggan", progressive="none",
                     progressive_input="residual", progressive_combine="sum", fourier_scale=16.,
                     not_use_tanh=False, centered=True, beta_min=0.1, beta_max=20., use_geometric=False)
# checkpoint name for (generator, direction), as train.py saves them
NETWORKS = {("diffusive", 2): "gen_diffusive_2", ("diffusive", 1): "gen_diffusive_1",
            ("non_diffusive", 2): "gen_non_diffusive_1to2", ("non_diffusive", 1): "gen_non_diffusive_2to1"}


def to_uint8(x):
    """[-1, 1] -> 0-255, the inverse of SynDiff's (x - 0.5) / 0.5 on our 0-1 data."""
    x = (np.asarray(x, dtype=np.float64) + 1.0) / 2.0
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def row_names(data, phase):
    """Filenames of the rows of data_<phase>_*.mat, in row order."""
    if phase == "test":
        return json.load(open(os.path.join(data, "test_files.json")))
    return json.load(open(os.path.join(data, "manifest.json")))["rows"]["val"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, help="training output folder with gen_*_<epoch>.pth")
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--data", required=True, help="make_mat.py output: data_*_T1/T2.mat, test_files.json, manifest.json")
    ap.add_argument("--out", required=True, help="folder for the generated PNGs")
    ap.add_argument("--direction", type=int, choices=[1, 2], default=2, help="2: T1 -> T2, 1: T2 -> T1")
    ap.add_argument("--generator", choices=["diffusive", "non_diffusive"], default="diffusive")
    ap.add_argument("--phase", choices=["test", "val"], default="test",
                    help="test: the 200 eval slices; val: the validation patients")
    ap.add_argument("--syndiff", default=os.path.expanduser("~/SynDiff"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42, help="test.py uses torch.manual_seed(42)")
    cli = ap.parse_args()

    names = row_names(cli.data, cli.phase)
    net_name = NETWORKS[(cli.generator, cli.direction)]
    ckpt = os.path.join(cli.exp_dir, f"{net_name}_{cli.epoch}.pth")
    if not os.path.exists(ckpt):
        raise SystemExit(f"No checkpoint {ckpt}")

    # SynDiff code, unmodified. test.py is loaded under another name: "test" is also a Python stdlib package.
    sys.path.insert(0, cli.syndiff)
    import torch
    from backbones.ncsnpp_generator_adagn import NCSNpp
    import backbones.generator_resnet
    from dataset import CreateDatasetSynthesis
    spec = importlib.util.spec_from_file_location("syndiff_test", os.path.join(cli.syndiff, "test.py"))
    syndiff_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(syndiff_test)

    args = argparse.Namespace(**README_SETTINGS, **CODE_DEFAULTS)
    torch.manual_seed(cli.seed)
    torch.cuda.set_device(cli.gpu)
    device = torch.device(f"cuda:{cli.gpu}")

    dataset = CreateDatasetSynthesis(cli.phase, cli.data, "T1", "T2")
    if len(dataset) != len(names):
        raise SystemExit(f"{len(dataset)} {cli.phase} rows but {len(names)} names")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    if cli.generator == "diffusive":
        net = NCSNpp(args).to(device)
    else:  # built as train.py builds it (DataParallel inside), so the keys match once "module." is stripped
        net = backbones.generator_resnet.define_G(netG="resnet_6blocks", gpu_ids=[cli.gpu])
    # load_checkpoint strips the DDP "module." prefix and calls eval()
    syndiff_test.load_checkpoint(os.path.join(cli.exp_dir, "{}_{}.pth"), net, net_name,
                                 epoch=str(cli.epoch), device=device)
    T = syndiff_test.get_time_schedule(args, device)
    pos_coeff = syndiff_test.Posterior_Coefficients(args, device)

    os.makedirs(cli.out, exist_ok=True)
    with torch.no_grad():
        for i, (t1, t2) in enumerate(loader):
            source, target = (t1, t2) if cli.direction == 2 else (t2, t1)
            source = source.to(device, non_blocking=True)
            if cli.generator == "diffusive":
                # the target only sets the shape of the starting noise, as in test.py
                x_t = torch.cat((torch.randn_like(target.to(device)), source), axis=1)
                fake = syndiff_test.sample_from_model(pos_coeff, net, args.num_timesteps, x_t, T, args)
            else:
                fake = net(source)
            Image.fromarray(to_uint8(fake[0, 0].cpu().numpy())).save(os.path.join(cli.out, names[i]))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(names)} saved", flush=True)

    what = "T1 -> T2" if cli.direction == 2 else "T2 -> T1"
    print(f"{len(names)} {cli.phase} slices, {what}, {net_name} epoch {cli.epoch} -> {cli.out}")


if __name__ == "__main__":
    main()
