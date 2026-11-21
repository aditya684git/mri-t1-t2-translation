# Diffusion: SynDiff (Aditya)

T1 → T2 with [SynDiff](https://github.com/icon-lab/SynDiff) (Özbey, Dalmaz et al., IEEE TMI 2023), an
unpaired adversarial diffusion model with a cycle-consistent translation network. We use the official code,
unmodified, at commit `fff3d8449e8c7ba38339be2f9ffd4aa5572beb4b` (also in `../upstream_versions.txt`),
on the same data, split, eval slices and `evaluate.py` as CycleGAN and CUT.

| File | What it does |
|---|---|
| `make_mat.py` | our PNG dataset → SynDiff's six `.mat` files + `test_files.json` + `manifest.json` |
| `infer_syndiff.py` | trained generator → one 256 × 256 PNG per slice, named like `testA`. Default `gen_diffusive_2` (T1 → T2) on the eval slices; `--direction 1` (T2 → T1), `--generator non_diffusive`, `--phase val` |
| `run_syndiff_train.py` | runs SynDiff's `train.py` unmodified, with anomaly detection off, its EMA optimizer fixed for new PyTorch, and per-epoch timing |
| `syndiff_smoke.sbatch` | 500 slices: 1 epoch, check `content.pth`, resume for 1 more, then inference + scoring |
| `syndiff_train.sbatch` | full training; resubmit the same command to resume |
| `syndiff_infer.sbatch` | inference on the 200 eval slices, then `evaluate.py` (or, for direction 1, `shift_check.py --target T1`) |
| `test_local.py` | laptop check of `make_mat.py` on fake PNGs, through SynDiff's own loader |
| `align_to_input.py` | undoes whole-pixel output shifts by aligning each output to its input T1 (any model) |
| `test_ema_patch.py` | laptop check (CPU) that the EMA fix saves and restores the EMA weights |
| `test_align.py` | laptop check of `align_to_input.py` on fake shifted outputs in all three file layouts |
| `syndiff_shift_cause.sbatch` | measures the shift of all four generators of one checkpoint (section 7) |
| `syndiff_train_sc.py` | SynDiff's `train.py` plus `--struct_loss {none,silhouette,mind}`; added lines tagged `# [sc]` |
| `struct_loss.py` | the silhouette and MIND losses |
| `syndiff_sc_smoke.sbatch` | 500 slices, both losses: 1 epoch, resume for 1 more, validation scoring |
| `syndiff_sc_train.sbatch` | training with a struct loss; resubmit the same command to resume |
| `syndiff_val.sbatch` | inference + `score_val.py` on the validation patients for every saved epoch of a run |
| `score_val.py` | validation scores, input-based shift, and the selection rule with its baseline quality floor (`--select`) |
| `test_struct_loss.py` | laptop check (CPU) of both losses, and that `--struct_loss none` changes nothing |
| `test_score_val.py` | laptop check of `score_val.py` on fake shifted outputs |

## Data format (`make_mat.py`)

Copied from `SynDiff_sample_data/T1.mat`: HDF5 with one dataset `data_fs`, float32, `(N, 256, 256)`,
contiguous, values 0–1. SynDiff's loader transposes every slice (MATLAB column order), so each image is
stored transposed and comes out of the loader upright, identical to our PNG (checked pixel for pixel by
`test_local.py`). The loader then maps 0–1 to −1…1.

| File | Rows |
|---|---|
| `data_train_T1/T2.mat` | `--n_train` random slices from `trainA` / `trainB` (seed 42), shuffled independently. SynDiff pairs rows by index, so both need the same length; the rows come from different patients, so training stays unpaired. Default: all = 7,514 (the smaller of the two). |
| `data_val_T1/T2.mat` | every slice of 5 test-split patients **not** in the eval subset, T1/T2 paired |
| `data_test_T1/T2.mat` | exactly the 200 `eval_subset.files` from `split.json`, in that order; `test_files.json` lists them |

`manifest.json` records the seed, the val patients and every row's filename.

## Commands, in order (on Palmetto)

### 1. One-time setup

```bash
cd ~
git clone https://github.com/icon-lab/SynDiff.git
cd ~/SynDiff && git checkout fff3d8449e8c7ba38339be2f9ffd4aa5572beb4b    # never modify or push

cd ~/mri-t1-t2-translation          # your clone of this repo; every sbatch below is submitted from here
mkdir -p logs

module load anaconda3/2023.09-0
module load cuda/12.3.0             # SynDiff compiles two CUDA ops on first import and needs nvcc
module load gcc/12.3.0              # a C++20 compiler that CUDA 12.3 accepts
export CC=gcc CXX=g++
export TORCH_CUDA_ARCH_LIST="7.0;8.0"   # V100, A100
source activate mri
export PYTHONNOUSERSITE=1
python -m pip install h5py ninja
which nvcc                          # must show the cuda/12.3.0 module's nvcc
```

All three sbatch scripts set the same four lines. Not the default modules: CUDA 13.0.2 does not match the
cu126 PyTorch, the system `c++` is too old for C++20, and the default gcc/13.2.0 is too new for CUDA 12.3.

### 2. Make the data (a few minutes each; use an interactive session, not the login node)

```bash
D=/scratch/$USER/brats/brats_t1t2
python diffusion/make_mat.py --data $D --out /scratch/$USER/syndiff_data/n500  --n_train 500
python diffusion/make_mat.py --data $D --out /scratch/$USER/syndiff_data/n2000 --n_train 2000
python diffusion/make_mat.py --data $D --out /scratch/$USER/syndiff_data/n7514
```

Each prints the count, shape, dtype and min/max of every file. The full train files are ~2 GB each.

### 3. Smoke test

```bash
sbatch diffusion/syndiff_smoke.sbatch
tail -30 logs/syndiff_smoke_<jobid>.out
```

It trains epoch 0, checks that `content.pth` exists and holds the EMA weights of all 4 generator optimizers,
resumes with `--resume` for epoch 1, then generates and scores. Expect lines like
`[2026-10-30 10:12:03] epoch 1 took 312 s (training + saving + validation)`: use these to size the real run
(they exclude startup and compiling the CUDA ops; the "run N done" lines include it). Also expect 200 PNGs in
`/scratch/$USER/syndiff_outputs/syndiff_smoke/`, and scores in its `eval/` folder (not in `results/`).

### 4. Train

```bash
sbatch diffusion/syndiff_train.sbatch n7514 <epochs>        # epochs: a multiple of 5
```

Output: `/scratch/$USER/syndiff_runs/syndiff_n7514/` (`gen_diffusive_2_<epoch>.pth` every 5 epochs,
`content.pth` for resuming, sample images). If the job hits the 3-day limit, run the same command again: it
resumes from `content.pth`. Keep `<epochs>` the same, since the cosine learning-rate schedule is built for it.

### 5. Generate and score

```bash
sbatch diffusion/syndiff_infer.sbatch syndiff_n7514 <epoch> n7514 diffusion_v1
cat results/results.csv
```

PNGs go to `/scratch/$USER/syndiff_outputs/diffusion_v1/`, scores to a `diffusion_v1` row in
`results/results.csv`.

### 6. Undo output shifts (`align_to_input.py`)

**Finding.** SynDiff's outputs come out shifted by whole pixels: up to about 8 px, and by a different amount in
each run. `shift_check.py` (which slides each output against the real T2) found these best shifts (dy, dx):
diffusion_v1 (+2, −8), the epoch-5 checkpoint (+2, −8), n500 about (+1, −7), n2000 (+1, 0), and CycleGAN (0, 0).
A shift like that costs pixel-wise scores (SSIM, PSNR, MAE) even when the anatomy is right. Section 7 tests a
likely cause and adds a loss against it.

**Fix, using the input only.** For each output slice, `align_to_input.py` finds the integer shift within
±16 px that best overlaps the output's brain silhouette with the silhouette of its **input T1** (`testA`, same
filename). It maximises Dice, then applies that shift with zero fill. The silhouette is the pixels above 0.05,
with holes filled, since ventricles are dark in T1 and bright in T2. It never reads `testB` or `testMask`, so
this is legitimate test-time post-processing. The `shift_check.py` numbers above come from the test targets:
they are only for checking that the input-based shifts come out similar, and the script does not use them.

```bash
D=/scratch/$USER/brats/brats_t1t2
O=/scratch/$USER/syndiff_outputs
python diffusion/align_to_input.py --pred_dir $O/diffusion_v1 --out $O/diffusion_v1_aligned --data $D
python evaluate.py --run diffusion_v1_aligned --pred_dir $O/diffusion_v1_aligned --data $D

# same for the GANs (their file layouts are handled), into your own scratch folder
R=/scratch/$USER/mri_runs/results
python diffusion/align_to_input.py --pred_dir $R/cut_v1/test_latest/images --out /scratch/$USER/aligned/cut_v1 --data $D
python diffusion/align_to_input.py --pred_dir $R/cyclegan_v1/test_latest/images --out /scratch/$USER/aligned/cyclegan_v1 --data $D

# diagnostic only: aligned outputs should now peak at (0, 0)
python shift_check.py --data $D --radius 3 --runs diffusion_v1_aligned=$O/diffusion_v1_aligned
```

Each run prints the median shift, the IQR and range, how many slices got each shift, and the silhouette Dice
before and after. It also writes `shifts.csv` (one row per slice) next to the aligned PNGs. A warning flags
slices whose best shift sits at the ±16 px search limit. `--apply median` applies the run's median shift to every
slice instead of each slice's own; use it if the per-slice shifts look noisy but the run's shift is clearly constant.
Report scores both with and without alignment, label which is which, and align all three models the same way.
It takes about 0.2 s per slice (the 200 eval slices take under a minute).

### 7. Why the shift happens, and a position-preserving loss

**Hypothesis.** Nothing in SynDiff's losses fixes where the anatomy sits:
- The adversarial losses use patch discriminators, which judge local realism and barely see position.
- The cycle loss only needs the two translation networks to undo each other, so if `gen_non_diffusive_1to2`
  shifts by s and `gen_non_diffusive_2to1` by −s, the cycle still closes.

SynDiff then passes the shift on to its diffusive generators, which are trained on pairs made by the
*opposite* translation network:
- `gen_diffusive_2` learns to turn `gen_non_diffusive_2to1(T2)` back into that T2, so it learns to apply +s.
- `gen_diffusive_1` learns to apply −s.

At test time, `gen_diffusive_2` gets a real, unshifted T1 and applies +s. That would also explain why the shift
differs from run to run: nothing prefers one value of s. CUT and CycleGAN show no shift, which this account does
not explain.

**Step 1: test it.** `syndiff_shift_cause.sbatch` runs all four generators of one checkpoint on the eval
slices. It measures each one's shift with `shift_check.py` against the real image of the output's contrast,
using `--target T1` for the T2 → T1 generators. This is a diagnostic: it chooses nothing.

```bash
sbatch diffusion/syndiff_shift_cause.sbatch syndiff_n7514 30 n7514
# just the reverse diffusive direction:
sbatch diffusion/syndiff_infer.sbatch syndiff_n7514 30 n7514 diffusion_v1_dir1 1
```

| generator | direction | expected if the hypothesis holds (diffusion_v1) |
|---|---|---|
| `gen_diffusive_2` | T1 → T2 | (+2, −8), already measured |
| `gen_non_diffusive_1to2` | T1 → T2 | (+2, −8) |
| `gen_diffusive_1` | T2 → T1 | (−2, +8) |
| `gen_non_diffusive_2to1` | T2 → T1 | (−2, +8) |

The other outcomes would say:
- If the translation networks show (0, 0) and only the diffusive ones are shifted, the shift arises in the
  diffusive training instead.
- If the two directions are not opposite, the cancelling-in-the-cycle account is wrong.

**Step 2: the loss.** `syndiff_train_sc.py` is SynDiff's `train.py` with a minimal diff. Every added line ends
in `# [sc]`, and SynDiff itself is untouched. It adds `--struct_loss {none,silhouette,mind}` with weight
`--lambda_struct`. The generator loss gains λ × (sum of four terms). Each term compares a generator's predicted
clean image with that generator's own source, so no target of the other contrast is used:

| term | output | its source |
|---|---|---|
| `nd_1to2` | `gen_non_diffusive_1to2(T1)` | the real T1 |
| `nd_2to1` | `gen_non_diffusive_2to1(T2)` | the real T2 |
| `diff_1` | `gen_diffusive_1`'s predicted x₀ | its source, the translated T2 |
| `diff_2` | `gen_diffusive_2`'s predicted x₀ | its source, the translated T1 |

The two losses (`struct_loss.py`, on the 0–1 scale):
- **silhouette:** L1 between the output's soft brain mask, sigmoid((x − 0.03) / 0.006), and the source's mask.
  For a real source the mask is source > 0, since background is exactly 0 after preprocessing; for a generated
  source it is the same soft mask. Gradients reach only the brain outline, so this fixes the outline's position
  but says nothing about the anatomy inside it. When aligned the loss sits slightly above 0 (about 0.004), from
  the sigmoid tails.
- **mind:** L1 between MIND descriptors of output and source, averaged over the source's brain mask. MIND is
  from Heinrich et al. (2012). The settings follow Yang et al. (2018), who used it as a structure loss for
  unpaired brain MR → CT: 9 × 9 non-local region, 7 × 7 Gaussian patch with σ = 2, 4-neighbourhood variance.
  It does not change under an intensity scaling or inversion of the whole image, but it does change when
  structures move.

The terms are printed every 100 iterations under train.py's loss line. `--struct_loss none` (the default) runs
none of the added code, so it trains exactly as `train.py` does; `test_struct_loss.py` checks this from the
source. Both training scripts apply the same wrapper patches (`run_syndiff_train.apply_patches`).

**Starting weights.**
- **MIND.** The two papers that add a MIND loss to an unpaired cycle model both weight it at half the cycle
  weight: Yang et al. 2018 (λ_cycle 10, λ_structure 5, L1 on MIND) and Xu et al. 2020 (λ_cyc 10, λ_MIND 5).
  SynDiff's cycle weight is 0.5 (`--lambda_l1_loss`, which also weights its L1 terms), so the same per-term ratio
  gives **λ = 0.25 for mind**. We also try **1**, closer to those papers' MIND : adversarial ratio of 5 : 1, in
  case 0.25 is too weak against SynDiff's stronger adversarial terms.
- **Silhouette.** No paper uses this exact loss. It is an L1 term like SynDiff's own, so it starts at SynDiff's
  L1 weight, **λ = 0.5**, with **2** as the stronger option.
- **On the fake slices** in `test_struct_loss.py`, an 8 px shift raises mind from 0.024 to 0.175 and silhouette
  from 0.004 to 0.041.

**Step 3: commands, in order.**

```bash
sbatch diffusion/syndiff_sc_smoke.sbatch                    # both losses on n500, 2 epochs; compare "took" times with syndiff_smoke

# candidates on n500, whose baseline run shifted; same seed (1024), so the same initial weights as syndiff_n500
sbatch diffusion/syndiff_sc_train.sbatch n500 30 mind 0.25
sbatch diffusion/syndiff_sc_train.sbatch n500 30 mind 1
sbatch diffusion/syndiff_sc_train.sbatch n500 30 silhouette 0.5
sbatch diffusion/syndiff_sc_train.sbatch n500 30 silhouette 2

# validation scores for every saved epoch, the baseline included
sbatch diffusion/syndiff_val.sbatch syndiff_n500 n500
sbatch diffusion/syndiff_val.sbatch sc_n500_mind_0.25 n500      # ... and the other three
python diffusion/score_val.py --select                         # the ranking, by the rule below

# the selected loss and weight on all the data, then validation again for its epoch
sbatch diffusion/syndiff_sc_train.sbatch n7514 30 <loss> <lambda>
sbatch diffusion/syndiff_val.sbatch sc_n7514_<loss>_<lambda> n7514
sbatch diffusion/syndiff_shift_cause.sbatch sc_n7514_<loss>_<lambda> <epoch> n7514   # all four should be near (0, 0)
sbatch diffusion/syndiff_infer.sbatch sc_n7514_<loss>_<lambda> <epoch> n7514 diffusion_sc   # the eval slices, once
```

`syndiff_val.sbatch` writes one row per run and epoch, `<exp>_e<epoch>`, to `results/syndiff_val/val_results.csv`.
Each row has:
- the raw brain SSIM, PSNR and MAE against the real T2, from `evaluate.py`'s own scoring with no alignment;
- the median input-based shift from `align_to_input.py`'s search, and the share of slices within 1 px of (0, 0);
- the SSIM after undoing each slice's shift (the baseline's sets the quality floor below).

The validation slices are every slice of the 5 test-split patients in `manifest.json`'s `val_patients`, none of
them in the eval subset. They are the same 5 patients for n500, n2000 and n7514.

**Selection rule (validation only).**
1. Candidates are every saved epoch from 1 on of every run (loss, weight), the baseline included, scored only on
   the validation patients. Epoch 0 never counts.
2. Drop any candidate whose median input-based shift is more than 1 px from (0, 0) on either axis. The aim is
   outputs that need no alignment.
3. Drop any candidate whose raw validation SSIM is below the **quality floor**: the best aligned validation SSIM
   of the baseline run with the same data size (`syndiff_<data>`, any epoch from 1 on). For n500 that is 0.290,
   from `syndiff_n500_e30`.
   - An unshifted model only counts if it is at least as good as the baseline after alignment. Without this
     floor, the only unshifted baseline epoch, the undertrained `syndiff_n500_e5` (raw SSIM −0.025), won.
   - Score the baseline with `syndiff_val.sbatch` first. Without baseline rows there is no floor, and nothing of
     that data size passes.
4. Of the rest, pick the highest raw validation SSIM. SSIMs within 0.005 count as a tie, which goes to the lower
   MAE.
5. If no candidate passes, nothing is chosen. `score_val.py --select` then says "No candidate passes" and lists
   the best row of each run, with the reason it fails.
6. Choose the epoch of the n7514 run the same way (its floor comes from `syndiff_n7514`). Then run the 200 eval
   slices once and report its raw scores, with aligned scores secondary as in section 6. Eval numbers never feed
   back into any choice.

`score_val.py --select` applies steps 1–5. It reads the data size from the run name (`<exp>_e<epoch>`, the
`n<N>` part of `<exp>`), so keep the default experiment names.

**Caution.** The baseline n2000 run happened to come out almost unshifted, so one unshifted run with the loss
proves little. Look also for the `nd_1to2` / `nd_2to1` terms falling in the training log, and for
`syndiff_shift_cause.sbatch` on the selected run showing about (0, 0) for all four generators.

References:
- Heinrich et al., "MIND: Modality independent neighbourhood descriptor for multi-modal deformable
  registration", Med. Image Anal. 16(7), 2012.
- Yang et al., "Unpaired brain MR-to-CT synthesis using a structure-constrained CycleGAN", DLMIA 2018,
  [arXiv:1809.04536](https://arxiv.org/abs/1809.04536).
- Xu et al., "Adversarial uni- and multi-modal stream networks for multimodal image registration", MICCAI 2020,
  [arXiv:2007.02790](https://arxiv.org/abs/2007.02790).

### Local check (laptop, no GPU)

```bash
python diffusion/test_local.py --syndiff ../SynDiff
python diffusion/test_ema_patch.py --syndiff ../SynDiff
python diffusion/test_align.py
python diffusion/test_struct_loss.py --syndiff ../SynDiff
python diffusion/test_score_val.py                  # needs scikit-image
```

## SynDiff behaviour to know about (from reading its code, unmodified)

- **Epochs run `0..num_epoch` inclusive**, so `--num_epoch 50` trains 51 epochs.
- **The final validation pass always crashes** with an `IndexError` (its arrays are sized `num_epoch`).
  It happens after that epoch's checkpoints are saved, so the scripts check for the checkpoint instead of the
  exit code.
- **Validation numbers are only half useful.** Its second val loop uses `gen_diffusive_1` for both
  directions, so `val_psnr_values.npy[1]` is not T1 → T2 quality. Both val arrays restart at zero after a resume.
- **Resume on new PyTorch:** `content.pth` stores an `argparse.Namespace`, which torch ≥ 2.6 refuses to load by
  default; `syndiff_train.sbatch` sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.
- **EMA optimizer on new PyTorch.** SynDiff's `utils/EMA.py` subclasses `torch.optim.Optimizer` without calling
  its `__init__`, so on current PyTorch `state_dict()` fails at "Saving content" (`'EMA' object has no attribute
  '_optimizer_state_dict_pre_hooks'`). `run_syndiff_train.py` adds every attribute the installed torch's
  `Optimizer.__init__` sets and EMA lacks. It finds them from a throwaway optimizer, not a fixed list, and
  prints their names at startup. The EMA weights stay in the optimizer state, so `content.pth` saves them and
  `--resume` restores them (`test_ema_patch.py`).
- **No anomaly detection.** `train.py` calls `torch.autograd.set_detect_anomaly(True)` inside its loop, which
  slows every step a lot. Both training scripts run it through `run_syndiff_train.py`, which replaces that function
  with a no-op and then runs `train.py` with the same arguments, working directory and `sys.path`, so SynDiff
  stays unmodified. Training is otherwise identical. Anomaly detection only adds error checking (it reports
  where a NaN first appeared); to get it back for debugging, call `python train.py ...` directly.
- `test.py` is not used: it crops every output to 256 × 152 (`CenterCrop`), which cuts our slices and, as written,
  doesn't fit the 256 × 256 arrays it stores them in; it also rescales each output by its own max.
  `infer_syndiff.py` runs the same sampling, without either.
