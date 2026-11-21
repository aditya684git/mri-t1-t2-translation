# CycleGAN and CUT: data, training, evaluation and findings

Detailed notes for the GAN half of the project (the diffusion half is in [`../diffusion/README.md`](../diffusion/README.md);
the project overview and final results are in the [top-level README](../README.md)). All commands run from the repo root.

We compare three unpaired image-to-image translation methods on the same task:
generate a T2-weighted brain MRI slice from a T1-weighted slice of the same patient.

| Model | Owner | Code |
|---|---|---|
| CycleGAN (Zhu et al., 2017) | Shrinidhi | [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) |
| CUT (Park et al., 2020) | Shrinidhi | [taesungp/contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) |
| Diffusion: SynDiff (Özbey et al., 2023) | Aditya | [icon-lab/SynDiff](https://github.com/icon-lab/SynDiff); our scripts and findings in [`diffusion/README.md`](../diffusion/README.md) |

All three use **the same data, the same split, the same test slices and the same evaluation script**.

---

## Data

**BraTS 2021 Task 1** training set (Kaggle mirror `dschettler8845/brats-2021-task1`): 1,251 patients,
each with co-registered, skull-stripped T1 / T1ce / T2 / FLAIR volumes (240 × 240 × 155). We use T1 and T2 only.
The data is **not** in this repo (licence + size); see "Reproducing" below.

**Alignment check** (`check_alignment.py`): on 5 random patients, T1 and T2 have identical shapes, affines and brain
masks (Dice 1.000), and overlays line up → outputs can be compared to the real T2 pixel by pixel.
See `figures/alignment_check.png`.

**Preprocessing** (`preprocess.py`, settings recorded in `split.json`):

1. 400 random patients (seed 42), split **by patient**: 160 → `trainA` (T1 only), a *different* 160 → `trainB`
   (T2 only), 80 → test. No patient appears in two splits, so training is truly unpaired.
2. Per volume: clip to the 0.5–99.5th percentile of **brain** voxels, scale to 0–1, background stays 0.
3. Axial slices where the brain area is ≥ 40% of the volume's largest brain slice, every 2nd slice.
4. Rotate upright, zero-pad 240 → 256, save as 8-bit grayscale PNG named `<patient>_s<slice>.png`.
5. Test slices are saved as T1 (`testA`), T2 (`testB`) and brain mask (`testMask`) under identical filenames.

| Folder | Slices |
|---|---|
| trainA (T1) | 7,551 |
| trainB (T2) | 7,514 |
| testA / testB / testMask | 3,824 each |
| **Eval subset** (fixed, in `split.json`) | 200 (20 test patients × 10 slices) |

---

## Training (CycleGAN and CUT)

Identical budget and settings for both (`cut_v1.sbatch`, `cyclegan_v1.sbatch`):

- grayscale in/out (`--input_nc 1 --output_nc 1`), 256 × 256, no resizing (`--load_size 256 --crop_size 256`)
- batch size 1, ResNet-9-block generator (11.37 M parameters in both), PatchGAN discriminator
- 60 epochs: 30 at learning rate 2e-4, then 30 decaying linearly to 0
- random horizontal flip on (repo default) for both
- one V100 per run, ~17.5 min/epoch (CUT) and ~20 min/epoch (CycleGAN), ≈ 18–20 h each

Smoke tests (`cut_smoke.sbatch`, `cyclegan_smoke.sbatch`) ran 2 short epochs on 500 images to check each pipeline end to end.

## Training (SynDiff)

SynDiff uses the same data and eval slices, converted to its `.mat` format. It uses the network and optimiser
settings from its README, except the epoch count: 30 epochs on all 7,514 slices per contrast, one A100. The scripts, the shift it showed and the fix are in
[`diffusion/README.md`](../diffusion/README.md).

## Evaluation

`evaluate.py` scores generated T2 slices against the real T2 **inside the brain mask only** (background excluded,
since every model gets it right for free and it would inflate scores):

- **SSIM**: full SSIM map, averaged over brain pixels
- **PSNR**: from the mean squared error over brain pixels
- **MAE**: mean absolute error over brain pixels (0–1 scale)
- Also reported, for comparison with papers only: whole-image SSIM and PSNR (background included)

Default is the fixed 200-slice eval subset; `--subset all` scores every test slice. Output: a per-slice CSV
(for finding failure cases) and one row per run in `results/results.csv`.

`compare_outputs.py` makes side-by-side figures (input T1 | each model's fake T2 | real T2 | error maps).

### Results (eval subset, 200 slices from 20 test patients)

| Run | SSIM (brain) | PSNR (brain, dB) | MAE (brain) | SSIM (whole image) | PSNR (whole image, dB) |
|---|---|---|---|---|---|
| identity baseline (T1 used as T2) | −0.130 | 7.70 | 0.375 | | |
| CUT (`cut_v1`) | 0.146 ± 0.092 | 13.17 ± 1.84 | 0.171 ± 0.051 | 0.814 | 20.22 |
| CycleGAN (`cyclegan_v1`) | 0.512 ± 0.129 | 15.90 ± 2.71 | 0.129 ± 0.063 | 0.888 | 22.80 |
| SynDiff original (`diffusion_v1`), as generated | 0.105 ± 0.049 | 12.04 ± 1.47 | 0.190 ± 0.043 | 0.773 | 18.13 |
| SynDiff original, aligned to its input (`diffusion_v1_aligned`) | 0.507 ± 0.125 | 15.82 ± 2.38 | 0.127 ± 0.053 | 0.887 | 22.80 |
| **SynDiff + silhouette loss (`diffusion_sc_v1`)** | **0.543 ± 0.134** | **16.15 ± 2.65** | **0.124 ± 0.056** | **0.897** | **23.21** |

± is the standard deviation over the 200 slices. Side-by-side outputs and error maps:
`figures/compare_cut_vs_cyclegan.png` and `figures/final_four_models.png`. Slide-ready charts are in
`figures/slides/` (see its README).

### Findings so far

**1. CycleGAN beats CUT on every metric, the opposite of our proposal's hypothesis.**
The gap is largest inside the brain (SSIM 0.51 vs 0.15), and the error maps show why: CycleGAN reproduces the
cortical folds closely, while CUT's errors line the folds.

**2. It is not a misalignment.** `shift_check.py` slides each output ±2 px and re-scores. CycleGAN peaks exactly at
(0, 0). CUT has no peak at all: its *unshifted* position scores lowest, meaning its fine structure is partly
reversed relative to the real T2, rather than offset.

**3. CUT keeps T1's fine-scale contrast; CycleGAN reverses it, as real T2 does.** `polarity_check.py` correlates
only the fine detail (image minus a 2 px Gaussian blur) inside the brain:

| | fine detail vs input T1 | fine detail vs real T2 |
|---|---|---|
| real T1 vs real T2 (reference) | | −0.42 |
| CUT | **+0.43** | −0.13 |
| CycleGAN | −0.47 | **+0.42** |

In real scans, fine T1 and T2 detail run in opposite directions (thin fluid lines are dark in T1, bright in T2).
CycleGAN learns this reversal; CUT changes the overall brightness but keeps the T1 fine pattern. A plausible cause
is CUT's patch-wise contrastive loss, which pushes each output patch to stay recognisably similar to its input
patch, and so resists reversing fine contrast. Caveat: one training run per model.

**4. Brain-only metrics are much stricter than whole-image ones.** Background is trivially correct for every
model, so whole-image SSIM (0.81–0.89, in the range papers report) hides most of the in-brain differences.

**5. Ground-truth brightness varies between patients.** After per-volume normalization, the median brain
brightness of the real T2 ranges 0.18–0.66 across test patients, while the models output a fairly constant
"typical" level. Part of every model's error is this mismatch rather than wrong anatomy (`intensity_check.py`).
It affects all models, so the comparison stays fair. Failure-case example: on `BraTS2021_01163_s094` CycleGAN
makes the tumour much brighter than in the real T2.

**6. SynDiff's outputs were shifted by whole pixels; the cause is confirmed and fixed in training.** The original
SynDiff produced realistic T2s placed 8 px too far right and 2 px too high (best correction (dy, dx) = (+2, −8)),
which drops its brain SSIM to 0.105. CycleGAN and CUT are not shifted (their best correction is (0, 0)). The cause:
- Neither the adversarial nor the cycle loss constrains position, so its two translation networks learned
  opposite shifts that cancel in the cycle.
- Its diffusive generators inherit them. All four generators measure (+2, −8) or (−2, +8)
  (`results/shift_cause_n7514_e30.txt`).

The fix is a silhouette loss: L1 between each output's soft brain mask and its own source's mask. It brings all
four to (0, 0). The loss type and weight were chosen on validation patients outside the eval set. MIND, the
alternative, removed the shift but broke the translation.

With the loss, SynDiff needs no post-processing and beats CycleGAN, the best GAN. It is better on SSIM
(17/20 patients, p = 0.00017); PSNR and MAE are slightly better on average but not significant. The p-value is a
paired Wilcoxon signed-rank test over the 20 patient means.

Caveats:
- One training run per model.
- Epoch 30, SynDiff's final epoch, is reported, matching the GANs' final-epoch rule. The validation-selected
  epoch 5 scores eval SSIM 0.566.

Details: [`diffusion/README.md`](../diffusion/README.md), section 7.

---

## Reproducing

Run on Clemson's Palmetto 2 cluster (Slurm). Set `#SBATCH --account=<your_account>` in each `.sbatch` file.

### 1. Environment

```bash
module load anaconda3/2023.09-0
conda create -n mri python=3.10 -y
source activate mri
export PYTHONNOUSERSITE=1          # needed in every shell and job: ignores stale packages in ~/.local
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
# visdom (needed by CUT) fails to build with new setuptools:
python -m pip install "setuptools<81" && python -m pip install visdom --no-build-isolation
```

### 2. Model code (pinned versions)

```bash
git clone https://github.com/taesungp/contrastive-unpaired-translation.git
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
# check out the exact commits listed in upstream_versions.txt, then apply our CUT fix:
cd contrastive-unpaired-translation && git apply ../patches/cut_grayscale.patch && cd ..
```

**Why the CUT patch:** CUT's `UnalignedDataset` always loads images as RGB, so with `--input_nc 1` training crashes
with a channel mismatch. The patch passes `grayscale=(input_nc == 1)` to the transform, one line in
`data/unaligned_dataset.py`. CycleGAN's loader already handles this.

**CycleGAN note:** the current upstream version replaced visdom with wandb. `wandb` must be installed (it is
imported unconditionally) but stays off unless `--use_wandb` is passed, and `--display_id` no longer exists.

### 3. Data

```bash
# Kaggle legacy API key in ~/.kaggle/kaggle.json; the kaggle CLI hung, so we used curl with the same credentials
curl -L -u "$KAGGLE_USER:$KAGGLE_KEY" -o brats-2021-task1.zip \
  https://www.kaggle.com/api/v1/datasets/download/dschettler8845/brats-2021-task1
unzip brats-2021-task1.zip BraTS2021_Training_Data.tar
mkdir extracted && tar -xf BraTS2021_Training_Data.tar -C extracted

python check_alignment.py --root extracted --n 5 --out figures/alignment_check.png
python preprocess.py --root extracted --out brats_t1t2 --min_frac 0.4 --workers 10
python check_preprocess.py --data brats_t1t2 --out figures/preprocess_check.png
```

### 4. Train, test, evaluate

```bash
sbatch cut_v1.sbatch
sbatch cyclegan_v1.sbatch
sbatch run_test.sbatch cut_v1            # or: bash run_test.sbatch cut_v1   (inside an interactive GPU session)
sbatch run_test.sbatch cyclegan_v1
python evaluate.py --run cut_v1      --pred_dir <results>/cut_v1/test_latest/images
python evaluate.py --run cyclegan_v1 --pred_dir <results>/cyclegan_v1/test_latest/images
python evaluate.py --run identity    --pred_dir brats_t1t2/testA
python compare_outputs.py --runs cut_v1=<dir> cyclegan_v1=<dir>
```

### Scoring the diffusion model

Save generated T2 slices as 256 × 256 grayscale PNGs **with the same filenames as `testA`**
(e.g. `BraTS2021_00019_s031.png`) into one folder, then:

```bash
python evaluate.py --run diffusion_v1 --pred_dir <that folder>
```

At minimum, generate the 200 files listed under `eval_subset.files` in `split.json`.

---

## Files

| File | What it does |
|---|---|
| `check_alignment.py` | T1/T2 co-registration check (shape, affine, mask Dice, overlay figure) |
| `preprocess.py` | volumes → PNG folders + `split.json` |
| `check_preprocess.py` | preview grid of random slices from every folder |
| `split.json` | patient split, settings, slice counts, fixed eval subset |
| `gpu_test.sbatch` | checks a batch job gets a working GPU |
| `cut_smoke.sbatch`, `cyclegan_smoke.sbatch` | 2-epoch pipeline tests |
| `cut_v1.sbatch`, `cyclegan_v1.sbatch` | full 60-epoch training runs |
| `run_test.sbatch` | inference on every test slice for a trained run |
| `evaluate.py` | masked SSIM / PSNR / MAE → `results/` |
| `compare_outputs.py` | side-by-side comparison figure with error maps |
| `intensity_check.py` | per-patient brightness of real vs generated T2 |
| `shift_check.py` | checks whether outputs are spatially offset from the real T2 (`--target T1` for T2 → T1 outputs) |
| `polarity_check.py` | does fine detail follow the input T1 or the real T2? |
| `patches/cut_grayscale.patch` | our one-line fix to CUT's data loader |
| `upstream_versions.txt` | exact commits of the three model repos we used |
| `diffusion/` | SynDiff: data conversion, training and inference jobs, shift diagnosis, silhouette loss, validation selection ([README](../diffusion/README.md)) |
| `results/` | results table and per-slice scores; `syndiff_val/` holds SynDiff's validation scores |
| `figures/` | alignment, preprocessing and comparison figures |
| `figures/make_slide_figures.py`, `figures/make_closeups.py` | the presentation figures in `figures/slides/` |
