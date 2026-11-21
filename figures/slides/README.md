# Slide figures

Every figure here is a PNG (300 dpi) and an SVG with the same name. To regenerate them, run these from the repo root:

```bash
python figures/make_slide_figures.py   # the seven charts, read from results/ (runs anywhere)
python figures/make_closeups.py        # close-ups, needs the model outputs (run on Palmetto)
```

`make_slide_figures.py` reads every number from `results/results.csv`, `results/*_eval_per_slice.csv`,
`results/shift_cause_*_e30.txt` and `results/syndiff_val/val_results.csv`. If a result changes, rerun it.

**Colours.** Each model keeps one colour in every figure, from the colour-blind-safe Okabe–Ito palette:

| Model | Colour |
|---|---|
| CUT | orange |
| CycleGAN | blue |
| SynDiff original | vermillion |
| SynDiff original, aligned | reddish purple |
| SynDiff + silhouette loss | bluish green |

Runs that are not one of these models are grey.

| Figure | What it shows | Takeaway |
|---|---|---|
| `main_results` | Brain SSIM, PSNR and MAE on the 200 eval slices for CUT, CycleGAN, SynDiff original (raw and aligned) and SynDiff + silhouette loss. Error bars are 95% CIs over the 20 patient means. | With the silhouette loss, SynDiff scores best on all three metrics with no post-processing, while the original SynDiff is only competitive after alignment. |
| `shift_heatmaps` | Brain SSIM of the T1 → T2 diffusive generator's outputs when moved by each (dy, dx) within ±10 px, original vs fixed, on one colour scale. ○ marks the peak and + the centre. | The original model's best match is 8 px from where it should be; the fixed model's best match is exactly at the centre. |
| `generator_shifts` | Best alignment shift of all four SynDiff generators (diffusive and translation, both directions), original vs fixed, at full data, epoch 30. | The two directions carry equal and opposite shifts, (+2, −8) and (−2, +8), which cancel in the cycle loss; the silhouette loss brings all four to (0, 0). |
| `shift_drift_n500` | On the n500 validation patients: the median input-based shift (dx) and the validation SSIM over epochs, baseline vs silhouette loss, plus the baseline after alignment. Epoch 0 is left out because the model is untrained. | Without the loss the output drifts off-centre as training goes on and raw SSIM stalls; with it the shift stays at 0 and SSIM keeps rising, above even the aligned baseline. |
| `data_size` | Eval SSIM against the number of training slices (500, 2,000, 7,514): SynDiff original after alignment vs SynDiff + silhouette loss, with CycleGAN as a dashed reference. | The fix helps at every data size, most with the least data, and only the fixed model at full data passes CycleGAN. |
| `per_patient` | Patient-mean SSIM for CycleGAN and SynDiff + silhouette loss, one line per test patient. Lines are green when the patient improved and vermillion dashed when it got worse. The title's Wilcoxon p is computed by the script. | The gain over CycleGAN is consistent across patients, not driven by a few (17/20 improved). PSNR and MAE are slightly better on average but not significant (diffusion/README.md, section 7). |
| `loss_sweep` | The best raw validation SSIM (epoch ≥ 1) of each n500 run, with a dashed line at the baseline's best aligned score (the selection rule's quality floor). | Both silhouette weights clear the floor and λ = 0.5 is best; MIND removes the shift but breaks the translation (negative SSIM). |
| `closeup_<slice>` (made on Palmetto) | A cropped region of one slice: input T1, CUT, CycleGAN, SynDiff original, SynDiff + silhouette loss and real T2. Below them, each model's \|error\| inside the brain mask on one scale, titled with its slice SSIM, and a thumbnail showing where the crop is. | Shows where each model's errors are. A shift like the original SynDiff's shows up as a bright band along one side of the brain edge in its error map; the fixed model should have none. |

**Close-ups.** By default these are made for three slices: `BraTS2021_00271_s084` (tumour),
`BraTS2021_00359_s082` (ventricles) and `BraTS2021_01509_s117` (top slice).
- Set the crop for all slices with `--box Y0 X0 H W`, or give one slice its own crop as
  `--files NAME:Y0,X0,H,W`.
- The output folders default to the ones in `diffusion/README.md`; override them with `--cut`, `--cyclegan`,
  `--original` and `--fixed`.
