#!/bin/bash
#SBATCH --job-name=cyclegan_smoke
#SBATCH --account=<your_account>
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out

# CycleGAN smoke test: same data and settings as the CUT smoke test.
# Goal: prove the pipeline works and measure speed, not get good results.

module load anaconda3/2023.09-0
source activate mri
export PYTHONNOUSERSITE=1

DATA=/scratch/$USER/brats/brats_t1t2
RUNS=/scratch/$USER/mri_runs
mkdir -p $RUNS/checkpoints $RUNS/results

cd ~/mri-project/pytorch-CycleGAN-and-pix2pix

echo "=== TRAIN $(date) ==="
python train.py \
  --dataroot $DATA --name cyclegan_smoke --model cycle_gan \
  --input_nc 1 --output_nc 1 --load_size 256 --crop_size 256 \
  --max_dataset_size 500 --n_epochs 1 --n_epochs_decay 1 \
  --save_epoch_freq 1 --print_freq 50 --num_threads 4 \
  --checkpoints_dir $RUNS/checkpoints

echo "=== TEST $(date) ==="
python test.py \
  --dataroot $DATA --name cyclegan_smoke --model cycle_gan \
  --input_nc 1 --output_nc 1 --load_size 256 --crop_size 256 \
  --phase test --num_test 20 \
  --checkpoints_dir $RUNS/checkpoints --results_dir $RUNS/results

echo "=== DONE $(date) ==="
