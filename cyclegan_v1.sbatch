#!/bin/bash
#SBATCH --job-name=cyclegan_v1
#SBATCH --account=<your_account>
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out

# Full CycleGAN run: T1 <-> T2 on all preprocessed training data.
# Budget matches cut_v1: 30 epochs at full LR + 30 decaying to zero = 60 epochs.
# Expected ~20 h on a V100. Checkpoints every 5 epochs (+ "latest" every 5000 iters).
# To resume after a crash: add  --continue_train --epoch_count <next epoch>  and resubmit.
# Note: this repo version has no --display_id flag (visdom was replaced by wandb, which stays off).

module load anaconda3/2023.09-0
source activate mri
export PYTHONNOUSERSITE=1

DATA=/scratch/$USER/brats/brats_t1t2
RUNS=/scratch/$USER/mri_runs
mkdir -p $RUNS/checkpoints

cd ~/mri-project/pytorch-CycleGAN-and-pix2pix

echo "=== TRAIN $(date) ==="
python train.py \
  --dataroot $DATA --name cyclegan_v1 --model cycle_gan \
  --input_nc 1 --output_nc 1 --load_size 256 --crop_size 256 \
  --n_epochs 30 --n_epochs_decay 30 \
  --save_epoch_freq 5 --print_freq 500 --num_threads 4 \
  --checkpoints_dir $RUNS/checkpoints
echo "=== DONE $(date) ==="
