#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=<your_account>
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/test_%j.out

# Run a trained model on EVERY test slice (testA -> fake T2).
# Usage:   sbatch run_test.sbatch cut_v1
#          sbatch run_test.sbatch cyclegan_v1
# The run name must start with "cut" or "cyclegan" so the right repo is used.
# Outputs: /scratch/$USER/mri_runs/results/<run>/test_latest/images/

RUN=$1
if [ -z "$RUN" ]; then echo "Usage: sbatch run_test.sbatch <run name>"; exit 1; fi

module load anaconda3/2023.09-0
source activate mri
export PYTHONNOUSERSITE=1

DATA=/scratch/$USER/brats/brats_t1t2
RUNS=/scratch/$USER/mri_runs
NTEST=$(ls $DATA/testA | wc -l)

case $RUN in
  cut*)      cd ~/mri-project/contrastive-unpaired-translation; MODEL="--CUT_mode CUT" ;;
  cyclegan*) cd ~/mri-project/pytorch-CycleGAN-and-pix2pix;     MODEL="--model cycle_gan" ;;
  *) echo "Run name must start with cut or cyclegan"; exit 1 ;;
esac

echo "=== TEST $RUN on $NTEST slices, $(date) ==="
python test.py \
  --dataroot $DATA --name $RUN $MODEL \
  --input_nc 1 --output_nc 1 --load_size 256 --crop_size 256 \
  --phase test --num_test $NTEST --epoch latest \
  --checkpoints_dir $RUNS/checkpoints --results_dir $RUNS/results
echo "=== DONE $(date) ==="
