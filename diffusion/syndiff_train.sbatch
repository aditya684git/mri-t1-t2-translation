#!/bin/bash
#SBATCH --job-name=syndiff_train
#SBATCH --account=<your_account>
#SBATCH --partition=work1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out

# Full SynDiff training, T1 <-> T2, with the settings from SynDiff's README.
# Usage (from the repo root):  sbatch diffusion/syndiff_train.sbatch <data name> <epochs> [exp name]
#   e.g.  sbatch diffusion/syndiff_train.sbatch n7514 50
# Data from /scratch/$USER/syndiff_data/<data name> (make_mat.py), output to
# /scratch/$USER/syndiff_runs/<exp> (default exp: syndiff_<data name>).
#
# Resume: resubmit the SAME command (same epochs: the cosine LR schedule is built for it). If content.pth
# exists the run continues from the last save, losing at most SAVE_EVERY epochs.
# Known SynDiff quirks: epochs run 0..<epochs> inclusive, and the final validation pass always crashes
# with an IndexError after the last checkpoint is saved. val_psnr_values.npy restarts on resume.

module load anaconda3/2023.09-0
module load cuda/12.3.0   # nvcc for SynDiff's CUDA ops; the default 13.x does not match the cu126 PyTorch
module load gcc/12.3.0    # the system c++ lacks C++20; the default gcc/13.2.0 is too new for CUDA 12.3
export CC=gcc CXX=g++
export TORCH_CUDA_ARCH_LIST="7.0;8.0"   # V100, A100
source activate mri
export PYTHONNOUSERSITE=1

NAME=$1
EPOCHS=$2
EXP=${3:-syndiff_$NAME}
SAVE_EVERY=5
if [ -z "$NAME" ] || [ -z "$EPOCHS" ]; then echo "Usage: sbatch syndiff_train.sbatch <data name> <epochs> [exp]"; exit 1; fi
if [ $((EPOCHS % SAVE_EVERY)) -ne 0 ]; then echo "<epochs> must be a multiple of $SAVE_EVERY"; exit 1; fi

REPO=$SLURM_SUBMIT_DIR
SYNDIFF=~/SynDiff
DATA=/scratch/$USER/syndiff_data/$NAME
RUNS=/scratch/$USER/syndiff_runs
mkdir -p $RUNS

[ -f $REPO/diffusion/run_syndiff_train.py ] || { echo "Submit from the repo root"; exit 1; }
[ -f $DATA/data_train_T1.mat ] || { echo "No data in $DATA: run make_mat.py first"; exit 1; }
# SynDiff compiles two CUDA ops (utils/op) on first import, which needs nvcc and ninja
command -v nvcc >/dev/null || { echo "nvcc not found: load a CUDA module above"; exit 1; }

RESUME=""
if [ -f $RUNS/$EXP/content.pth ]; then
  RESUME="--resume"
  echo "Resuming from $RUNS/$EXP/content.pth"
fi
# content.pth stores an argparse.Namespace; torch >= 2.6 refuses to unpickle it by default
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "=== TRAIN $EXP on $NAME, $EPOCHS epochs, $(date) ==="
cd $SYNDIFF
python -u $REPO/diffusion/run_syndiff_train.py train.py --image_size 256 --exp $EXP --num_channels 2 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 \
  --num_timesteps 4 --num_res_blocks 2 --batch_size 1 --contrast1 T1 --contrast2 T2 --num_epoch $EPOCHS --ngf 64 \
  --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. --z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 \
  --lazy_reg 10 --num_process_per_node 1 --save_content --save_content_every $SAVE_EVERY \
  --save_ckpt_every $SAVE_EVERY --local_rank 0 --port_num $((20000 + SLURM_JOB_ID % 20000)) \
  --input_path $DATA --output_path $RUNS $RESUME
echo "train.py exit code $?"

if [ -f $RUNS/$EXP/gen_diffusive_2_$EPOCHS.pth ]; then
  echo "Finished: final checkpoint $RUNS/$EXP/gen_diffusive_2_$EPOCHS.pth"
  echo "(an IndexError just above is the known SynDiff bug in its last validation pass)"
else
  echo "Not finished. Saved checkpoints:"; ls $RUNS/$EXP/gen_diffusive_2_*.pth 2>/dev/null
  echo "Resubmit the same command to resume."
fi
echo "=== DONE $(date) ==="
