#!/bin/bash
#SBATCH --job-name=syndiff_infer
#SBATCH --account=<your_account>
#SBATCH --partition=work1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# Inference on the 200 eval slices with a trained SynDiff checkpoint, then scoring.
# Usage (from the repo root):  sbatch diffusion/syndiff_infer.sbatch <exp> <epoch> <data name> <run name> [direction]
#   e.g.  sbatch diffusion/syndiff_infer.sbatch syndiff_n7514 50 n7514 diffusion_v1
#         sbatch diffusion/syndiff_infer.sbatch syndiff_n7514 30 n7514 diffusion_v1_dir1 1
# direction 2 (default): T1 -> T2 (gen_diffusive_2), scored with evaluate.py against the real T2:
#   a <run name> row in results/results.csv (tell Shrinidhi before pushing it).
# direction 1: T2 -> T1 (gen_diffusive_1, the real T2 as the source), for diagnosing the shift: checked
#   with shift_check.py --target T1 against the real T1; nothing is written to results/.
# Outputs: /scratch/$USER/syndiff_outputs/<run name>/<testA filename>.png

module load anaconda3/2023.09-0
module load cuda/12.3.0   # nvcc for SynDiff's CUDA ops; the default 13.x does not match the cu126 PyTorch
module load gcc/12.3.0    # the system c++ lacks C++20; the default gcc/13.2.0 is too new for CUDA 12.3
export CC=gcc CXX=g++
export TORCH_CUDA_ARCH_LIST="7.0;8.0"   # V100, A100
source activate mri
export PYTHONNOUSERSITE=1

EXP=$1
EPOCH=$2
NAME=$3
RUN=$4
DIRECTION=${5:-2}
if [ -z "$RUN" ]; then echo "Usage: sbatch syndiff_infer.sbatch <exp> <epoch> <data name> <run name> [direction 2|1]"; exit 1; fi
case $DIRECTION in 1|2) ;; *) echo "direction must be 2 (T1 -> T2) or 1 (T2 -> T1)"; exit 1;; esac

REPO=$SLURM_SUBMIT_DIR
SYNDIFF=~/SynDiff
SRC=/scratch/$USER/brats/brats_t1t2
DATA=/scratch/$USER/syndiff_data/$NAME
RUNS=/scratch/$USER/syndiff_runs
OUT=/scratch/$USER/syndiff_outputs/$RUN

[ -f $REPO/evaluate.py ] || { echo "Submit from the repo root"; exit 1; }
# SynDiff compiles two CUDA ops (utils/op) on first import, which needs nvcc and ninja
command -v nvcc >/dev/null || { echo "nvcc not found: load a CUDA module above"; exit 1; }

echo "=== INFER $EXP epoch $EPOCH direction $DIRECTION -> $RUN, $(date) ==="
python $REPO/diffusion/infer_syndiff.py --syndiff $SYNDIFF --exp_dir $RUNS/$EXP --epoch $EPOCH \
  --data $DATA --out $OUT --direction $DIRECTION || exit 1

cd $REPO
if [ $DIRECTION = 2 ]; then
  echo "=== EVALUATE $(date) ==="
  python evaluate.py --run $RUN --pred_dir $OUT --data $SRC
else
  echo "=== SHIFT CHECK against the real T1 $(date) ==="
  python shift_check.py --data $SRC --radius 10 --target T1 --runs $RUN=$OUT
fi
echo "=== DONE $(date) ==="
