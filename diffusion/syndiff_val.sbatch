#!/bin/bash
#SBATCH --job-name=syndiff_val
#SBATCH --account=<your_account>
#SBATCH --partition=work1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

# Validation scoring for model selection: T1 -> T2 on the validation patients (never the 200 eval slices)
# for each saved epoch of one training run, then score_val.py (raw scores + input-based shift).
# Usage (from the repo root):  sbatch diffusion/syndiff_val.sbatch <exp> <data name> [epoch ...]
#   e.g.  sbatch diffusion/syndiff_val.sbatch sc_n500_mind_0.25 n500            # every saved epoch
#         sbatch diffusion/syndiff_val.sbatch syndiff_n500 n500 30               # just epoch 30
# Works for baseline runs (syndiff_<data>) and struct-loss runs (sc_...) alike.
# Rows <exp>_e<epoch> go to results/syndiff_val/val_results.csv; the job ends by printing the ranking of
# every row in that file by the selection rule (score_val.py --select, README section 7).
# Outputs: /scratch/$USER/syndiff_val_outputs/<exp>_e<epoch>/

module load anaconda3/2023.09-0
module load cuda/12.3.0   # nvcc for SynDiff's CUDA ops; the default 13.x does not match the cu126 PyTorch
module load gcc/12.3.0    # the system c++ lacks C++20; the default gcc/13.2.0 is too new for CUDA 12.3
export CC=gcc CXX=g++
export TORCH_CUDA_ARCH_LIST="7.0;8.0"   # V100, A100
source activate mri
export PYTHONNOUSERSITE=1

EXP=$1
NAME=$2
shift 2
if [ -z "$NAME" ]; then echo "Usage: sbatch syndiff_val.sbatch <exp> <data name> [epoch ...]"; exit 1; fi

REPO=$SLURM_SUBMIT_DIR
SYNDIFF=~/SynDiff
SRC=/scratch/$USER/brats/brats_t1t2
DATA=/scratch/$USER/syndiff_data/$NAME
RUNS=/scratch/$USER/syndiff_runs
OUTS=/scratch/$USER/syndiff_val_outputs

[ -f $REPO/diffusion/score_val.py ] || { echo "Submit from the repo root"; exit 1; }
[ -f $DATA/data_val_T1.mat ] || { echo "No validation data in $DATA"; exit 1; }
command -v nvcc >/dev/null || { echo "nvcc not found: load a CUDA module above"; exit 1; }

EPOCHS="$*"
if [ -z "$EPOCHS" ]; then  # every saved checkpoint, in order
  EPOCHS=$(ls $RUNS/$EXP/gen_diffusive_2_*.pth 2>/dev/null | sed -E 's/.*_([0-9]+)\.pth/\1/' | sort -n | tr '\n' ' ')
fi
[ -n "$EPOCHS" ] || { echo "No gen_diffusive_2_*.pth in $RUNS/$EXP"; exit 1; }
echo "=== $EXP on $NAME's validation patients, epochs: $EPOCHS ==="

cd $REPO
for EPOCH in $EPOCHS; do
  RUN=${EXP}_e$EPOCH
  echo "=== $RUN $(date) ==="
  python diffusion/infer_syndiff.py --syndiff $SYNDIFF --exp_dir $RUNS/$EXP --epoch $EPOCH --data $DATA \
    --out $OUTS/$RUN --phase val || exit 1
  python diffusion/score_val.py --run $RUN --pred_dir $OUTS/$RUN --syndata $DATA --data $SRC || exit 1
done

echo
echo "=== Ranking of every row in results/syndiff_val/val_results.csv $(date) ==="
python diffusion/score_val.py --select
echo "=== DONE $(date) ==="
