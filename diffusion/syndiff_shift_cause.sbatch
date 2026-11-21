#!/bin/bash
#SBATCH --job-name=syndiff_shift_cause
#SBATCH --account=<your_account>
#SBATCH --partition=work1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# Where does SynDiff's output shift come from? Runs all four generators of one checkpoint on the 200 eval
# slices and measures each one's shift with shift_check.py against the real image of its output contrast:
#   gen_diffusive_2          T1 -> T2   vs real T2
#   gen_non_diffusive_1to2   T1 -> T2   vs real T2
#   gen_diffusive_1          T2 -> T1   vs real T1
#   gen_non_diffusive_2to1   T2 -> T1   vs real T1
# If the translation networks learned opposite shifts that cancel in the cycle (README section 7), both
# T1 -> T2 generators show the known shift s and both T2 -> T1 generators show -s.
# Usage (from the repo root):  sbatch diffusion/syndiff_shift_cause.sbatch <exp> <epoch> <data name>
#   e.g.  sbatch diffusion/syndiff_shift_cause.sbatch syndiff_n7514 30 n7514
# Diagnostic only: nothing is written to results/.

module load anaconda3/2023.09-0
module load cuda/12.3.0   # nvcc for SynDiff's CUDA ops; the default 13.x does not match the cu126 PyTorch
module load gcc/12.3.0    # the system c++ lacks C++20; the default gcc/13.2.0 is too new for CUDA 12.3
export CC=gcc CXX=g++
export TORCH_CUDA_ARCH_LIST="7.0;8.0"   # V100, A100
source activate mri
export PYTHONNOUSERSITE=1
set -o pipefail   # a failing python before "| tee" fails the job

EXP=$1
EPOCH=$2
NAME=$3
if [ -z "$NAME" ]; then echo "Usage: sbatch syndiff_shift_cause.sbatch <exp> <epoch> <data name>"; exit 1; fi

REPO=$SLURM_SUBMIT_DIR
SYNDIFF=~/SynDiff
SRC=/scratch/$USER/brats/brats_t1t2
DATA=/scratch/$USER/syndiff_data/$NAME
RUNS=/scratch/$USER/syndiff_runs
OUTS=/scratch/$USER/syndiff_outputs/shift_cause/${EXP}_e$EPOCH
RADIUS=10   # covers the 8 px seen so far

[ -f $REPO/shift_check.py ] || { echo "Submit from the repo root"; exit 1; }
command -v nvcc >/dev/null || { echo "nvcc not found: load a CUDA module above"; exit 1; }
mkdir -p $OUTS
SUMMARY=$OUTS/summary.txt
: > $SUMMARY

for SPEC in diffusive:2 non_diffusive:2 diffusive:1 non_diffusive:1; do
  GEN=${SPEC%%:*}; DIRECTION=${SPEC##*:}
  if [ $DIRECTION = 2 ]; then TARGET=T2; WHAT="T1->T2"; else TARGET=T1; WHAT="T2->T1"; fi
  RUN=${GEN}_dir${DIRECTION}
  echo "=== $GEN $WHAT, $(date) ==="
  python $REPO/diffusion/infer_syndiff.py --syndiff $SYNDIFF --exp_dir $RUNS/$EXP --epoch $EPOCH --data $DATA \
    --out $OUTS/$RUN --direction $DIRECTION --generator $GEN || exit 1
  (cd $REPO && python shift_check.py --data $SRC --radius $RADIUS --target $TARGET --runs "$RUN ($WHAT)=$OUTS/$RUN") \
    | tee -a $SUMMARY || exit 1
done

echo
echo "=== $EXP epoch $EPOCH: best shift (dy, dx) of each generator, $(date) ==="
grep -E "vs real|best shift" $SUMMARY | paste - - | sed -E 's/: brain SSIM by shift.*best shift/: best shift/'
echo "Hypothesis: both T1->T2 rows show the same shift s, both T2->T1 rows show -s."
echo "Full grids: $SUMMARY"
