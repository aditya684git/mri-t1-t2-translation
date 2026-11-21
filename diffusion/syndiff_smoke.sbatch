#!/bin/bash
#SBATCH --job-name=syndiff_smoke
#SBATCH --account=<your_account>
#SBATCH --partition=work1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# SynDiff smoke test on 500 training slices per contrast, exercising saving and resuming:
#   1. train 1 epoch (epoch 0), then check content.pth was written with the EMA weights in it
#   2. run again with --resume and train 1 more epoch (epoch 1)
#   3. T1->T2 inference on the 200 eval slices with the epoch 1 checkpoint, and scoring
#      (scores go to the run folder, not results/results.csv)
# Goal: prove the pipeline works end to end and measure time per epoch, not get good results.
# Submit from the repo root:  sbatch diffusion/syndiff_smoke.sbatch
#
# Both runs pass --num_epoch 2, so the cosine LR schedule and SynDiff's validation arrays are sized for the
# whole run; SYNDIFF_STOP_AT_EPOCH makes run_syndiff_train.py stop cleanly after the epoch we want. The
# wrapper logs each epoch's start with a date timestamp and the previous epoch's wall-clock time.

module load anaconda3/2023.09-0
module load cuda/12.3.0   # nvcc for SynDiff's CUDA ops; the default 13.x does not match the cu126 PyTorch
module load gcc/12.3.0    # the system c++ lacks C++20; the default gcc/13.2.0 is too new for CUDA 12.3
export CC=gcc CXX=g++
export TORCH_CUDA_ARCH_LIST="7.0;8.0"   # V100, A100
source activate mri
export PYTHONNOUSERSITE=1

REPO=$SLURM_SUBMIT_DIR
SYNDIFF=~/SynDiff
SRC=/scratch/$USER/brats/brats_t1t2
DATA=/scratch/$USER/syndiff_data/n500
RUNS=/scratch/$USER/syndiff_runs
EXP=syndiff_smoke
OUT=/scratch/$USER/syndiff_outputs/$EXP

[ -f $REPO/diffusion/make_mat.py ] || { echo "Submit from the repo root"; exit 1; }
# SynDiff compiles two CUDA ops (utils/op) on first import, which needs nvcc and ninja
command -v nvcc >/dev/null || { echo "nvcc not found: load a CUDA module above"; exit 1; }

if [ ! -f $DATA/data_test_T2.mat ]; then
  echo "=== MAKE DATA $(date) ==="
  python $REPO/diffusion/make_mat.py --data $SRC --out $DATA --n_train 500 || exit 1
fi

rm -rf "$RUNS/$EXP" "$OUT"   # fresh run every time
mkdir -p $RUNS
# content.pth stores an argparse.Namespace; torch >= 2.6 refuses to unpickle it by default
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

TRAIN_ARGS=(--image_size 256 --exp $EXP --num_channels 2 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4
  --num_timesteps 4 --num_res_blocks 2 --batch_size 1 --contrast1 T1 --contrast2 T2 --num_epoch 2 --ngf 64
  --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. --z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4
  --lazy_reg 10 --num_process_per_node 1 --save_content --save_content_every 1 --save_ckpt_every 1 --local_rank 0
  --port_num $((20000 + SLURM_JOB_ID % 20000)) --input_path $DATA --output_path $RUNS)
cd $SYNDIFF

echo "=== TRAIN run 1 (epoch 0): $(date) ==="
T0=$(date +%s)
SYNDIFF_STOP_AT_EPOCH=1 python -u $REPO/diffusion/run_syndiff_train.py train.py "${TRAIN_ARGS[@]}" \
  || { echo "run 1 failed"; exit 1; }
echo "=== run 1 done: $(date), $(( $(date +%s) - T0 )) s wall clock including startup ==="

[ -f $RUNS/$EXP/content.pth ] || { echo "content.pth was not written"; exit 1; }
python - $RUNS/$EXP/content.pth <<'EOF' || { echo "content.pth check failed"; exit 1; }
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
opts = {k: v for k, v in c.items() if k.startswith("optimizer_gen")}  # the 4 EMA-wrapped generators
ema = all(v["state"] and all("ema" in s for s in v["state"].values()) for v in opts.values())
print(f"content.pth: next epoch {c['epoch']}, EMA weights in all {len(opts)} generator optimizers: {ema}")
sys.exit(0 if c["epoch"] == 1 and len(opts) == 4 and ema else 1)
EOF

echo "=== TRAIN run 2 (--resume, epoch 1): $(date) ==="
T0=$(date +%s)
SYNDIFF_STOP_AT_EPOCH=2 python -u $REPO/diffusion/run_syndiff_train.py train.py "${TRAIN_ARGS[@]}" --resume \
  || { echo "run 2 failed"; exit 1; }
echo "=== run 2 done: $(date), $(( $(date +%s) - T0 )) s wall clock including startup ==="
[ -f $RUNS/$EXP/gen_diffusive_2_1.pth ] || { echo "No epoch 1 checkpoint after resuming"; exit 1; }

echo "=== INFER $(date) ==="
python $REPO/diffusion/infer_syndiff.py --syndiff $SYNDIFF --exp_dir $RUNS/$EXP --epoch 1 --data $DATA --out $OUT || exit 1

echo "=== EVALUATE $(date) ==="
cd $REPO
python evaluate.py --run $EXP --pred_dir $OUT --data $SRC --out_dir $OUT/eval
echo "=== DONE $(date) ==="
