#!/bin/bash
#SBATCH --job-name=syndiff_sc_smoke
#SBATCH --account=<your_account>
#SBATCH --partition=work1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

# Smoke test of syndiff_train_sc.py on 500 training slices, for both loss types (silhouette 0.5, mind 0.25):
#   1. train epoch 0; check the struct terms were logged (finite) and content.pth records the loss
#   2. resume with --resume for epoch 1
#   3. T1 -> T2 on the validation patients with the epoch 1 checkpoint, scored by score_val.py
#      (scores go to the smoke folder, not results/)
# Then, once, every generator/direction of infer_syndiff.py on a few eval slices, through shift_check.py
# (the code paths syndiff_shift_cause.sbatch uses). Goal: the pipeline works, and time per epoch with each
# loss (compare with syndiff_smoke.sbatch's), not good results.
# Submit from the repo root:  sbatch diffusion/syndiff_sc_smoke.sbatch

module load anaconda3/2023.09-0
module load cuda/12.3.0   # nvcc for SynDiff's CUDA ops; the default 13.x does not match the cu126 PyTorch
module load gcc/12.3.0    # the system c++ lacks C++20; the default gcc/13.2.0 is too new for CUDA 12.3
export CC=gcc CXX=g++
export TORCH_CUDA_ARCH_LIST="7.0;8.0"   # V100, A100
source activate mri
export PYTHONNOUSERSITE=1
set -o pipefail   # a failing python before "| tee" fails the job

REPO=$SLURM_SUBMIT_DIR
SYNDIFF=~/SynDiff
SRC=/scratch/$USER/brats/brats_t1t2
DATA=/scratch/$USER/syndiff_data/n500
RUNS=/scratch/$USER/syndiff_runs
SMOKE=/scratch/$USER/syndiff_outputs/sc_smoke

[ -f $REPO/diffusion/syndiff_train_sc.py ] || { echo "Submit from the repo root"; exit 1; }
command -v nvcc >/dev/null || { echo "nvcc not found: load a CUDA module above"; exit 1; }
if [ ! -f $DATA/data_test_T2.mat ]; then
  echo "=== MAKE DATA $(date) ==="
  python $REPO/diffusion/make_mat.py --data $SRC --out $DATA --n_train 500 || exit 1
fi
# content.pth stores an argparse.Namespace; torch >= 2.6 refuses to unpickle it by default
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
rm -rf $SMOKE; mkdir -p $SMOKE $RUNS

for SPEC in silhouette:0.5 mind:0.25; do
  LOSS=${SPEC%%:*}; LAMBDA=${SPEC##*:}; EXP=sc_smoke_$LOSS; LOG=$SMOKE/$LOSS
  rm -rf "$RUNS/$EXP"; mkdir -p $LOG
  TRAIN_ARGS=(--image_size 256 --exp $EXP --num_channels 2 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4
    --num_timesteps 4 --num_res_blocks 2 --batch_size 1 --contrast1 T1 --contrast2 T2 --num_epoch 2 --ngf 64
    --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. --z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4
    --lazy_reg 10 --num_process_per_node 1 --save_content --save_content_every 1 --save_ckpt_every 1 --local_rank 0
    --port_num $((20000 + SLURM_JOB_ID % 20000)) --input_path $DATA --output_path $RUNS
    --struct_loss $LOSS --lambda_struct $LAMBDA)
  cd $SYNDIFF

  echo "=== $LOSS (lambda $LAMBDA) run 1 (epoch 0): $(date) ==="
  SYNDIFF_STOP_AT_EPOCH=1 python -u $REPO/diffusion/syndiff_train_sc.py "${TRAIN_ARGS[@]}" 2>&1 | tee $LOG/run1.log \
    || { echo "$LOSS run 1 failed"; exit 1; }
  N=$(grep -c "struct $LOSS" $LOG/run1.log)
  grep "struct $LOSS" $LOG/run1.log | grep -qiE "nan|inf" && { echo "non-finite struct term"; exit 1; }
  [ "$N" -ge 1 ] || { echo "no struct-loss lines in the log"; exit 1; }
  echo "$N struct-loss lines logged; first and last:"; grep "struct $LOSS" $LOG/run1.log | sed -n '1p;$p'
  python - $RUNS/$EXP/content.pth $LOSS <<'EOF' || { echo "content.pth check failed"; exit 1; }
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
a = c["args"]
print(f"content.pth: next epoch {c['epoch']}, struct_loss {a.struct_loss}, lambda {a.lambda_struct}")
sys.exit(0 if c["epoch"] == 1 and a.struct_loss == sys.argv[2] else 1)
EOF

  echo "=== $LOSS run 2 (--resume, epoch 1): $(date) ==="
  SYNDIFF_STOP_AT_EPOCH=2 python -u $REPO/diffusion/syndiff_train_sc.py "${TRAIN_ARGS[@]}" --resume 2>&1 \
    | tee $LOG/run2.log || { echo "$LOSS run 2 failed"; exit 1; }
  [ -f $RUNS/$EXP/gen_diffusive_2_1.pth ] || { echo "No epoch 1 checkpoint after resuming"; exit 1; }

  echo "=== $LOSS validation inference + scoring: $(date) ==="
  cd $REPO
  python diffusion/infer_syndiff.py --syndiff $SYNDIFF --exp_dir $RUNS/$EXP --epoch 1 --data $DATA \
    --out $LOG/val --phase val || exit 1
  python diffusion/score_val.py --run $EXP --pred_dir $LOG/val --syndata $DATA --data $SRC --out_dir $LOG/eval || exit 1
done

echo "=== every generator and direction, 5 eval slices each: $(date) ==="
for SPEC in diffusive:2 non_diffusive:2 diffusive:1 non_diffusive:1; do
  GEN=${SPEC%%:*}; DIRECTION=${SPEC##*:}; TARGET=$([ $DIRECTION = 2 ] && echo T2 || echo T1)
  python diffusion/infer_syndiff.py --syndiff $SYNDIFF --exp_dir $RUNS/sc_smoke_mind --epoch 1 --data $DATA \
    --out $SMOKE/${GEN}_dir$DIRECTION --direction $DIRECTION --generator $GEN || exit 1
  python shift_check.py --data $SRC --radius 1 --n 5 --target $TARGET --runs ${GEN}_dir$DIRECTION=$SMOKE/${GEN}_dir$DIRECTION \
    | grep "best shift" || exit 1
done

echo
echo "=== time per epoch (training + saving + validation) ==="
grep -H "took" $SMOKE/*/run*.log
echo "=== DONE $(date) ==="
