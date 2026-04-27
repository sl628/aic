#!/usr/bin/env bash
set -euo pipefail

source /home/yifeng/miniconda3/etc/profile.d/conda.sh
conda activate xvla

XVLA_REPO=/home/yifeng/workspace/X-VLA
AIC_XVLA_PKG=/home/yifeng/aic/.claude/worktrees/yf_phase-classifier/aic_utils/aic_xvla
export PYTHONPATH=$XVLA_REPO:$AIC_XVLA_PKG
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for p in 0 1 2 3; do
  echo "=== Training phase $p adapter ==="
  accelerate launch --num_processes 1 --mixed_precision bf16 -m aic_xvla.train \
    --mode peft --models 2toINF/X-VLA-Pt \
    --train_metas_path /home/yifeng/aic_xvla_data/phase_${p}_train_meta.json \
    --val_metas_path /home/yifeng/aic_xvla_data/phase_${p}_val_meta.json \
    --output_dir /home/yifeng/aic_xvla_data/xvla_phase_${p} \
    --batch_size 1 --learning_rate 5e-4 --iters 2000 --save_interval 1000
done
