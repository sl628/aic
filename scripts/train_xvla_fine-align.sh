#!/usr/bin/env bash
set -euo pipefail

PHASE="fine-align"

: "${XVLA_REPO:=/home/user/X-VLA}"
: "${CONDA_SH:=/home/user/miniconda3/etc/profile.d/conda.sh}"
: "${CONDA_ENV:=xvla-stable}"

set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

WT="/home/user/aic/.claude/worktrees/yf_xvla-${PHASE}"
DATA="/home/user/aic_xvla_data/${PHASE}"
OUT="${DATA}/xvla_ckpt"
mkdir -p "$OUT"

export PYTHONPATH="${XVLA_REPO}:${WT}/aic_utils/aic_xvla"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AIC_XVLA_ACTION_ENCODING="${AIC_XVLA_ACTION_ENCODING:-delta}"

accelerate launch --num_processes 1 --mixed_precision bf16 -m aic_xvla.train \
    --mode peft --models 2toINF/X-VLA-Pt \
    --train_metas_path "${DATA}/aic_train_meta.json" \
    --output_dir       "$OUT" \
    --batch_size 1 --learning_rate 5e-4 \
    --iters 20000 --save_interval 5000 \
    --wandb-project xvla-cableholder \
    --wandb-run-name "phase-${PHASE}"
