#!/usr/bin/env bash
set -euo pipefail

source /home/yifeng/miniconda3/etc/profile.d/conda.sh
conda activate xvla

XVLA_REPO=/home/yifeng/workspace/X-VLA
AIC_XVLA_PKG=/home/yifeng/aic/.claude/worktrees/yf_phase-classifier/aic_utils/aic_xvla
export PYTHONPATH=$XVLA_REPO:$AIC_XVLA_PKG
export CUDA_VISIBLE_DEVICES=0

python -m aic_xvla.serve \
  --base-model 2toINF/X-VLA-Pt \
  --checkpoints \
/home/yifeng/aic_xvla_data/xvla_phase_0/ckpt-2000,\
/home/yifeng/aic_xvla_data/xvla_phase_1/ckpt-2000,\
/home/yifeng/aic_xvla_data/xvla_phase_2/ckpt-2000,\
/home/yifeng/aic_xvla_data/xvla_phase_3/ckpt-2000 \
  --classifier /home/yifeng/aic_xvla_data/phase_classifier_v2.pt
