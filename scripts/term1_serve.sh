#!/usr/bin/env bash
set -eo pipefail

source /home/yifeng/miniconda3/etc/profile.d/conda.sh
conda activate xvla

cd /home/yifeng/aic/.claude/worktrees/yf_phase-inference
export PYTHONPATH=/home/yifeng/workspace/X-VLA:$PWD/aic_utils/aic_xvla
export CUDA_VISIBLE_DEVICES=0

python -m aic_xvla.serve \
  --base-model 2toINF/X-VLA-Pt \
  --checkpoints siyulw2025/cableholder-approach-xvla-lora,siyulw2025/cableholder-coarse-align-xvla-lora,siyulw2025/cableholder-fine-align-xvla-lora,siyulw2025/cableholder-insert-xvla-lora \
  --classifier /home/yifeng/aic_xvla_data/phase_classifier_v2.pt \
  --phase-hysteresis 5 \
  --phase-confidence 0.5
