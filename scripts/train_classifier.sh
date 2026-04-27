#!/usr/bin/env bash
set -euo pipefail

cd /home/yifeng/aic/.claude/worktrees/yf_phase-classifier

PYTHONUNBUFFERED=1 PYTHONPATH=$PWD/aic_utils/aic_xvla \
/home/yifeng/aic/.pixi/envs/default/bin/python3 -m aic_xvla.train_phase_classifier \
  --epochs 15 --batch-size 64 --num-workers 4 \
  --subsample-p3 --target-p3-ratio 0.3 --class-weights \
  --wandb-project aic-phase-classifier --wandb-run-name resnet18-balanced-v2
