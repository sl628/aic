#!/usr/bin/env bash
set -eo pipefail

# Empty policy — robot does nothing. No inference server needed.
cd /home/yifeng/aic/.claude/worktrees/yf_phase-inference
export PYTHONPATH=$PWD/aic_utils/aic_xvla

pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true -p policy:=aic_xvla.ros.EmptyPolicy
