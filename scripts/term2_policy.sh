#!/usr/bin/env bash
set -eo pipefail

cd /home/yifeng/aic/.claude/worktrees/yf_phase-inference
export PYTHONPATH=$PWD/aic_utils/aic_xvla
export AIC_XVLA_SERVER_URL=http://127.0.0.1:8010
export AIC_XVLA_CMD_MODE=pose
export AIC_XVLA_REPLAN=15
export AIC_XVLA_TASK_TIMEOUT_S=180

pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true -p policy:=aic_xvla.ros.RunXVLA
