#!/usr/bin/env bash
set -eo pipefail

# Run the AIC eval in Docker with Gazebo GUI + RViz visible on your desktop.
# Connect model container in Docker OR run term2_policy.sh locally.
# The eval container starts its own Zenoh router (port 7447).

docker compose -f /home/yifeng/aic/.claude/worktrees/yf_phase-inference/docker/docker-compose.yaml up \
  --abort-on-container-exit
