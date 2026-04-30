#!/usr/bin/env bash
set -eo pipefail

# Eval engine with GUI. Connect to Zenoh router on localhost:7447.
docker run --rm --gpus all --network host \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  ghcr.io/intrinsic-dev/aic/aic_eval:latest \
  /entrypoint.sh ground_truth:=false start_aic_engine:=true
