#!/usr/bin/env bash
set -eo pipefail

source /home/yifeng/aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_ROUTER_CHECK_ATTEMPTS=-1
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true;transport/shared_memory/transport_optimization/pool_size=536870912'

ros2 run rmw_zenoh_cpp rmw_zenohd
