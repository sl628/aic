#!/usr/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE="${ZENOH_CONFIG_OVERRIDE:-transport/shared_memory/enabled=false}"

# pi05 env only: LD_PRELOAD conda's libtiff/libjpeg so ROS (which else would
# load the system /lib/x86_64-linux-gnu/libtiff.so.6 — built against libjpeg8
# expecting symbol `jpeg12_write_raw_data` that conda-forge libjpeg-turbo
# does not export) never pulls them in first. Observed symptom in default
# eval: `import cv2` in the policy action thread crashes with
# `ImportError: libtiff.so.6: undefined symbol: jpeg12_write_raw_data`.
# Guarded on CONDA_PREFIX so the default env is not affected.
case "${CONDA_PREFIX:-}" in
  */envs/pi05)
    # Build LD_PRELOAD out of whichever of these conda libs exist. Order
    # matters: the first resolver hit wins, but LD_PRELOAD loads all of
    # them up front into the process before any ROS shared object has a
    # chance to pull in their system counterparts.
    _pi05_preload=""
    for _lib in libjpeg.so.8 libpng16.so.16 libtiff.so.6 libwebp.so.7 libopenjp2.so.7; do
      if [[ -f "${CONDA_PREFIX}/lib/${_lib}" ]]; then
        _pi05_preload="${_pi05_preload:+${_pi05_preload}:}${CONDA_PREFIX}/lib/${_lib}"
      fi
    done
    if [[ -n "${_pi05_preload}" ]]; then
      export LD_PRELOAD="${_pi05_preload}${LD_PRELOAD:+:${LD_PRELOAD}}"
    fi
    unset _pi05_preload _lib

    ;;
esac
