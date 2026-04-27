#!/bin/bash
# entrypoint for aic_xvla submission image.
#
# 1) Launch X-VLA inference FastAPI server in background (conda XVLA env).
# 2) Wait for /health.
# 3) Set RunXVLA env vars matching the validated 63.5-score recipe.
# 4) Replicate docker/aic_model/Dockerfile's Zenoh wiring verbatim.
# 5) exec the ROS aic_model node with policy=aic_xvla.ros.RunXVLA.
set -e

# ----- 1. Background X-VLA inference server -----
export HF_HOME=/opt/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH=/opt/X-VLA:/ws_aic/src/aic/aic_utils/aic_xvla
export AIC_XVLA_ACTION_ENCODING=delta  # match training; eval.py auto-detects via sidecar

mkdir -p /tmp/xvla
/opt/conda/envs/XVLA/bin/python -m aic_xvla.serve \
    --checkpoint /opt/aic_xvla_ckpt \
    --host 127.0.0.1 --port 8010 \
    > /tmp/xvla/serve.log 2>&1 &
SERVER_PID=$!

# ----- 2. Wait for /health (max 180s — first model load on cold GPU) -----
echo "waiting for X-VLA server to become ready..."
for i in {1..90}; do
    if curl -sf http://127.0.0.1:8010/health >/dev/null 2>&1; then
        echo "X-VLA server ready after ${i}x2s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "X-VLA server died during startup. Logs:"
        cat /tmp/xvla/serve.log
        exit 1
    fi
    sleep 2
done
if ! curl -sf http://127.0.0.1:8010/health >/dev/null 2>&1; then
    echo "X-VLA server did not become healthy within 180s. Logs:"
    cat /tmp/xvla/serve.log
    exit 1
fi

# ----- 3. RunXVLA configuration (matches the validated 94-score ckpt-20000 dev run) -----
export AIC_XVLA_SERVER_URL=http://127.0.0.1:8010
export AIC_XVLA_CMD_MODE=pose
export AIC_XVLA_REPLAN=1
export AIC_XVLA_TASK_TIMEOUT_S=60

# ----- 4. Zenoh wiring -----
# Local docker-compose sets AIC_ROUTER_ADDR; submission portal sets
# AIC_MODEL_ROUTER_ADDR per docs/custom_dockerfile.md. Accept either.
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

ROUTER_ADDR="${AIC_MODEL_ROUTER_ADDR:-${AIC_ROUTER_ADDR:-}}"
if [[ -z "$ROUTER_ADDR" ]]; then
    echo "Neither AIC_MODEL_ROUTER_ADDR nor AIC_ROUTER_ADDR is set"
    exit 1
fi

should_enable_acl() {
    [[ "$AIC_ENABLE_ACL" == "true" || "$AIC_ENABLE_ACL" == "1" ]]
}

if should_enable_acl; then
    if [[ -z "$AIC_MODEL_PASSWD" ]]; then
        echo "AIC_MODEL_PASSWD must be provided when ACL is enabled"
        exit 1
    fi
    echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
fi

ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/'"$ROUTER_ADDR"'"]'
ZENOH_CONFIG_OVERRIDE+=';transport/shared_memory/enabled=false'
if should_enable_acl; then
    ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/user="model"'
    ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/password="'"$AIC_MODEL_PASSWD"'"'
    ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/dictionary_file="/credentials.txt"'
fi
export ZENOH_CONFIG_OVERRIDE
echo "ZENOH_CONFIG_OVERRIDE=$ZENOH_CONFIG_OVERRIDE"

# ----- 5. Hand off to aic_model ROS node -----
exec pixi run --as-is ros2 run aic_model aic_model "$@"
