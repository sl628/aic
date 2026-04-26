#!/bin/bash
# Pre-fetch heavy artifacts that get COPY'd into the Docker image.
# Run from the repo root before `docker compose build model`.
#
# Produces (all gitignored):
#   docker/aic_xvla/hf_cache/         (~3.3 GB) - HF cache for 2toINF/X-VLA-Pt base model
#   docker/aic_xvla/aic_xvla_ckpt/    (46 MB)   - LoRA adapter ckpt-3000 + sidecar
#   docker/aic_xvla/X-VLA-src.tar     (~5 MB)   - X-VLA repo at the commit used to train
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ASSETS_DIR="$REPO_ROOT/docker/aic_xvla"

# 1. HF base model cache
# Prefer copying from existing host cache (~/.cache/huggingface). Fall back to
# fresh download via host XVLA python if missing.
HOST_HF_CACHE="${HOST_HF_CACHE:-$HOME/.cache/huggingface}"
XVLA_PYTHON="${XVLA_PYTHON:-$HOME/miniconda3/envs/XVLA/bin/python}"
TARGET_MODEL_DIR="$ASSETS_DIR/hf_cache/hub/models--2toINF--X-VLA-Pt"
if [[ -d "$TARGET_MODEL_DIR" ]]; then
    echo "==> HF cache already present at $ASSETS_DIR/hf_cache, skipping."
elif [[ -d "$HOST_HF_CACHE/hub/models--2toINF--X-VLA-Pt" ]]; then
    echo "==> Copying HF cache from $HOST_HF_CACHE/hub/models--2toINF--X-VLA-Pt (~3.3 GB)..."
    mkdir -p "$ASSETS_DIR/hf_cache/hub"
    cp -r "$HOST_HF_CACHE/hub/models--2toINF--X-VLA-Pt" "$ASSETS_DIR/hf_cache/hub/"
else
    if [[ ! -x "$XVLA_PYTHON" ]]; then
        echo "ERROR: no host HF cache and XVLA python not found at $XVLA_PYTHON"
        echo "Either populate $HOST_HF_CACHE/hub/models--2toINF--X-VLA-Pt manually,"
        echo "or set XVLA_PYTHON env var to a python with huggingface_hub installed."
        exit 1
    fi
    echo "==> Pre-fetching 2toINF/X-VLA-Pt into $ASSETS_DIR/hf_cache (~3.3 GB)..."
    echo "    using $XVLA_PYTHON"
    mkdir -p "$ASSETS_DIR/hf_cache"
    HF_HOME="$ASSETS_DIR/hf_cache" "$XVLA_PYTHON" -c \
        "from huggingface_hub import snapshot_download; \
         snapshot_download('2toINF/X-VLA-Pt')"
fi

# 2. LoRA ckpt
CKPT_SRC="${AIC_XVLA_CKPT_SRC:-/home/yifeng/aic_xvla_overfit_abs/ckpt-3000}"
SIDECAR_SRC="${AIC_XVLA_META_SRC:-/home/yifeng/aic_xvla_overfit_abs/aic_xvla_meta.json}"
if [[ ! -d "$ASSETS_DIR/aic_xvla_ckpt" ]]; then
    echo "==> Copying LoRA ckpt from $CKPT_SRC..."
    mkdir -p "$ASSETS_DIR/aic_xvla_ckpt"
    cp -r "$CKPT_SRC"/. "$ASSETS_DIR/aic_xvla_ckpt/"
    cp "$SIDECAR_SRC" "$ASSETS_DIR/aic_xvla_ckpt/"
else
    echo "==> LoRA ckpt already present at $ASSETS_DIR/aic_xvla_ckpt, skipping."
fi

# 3. X-VLA source archive (pinned to current HEAD of ~/workspace/X-VLA)
XVLA_REPO="${XVLA_REPO:-/home/yifeng/workspace/X-VLA}"
if [[ ! -f "$ASSETS_DIR/X-VLA-src.tar" ]]; then
    echo "==> Archiving $XVLA_REPO@HEAD to $ASSETS_DIR/X-VLA-src.tar..."
    git -C "$XVLA_REPO" archive --format=tar HEAD > "$ASSETS_DIR/X-VLA-src.tar"
    echo "    (commit: $(git -C "$XVLA_REPO" rev-parse HEAD))"
else
    echo "==> X-VLA src archive already present, skipping."
fi

echo
echo "All assets ready. Sizes:"
du -sh "$ASSETS_DIR/hf_cache" "$ASSETS_DIR/aic_xvla_ckpt" "$ASSETS_DIR/X-VLA-src.tar"
