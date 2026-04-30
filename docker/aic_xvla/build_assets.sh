#!/bin/bash
# Pre-fetch heavy artifacts for phase-aware X-VLA submission image.
# Run from the repo root before `docker compose build model`.
#
# Produces (all gitignored):
#   docker/aic_xvla/hf_cache/           (~3.3 GB) - HF cache for 2toINF/X-VLA-Pt base model
#   docker/aic_xvla/aic_xvla_ckpt/      (4 LoRA + classifier) - 4 phase adapters + phase classifier
#   docker/aic_xvla/X-VLA-src.tar       (~5 MB)   - X-VLA repo at the commit used to train
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ASSETS_DIR="$REPO_ROOT/docker/aic_xvla"
CKPT_DIR="$ASSETS_DIR/aic_xvla_ckpt"

# 1. HF base model cache (same as before)
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
        exit 1
    fi
    echo "==> Pre-fetching 2toINF/X-VLA-Pt into $ASSETS_DIR/hf_cache (~3.3 GB)..."
    mkdir -p "$ASSETS_DIR/hf_cache"
    HF_HOME="$ASSETS_DIR/hf_cache" "$XVLA_PYTHON" -c \
        "from huggingface_hub import snapshot_download; \
         snapshot_download('2toINF/X-VLA-Pt')"
fi

# 2. Phase-specific LoRA adapters + phase classifier from Hugging Face
mkdir -p "$CKPT_DIR"

download_hf_repo() {
    local repo="$1" target="$2"
    if [[ ! -d "$target" ]]; then
        echo "==> Downloading $repo -> $target..."
        mkdir -p "$target"
        HF_HOME="$ASSETS_DIR/hf_cache" "$XVLA_PYTHON" -c "
from huggingface_hub import snapshot_download
import shutil, os
src = snapshot_download('$repo')
for f in os.listdir(src):
    shutil.copy2(os.path.join(src, f), '$target')
"
    else
        echo "==> $target already exists, skipping."
    fi
}

download_hf_repo siyulw2025/cableholder-approach-xvla-lora       "$CKPT_DIR/phase_0"
download_hf_repo siyulw2025/cableholder-coarse-align-xvla-lora    "$CKPT_DIR/phase_1"
download_hf_repo siyulw2025/cableholder-fine-align-xvla-lora       "$CKPT_DIR/phase_2"
download_hf_repo siyulw2025/cableholder-insert-xvla-lora           "$CKPT_DIR/phase_3"

# Download phase classifier .pt
CLASSIFIER_TARGET="$CKPT_DIR/phase_classifier_v2.pt"
if [[ ! -f "$CLASSIFIER_TARGET" ]]; then
    echo "==> Downloading phase classifier from siyulw2025/aic-phase-classifier..."
    HF_HOME="$ASSETS_DIR/hf_cache" "$XVLA_PYTHON" -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download('siyulw2025/aic-phase-classifier', 'phase_classifier_v2.pt')
shutil.copy(path, '$CLASSIFIER_TARGET')
"
else
    echo "==> Phase classifier already present, skipping."
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
du -sh "$ASSETS_DIR/hf_cache" "$CKPT_DIR" "$ASSETS_DIR/X-VLA-src.tar"
echo
echo "Phase adapters:"
ls -d "$CKPT_DIR"/phase_*
echo "Classifier: $CLASSIFIER_TARGET"
