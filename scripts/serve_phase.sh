#!/usr/bin/env bash
set -eo pipefail

source /home/yifeng/miniconda3/etc/profile.d/conda.sh
conda activate xvla

XVLA_REPO=/home/yifeng/workspace/X-VLA
AIC_XVLA_PKG=/home/yifeng/aic/aic_utils/aic_xvla
export PYTHONPATH=$XVLA_REPO:$AIC_XVLA_PKG
export CUDA_VISIBLE_DEVICES=0

# Download phase classifier from HF if not cached.
CLASSIFIER_PATH=/home/yifeng/aic_xvla_data/phase_classifier_v2.pt
if [ ! -f "$CLASSIFIER_PATH" ]; then
  echo "Downloading phase classifier from Hugging Face..."
  mkdir -p "$(dirname "$CLASSIFIER_PATH")"
  python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download('siyulw2025/aic-phase-classifier', 'phase_classifier_v2.pt')
import shutil
shutil.copy(path, '$CLASSIFIER_PATH')
print('Downloaded to $CLASSIFIER_PATH')
"
fi

python -m aic_xvla.serve \
  --base-model 2toINF/X-VLA-Pt \
  --checkpoints \
siyulw2025/cableholder-approach-xvla-lora,\
siyulw2025/cableholder-coarse-align-xvla-lora,\
siyulw2025/cableholder-fine-align-xvla-lora,\
siyulw2025/cableholder-insert-xvla-lora \
  --classifier "$CLASSIFIER_PATH" \
  --phase-hysteresis 5 \
  --phase-confidence 0.5
