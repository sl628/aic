#!/usr/bin/env python3
#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""Offline XVLA embedding extraction for RLT Phase 1 pretraining.

Reads every frame from a LeRobot v3.0 dataset, runs it through the XVLA
Florence-2 encoder, and saves per-episode embedding tensors to disk.

This script runs once and its output is consumed by scripts/train.py
(--mode pretrain_rl_token) via LeRobotEmbeddingDataset.

Usage:
    pixi run python scripts/prepare_embeddings.py \\
        --data_dir /home/yifeng/aic_data \\
        --model_dir /home/yifeng/models/xvla-base \\
        --output_dir /home/yifeng/aic_data/embeddings \\
        --camera center_camera \\
        --batch_size 8

Output:
    <output_dir>/episode_0000.pt  →  {"vla_embeddings": Tensor(T, num_tokens, 1024)}
    <output_dir>/episode_0001.pt  →  ...
    ...
"""

import argparse
import io
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aic_rlt.vla.xvla_wrapper import XVLAWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_image(image_bytes: bytes, image_size: int) -> np.ndarray:
    """Decode PNG bytes from parquet → (H, W, 3) uint8 RGB resized to image_size."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def load_episode_frames(parquet_files, episode_index: int, camera: str):
    """Load all frames for one episode from parquet files.

    Returns:
        frames: list of dicts sorted by frame_index, each with:
            - frame_index (int)
            - image (bytes)
    """
    img_col = f"observation.images.{camera}"
    rows = []

    for pf in parquet_files:
        table = pq.read_table(pf, columns=["episode_index", "frame_index", img_col])
        d = table.to_pydict()
        n = len(d["episode_index"])
        for i in range(n):
            if int(d["episode_index"][i]) == episode_index:
                img_entry = d[img_col][i]
                # pyarrow stores struct<bytes,path>; extract bytes field
                img_bytes = img_entry["bytes"] if isinstance(img_entry, dict) else img_entry
                rows.append({
                    "frame_index": int(d["frame_index"][i]),
                    "image_bytes": img_bytes,
                })

    rows.sort(key=lambda r: r["frame_index"])
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Extract XVLA embeddings from aic_data")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root of LeRobot v3.0 dataset")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to XVLA model directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save per-episode .pt files")
    parser.add_argument("--camera", type=str, default="center_camera",
                        help="Camera key to use (default: center_camera)")
    parser.add_argument("--instruction", type=str,
                        default="Insert SFP cable into NIC port",
                        help="Language instruction for the task")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Frames per forward pass (default: 8)")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Resize images to this size (default: 256)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if output file already exists")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Load XVLA wrapper
    vla = XVLAWrapper(
        model_dir=args.model_dir,
        device=device,
        instruction=args.instruction,
        image_size=args.image_size,
    )

    # Collect all parquet files
    data_dir = Path(args.data_dir)
    parquet_files = sorted(data_dir.glob("data/**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}/data/")
    logger.info(f"Found {len(parquet_files)} parquet files.")

    # Discover all episode indices
    episode_indices = set()
    for pf in parquet_files:
        table = pq.read_table(pf, columns=["episode_index"])
        for ep in table["episode_index"].to_pylist():
            episode_indices.add(int(ep))
    episode_indices = sorted(episode_indices)
    logger.info(f"Found {len(episode_indices)} episodes: {episode_indices[:5]} ...")

    # Process each episode
    for ep_idx in episode_indices:
        out_path = output_dir / f"episode_{ep_idx:04d}.pt"
        if out_path.exists() and not args.overwrite:
            logger.info(f"  Episode {ep_idx:04d}: already exists, skipping.")
            continue

        logger.info(f"  Episode {ep_idx:04d}: loading frames ...")
        frames = load_episode_frames(parquet_files, ep_idx, args.camera)
        if not frames:
            logger.warning(f"  Episode {ep_idx:04d}: no frames found for camera '{args.camera}', skipping.")
            continue

        T = len(frames)
        all_embeddings = []

        # Process in batches
        for batch_start in range(0, T, args.batch_size):
            batch_frames = frames[batch_start : batch_start + args.batch_size]
            batch_imgs = [decode_image(f["image_bytes"], args.image_size)
                          for f in batch_frames]

            batch_embs = []
            for img_np in batch_imgs:
                emb = vla.get_embeddings(img_np)   # (num_tokens, 1024)
                batch_embs.append(emb)

            all_embeddings.extend(batch_embs)

            if (batch_start // args.batch_size) % 10 == 0:
                logger.info(f"    {batch_start}/{T} frames processed")

        embeddings = torch.stack(all_embeddings, dim=0)   # (T, num_tokens, 1024)
        torch.save({"vla_embeddings": embeddings}, out_path)
        logger.info(f"  Episode {ep_idx:04d}: saved {embeddings.shape} → {out_path}")

    logger.info("Embedding extraction complete.")


if __name__ == "__main__":
    main()
