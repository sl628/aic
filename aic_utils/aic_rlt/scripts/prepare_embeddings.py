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
from aic_rlt.trainer import PHASE_NAMES, DEFAULT_PHASE_PROMPTS

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
            - image_bytes (bytes)
            - prop (np.ndarray, 26D) — observation.state
    """
    img_col = f"observation.images.{camera}"
    rows = []

    for pf in parquet_files:
        table = pq.read_table(
            pf,
            columns=["episode_index", "frame_index", img_col, "observation.state"],
        )
        d = table.to_pydict()
        n = len(d["episode_index"])
        for i in range(n):
            if int(d["episode_index"][i]) == episode_index:
                img_entry = d[img_col][i]
                img_bytes = img_entry["bytes"] if isinstance(img_entry, dict) else img_entry
                rows.append({
                    "frame_index": int(d["frame_index"][i]),
                    "image_bytes": img_bytes,
                    "prop": np.array(d["observation.state"][i], dtype=np.float32),
                })

    rows.sort(key=lambda r: r["frame_index"])
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Extract VLA embeddings from aic_data")
    parser.add_argument("--backend", type=str, default="xvla", choices=["xvla", "pi05"],
                        help="VLA backend: 'xvla' or 'pi05'")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root of LeRobot v3.0 dataset")
    parser.add_argument("--model_dir", type=str, default="",
                        help="Path to XVLA model directory (for --backend xvla)")
    parser.add_argument("--pi05_checkpoint", type=str,
                        default="/home/yifeng/workspace/pi05_base/pi05_base",
                        help="Pi0.5 checkpoint dir (for --backend pi05)")
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
    parser.add_argument("--chunk_length", type=int, default=10,
                        help="VLA reference action chunk length")
    parser.add_argument("--extract_ref_actions", action="store_true", default=True,
                        help="Also extract per-frame VLA action chunks so Phase 2 "
                             "offline RL sees the same reference distribution as deploy")
    parser.add_argument("--no_ref_actions", dest="extract_ref_actions",
                        action="store_false")
    parser.add_argument("--extract_phase_prompts", action="store_true", default=False,
                        help="Extract embeddings+ref_actions for all 4 phase prompts "
                             "(approach/align/insert/verify) in addition to the main "
                             "instruction. Enables phase-matched training. ~4× slower.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if output file already exists")
    parser.add_argument("--max_episodes", type=int, default=0,
                        help="If >0, process only the first N episodes (smoke-test).")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Load VLA backend
    if args.backend == "xvla":
        vla = XVLAWrapper(
            model_dir=args.model_dir,
            device=device,
            instruction=args.instruction,
            image_size=args.image_size,
            chunk_length=args.chunk_length,
        )
    elif args.backend == "pi05":
        from aic_rlt.vla.pi05_backend import Pi05Backend
        vla = Pi05Backend(
            checkpoint_dir=args.pi05_checkpoint,
            device=device,
            instruction=args.instruction,
        )
        logger.info(f"Pi0.5 backend: {vla.num_tokens} tokens x {vla.embed_dim} dim")
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

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
    if args.max_episodes and args.max_episodes > 0:
        episode_indices = episode_indices[: args.max_episodes]
        logger.info(f"--max_episodes={args.max_episodes}: limited to {episode_indices}")
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
        all_ref_actions = []  # (T, chunk_length, 7) for xvla when extract_ref_actions
        want_ref = args.extract_ref_actions and args.backend == "xvla"
        if args.extract_ref_actions and args.backend != "xvla":
            logger.warning(
                "extract_ref_actions is only wired for backend=xvla; "
                "skipping ref-action extraction for backend=%s", args.backend,
            )

        # Process in batches
        for batch_start in range(0, T, args.batch_size):
            batch_frames = frames[batch_start : batch_start + args.batch_size]
            batch_imgs = [decode_image(f["image_bytes"], args.image_size)
                          for f in batch_frames]

            batch_embs = []
            if args.backend == "xvla":
                for img_np, frame in zip(batch_imgs, batch_frames):
                    emb = vla.get_embeddings(img_np)   # (num_tokens, embed_dim)
                    batch_embs.append(emb)
                    if want_ref:
                        ref = vla.get_action_chunk(img_np, frame["prop"])  # (C, 7)
                        all_ref_actions.append(ref)
            elif args.backend == "pi05":
                import jax
                import jax.numpy as jnp
                from openpi.models import model as _model
                for img_np in batch_imgs:
                    # Build a minimal Pi0.5 observation from the image
                    img_f = (img_np.astype(np.float32) / 127.5) - 1.0
                    img_224 = img_f if img_f.shape[:2] == (224, 224) else \
                        np.array(Image.fromarray(img_np).resize((224, 224)))
                    if img_224.dtype == np.uint8:
                        img_224 = (img_224.astype(np.float32) / 127.5) - 1.0
                    obs = _model.Observation(
                        images={k: jnp.array(img_224)[None] for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
                        image_masks={k: jnp.ones((1,), dtype=jnp.bool_) for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
                        state=jnp.zeros((1, 32), dtype=jnp.float32),
                        tokenized_prompt=jnp.zeros((1, 200), dtype=jnp.int32),
                        tokenized_prompt_mask=jnp.zeros((1, 200), dtype=jnp.bool_),
                    )
                    obs = _model.preprocess_observation(None, obs, train=False)
                    prefix_tokens, _, _ = vla._model_obj.embed_prefix(obs)
                    # bfloat16 → float32 via JAX
                    prefix_f32 = prefix_tokens[0].astype(jnp.float32)
                    emb = torch.from_numpy(np.array(jax.device_get(prefix_f32), dtype=np.float32))
                    batch_embs.append(emb)

            all_embeddings.extend(batch_embs)

            if (batch_start // args.batch_size) % 5 == 0:
                logger.info(f"    {batch_start}/{T} frames processed")

        embeddings = torch.stack(all_embeddings, dim=0)   # (T, num_tokens, 1024)
        out_blob = {"vla_embeddings": embeddings}
        if want_ref and all_ref_actions:
            ref_actions = torch.from_numpy(
                np.stack(all_ref_actions, axis=0)
            ).float()                                      # (T, chunk_length, 7)
            out_blob["ref_actions"] = ref_actions
            out_blob["ref_instruction"] = args.instruction
            logger.info(
                f"  Episode {ep_idx:04d}: ref_actions {tuple(ref_actions.shape)}"
            )

        # Phase-conditioned embeddings + ref_actions (4 prompts × T frames)
        if args.extract_phase_prompts and args.backend == "xvla":
            phase_embs = {}   # name → Tensor(T, N, D)
            phase_refs = {}   # name → Tensor(T, C, 7)
            for phase_id, phase_name in enumerate(PHASE_NAMES):
                prompt = DEFAULT_PHASE_PROMPTS[phase_id]
                vla.set_instruction(prompt)
                p_embs = []
                p_refs = []
                for batch_start in range(0, T, args.batch_size):
                    batch_frames = frames[batch_start : batch_start + args.batch_size]
                    batch_imgs = [decode_image(f["image_bytes"], args.image_size)
                                  for f in batch_frames]
                    for img_np, frame in zip(batch_imgs, batch_frames):
                        p_embs.append(vla.get_embeddings(img_np))
                        if want_ref:
                            p_refs.append(vla.get_action_chunk(img_np, frame["prop"]))
                phase_embs[phase_name] = torch.stack(p_embs, dim=0)
                if want_ref and p_refs:
                    phase_refs[phase_name] = torch.from_numpy(
                        np.stack(p_refs, axis=0)
                    ).float()
                logger.info(
                    f"  Episode {ep_idx:04d}: phase '{phase_name}' embeddings extracted"
                )
            # Restore default instruction
            vla.set_instruction(args.instruction)
            out_blob["phase_embeddings"] = phase_embs
            out_blob["phase_prompts"] = list(DEFAULT_PHASE_PROMPTS)
            if phase_refs:
                out_blob["phase_ref_actions"] = phase_refs

        torch.save(out_blob, out_path)
        logger.info(f"  Episode {ep_idx:04d}: saved {embeddings.shape} → {out_path}")

    logger.info("Embedding extraction complete.")


if __name__ == "__main__":
    main()
