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
# xvla_wrapper is imported lazily inside main() so that pi05-only runs
# (which use openpi's venv, where the xvla lerobot policy isn't installed)
# don't fail at script import time.
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


def _select_parquet_files_for_episode(parquet_files, episode_index: int):
    """Narrow the file list to only those containing `episode_index`.

    aic's LeRobot v3.0 dataset stores one episode per parquet (verified:
    file-XXX.parquet contains episode XXX). If that convention holds,
    we only read that one file instead of all 100 — ~100× faster.
    We fall back to the full list if the convention doesn't match.
    """
    import pyarrow.parquet as _pq

    # Fast path: filename pattern file-{NNN}.parquet matches episode NNN.
    for pf in parquet_files:
        stem = Path(pf).stem  # e.g. "file-007"
        if stem.startswith("file-"):
            try:
                n = int(stem.split("-", 1)[1])
            except (ValueError, IndexError):
                continue
            if n == episode_index:
                # Sanity-check: read a tiny slice of episode_index column
                # to confirm the convention holds for this dataset.
                t = _pq.read_table(pf, columns=["episode_index"]).to_pydict()
                uniq = set(int(x) for x in t["episode_index"])
                if uniq == {episode_index}:
                    return [pf]
                # Convention violated — fall through to full scan.
                break
    return parquet_files


def load_episode_frames(parquet_files, episode_index: int, cameras):
    """Load all frames for one episode from parquet files.

    Args:
        parquet_files: list of parquet file paths (sorted).
        episode_index: episode to load.
        cameras: str or list of str. If str, behaves as the single-camera path
            and stores bytes under `image_bytes`. If list, stores a dict
            `images_bytes = {cam_name: bytes}` so multi-camera backends (pi05)
            can pull what they need.

    Returns:
        frames: list of dicts sorted by frame_index, each with:
            - frame_index (int)
            - image_bytes (bytes)                     # single-camera
              or images_bytes (dict[str, bytes])      # multi-camera
            - prop (np.ndarray, 26D) — observation.state
    """
    multi = isinstance(cameras, (list, tuple))
    cam_list = list(cameras) if multi else [cameras]
    img_cols = [f"observation.images.{c}" for c in cam_list]

    def _bytes(entry):
        return entry["bytes"] if isinstance(entry, dict) else entry

    narrowed = _select_parquet_files_for_episode(parquet_files, episode_index)
    rows = []
    for pf in narrowed:
        table = pq.read_table(
            pf,
            columns=["episode_index", "frame_index", *img_cols, "observation.state"],
        )
        d = table.to_pydict()
        n = len(d["episode_index"])
        for i in range(n):
            if int(d["episode_index"][i]) != episode_index:
                continue
            row = {
                "frame_index": int(d["frame_index"][i]),
                "prop": np.array(d["observation.state"][i], dtype=np.float32),
            }
            if multi:
                row["images_bytes"] = {
                    c: _bytes(d[col][i]) for c, col in zip(cam_list, img_cols)
                }
            else:
                row["image_bytes"] = _bytes(d[img_cols[0]][i])
            rows.append(row)

    rows.sort(key=lambda r: r["frame_index"])
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Extract VLA embeddings from aic_data")
    parser.add_argument(
        "--backend",
        type=str,
        default="xvla",
        choices=["xvla", "pi05"],
        help="VLA backend: 'xvla' or 'pi05'",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Root of LeRobot v3.0 dataset"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Path to XVLA model directory (for --backend xvla)",
    )
    parser.add_argument(
        "--pi05_checkpoint",
        type=str,
        default="/home/yifeng/workspace/pi05_base/pi05_base",
        help="Pi0.5 checkpoint dir (for --backend pi05)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save per-episode .pt files",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="center_camera",
        help="Camera key to use (default: center_camera)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Insert SFP cable into NIC port",
        help="Language instruction for the task",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Frames per forward pass (default: 8)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Resize images to this size (default: 256)",
    )
    parser.add_argument(
        "--chunk_length", type=int, default=10, help="VLA reference action chunk length"
    )
    # Default: xvla extracts ref_actions (its native TCP outputs match aic demos);
    # pi05 does NOT (Option B — pi0.5 as embedding-only feature extractor; its
    # joint-space predictions are OOD for aic's workspace and not useful as BC).
    parser.add_argument(
        "--extract_ref_actions",
        dest="extract_ref_actions",
        action="store_true",
        default=None,
        help="Force-on per-frame VLA action chunk extraction. "
        "Defaults: xvla=on, pi05=off.",
    )
    parser.add_argument(
        "--no_ref_actions",
        dest="extract_ref_actions",
        action="store_false",
        help="Force-off ref_actions regardless of backend.",
    )
    parser.add_argument(
        "--extract_phase_prompts",
        action="store_true",
        default=False,
        help="Extract embeddings+ref_actions for all 4 phase prompts "
        "(approach/align/insert/verify) in addition to the main "
        "instruction. Enables phase-matched training. ~4× slower.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract even if output file already exists",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help="If >0, process only the first N episodes (smoke-test).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Resolve per-backend default for extract_ref_actions.
    # None = user didn't specify; pick the right default per backend.
    if args.extract_ref_actions is None:
        args.extract_ref_actions = args.backend == "xvla"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Load VLA backend
    if args.backend == "xvla":
        from aic_rlt.vla.xvla_wrapper import XVLAWrapper

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
        if args.backend == "pi05":
            # pi05/ur5 recipe uses 2 cameras: base + single wrist.
            # aic records 3 (left/center/right); we map center→base, left→wrist.
            frames = load_episode_frames(
                parquet_files,
                ep_idx,
                cameras=["center_camera", "left_camera"],
            )
        else:
            frames = load_episode_frames(parquet_files, ep_idx, args.camera)
        if not frames:
            logger.warning(f"  Episode {ep_idx:04d}: no frames found, skipping.")
            continue

        T = len(frames)
        all_embeddings = []
        all_ref_actions = []
        # Both backends now extract ref_actions; enable unless the user opts out.
        want_ref = args.extract_ref_actions

        # Process in batches
        for batch_start in range(0, T, args.batch_size):
            batch_frames = frames[batch_start : batch_start + args.batch_size]

            batch_embs = []
            if args.backend == "xvla":
                batch_imgs = [
                    decode_image(f["image_bytes"], args.image_size)
                    for f in batch_frames
                ]
                for img_np, frame in zip(batch_imgs, batch_frames):
                    emb = vla.get_embeddings(img_np)  # (num_tokens, embed_dim)
                    batch_embs.append(emb)
                    if want_ref:
                        ref = vla.get_action_chunk(img_np, frame["prop"])  # (C, 7 or 9)
                        all_ref_actions.append(ref)
            elif args.backend == "pi05":
                for frame in batch_frames:
                    base_img = decode_image(frame["images_bytes"]["center_camera"], 224)
                    wrist_img = decode_image(frame["images_bytes"]["left_camera"], 224)
                    # aic state layout: prop[19:25]=joint_0..5, prop[25]=joint_6(gripper)
                    prop = frame["prop"]
                    joints = prop[19:25]  # (6,)
                    gripper = np.array([prop[25]], dtype=np.float32)  # (1,)
                    backend_obs = vla.frame_to_backend_input(
                        joints=joints,
                        gripper=gripper,
                        base_rgb=base_img,
                        wrist_rgb=wrist_img,
                    )
                    if want_ref:
                        # Slow path: run the full forward pass (embed + denoise).
                        emb_t, ref = vla.get_embeddings_and_actions(backend_obs)
                        batch_embs.append(emb_t.squeeze(0).cpu())
                        all_ref_actions.append(ref)  # (C, 7)
                    else:
                        # Fast path: Option B — embeddings only, skip denoise.
                        emb_t = vla.get_embeddings(backend_obs)
                        batch_embs.append(emb_t.squeeze(0).cpu())

            all_embeddings.extend(batch_embs)

            if (batch_start // args.batch_size) % 5 == 0:
                logger.info(f"    {batch_start}/{T} frames processed")

        embeddings = torch.stack(all_embeddings, dim=0)  # (T, num_tokens, D)
        out_blob = {
            "vla_embeddings": embeddings,
            # Always record which instruction produced these embeddings, so
            # downstream training can detect instruction drift between extraction
            # and inference (see plan risk #13).
            "instruction": args.instruction,
            "backend": args.backend,
        }
        if want_ref and all_ref_actions:
            ref_actions = torch.from_numpy(
                np.stack(all_ref_actions, axis=0)
            ).float()  # (T, C, action_dim)
            out_blob["ref_actions"] = ref_actions
            # Back-compat alias used by older validators/trainers.
            out_blob["ref_instruction"] = args.instruction
            out_blob["action_dim"] = int(ref_actions.shape[-1])
            logger.info(
                f"  Episode {ep_idx:04d}: ref_actions {tuple(ref_actions.shape)}"
            )

        # Phase-conditioned embeddings + ref_actions (4 prompts × T frames)
        if args.extract_phase_prompts and args.backend == "xvla":
            phase_embs = {}  # name → Tensor(T, N, D)
            phase_refs = {}  # name → Tensor(T, C, 7)
            for phase_id, phase_name in enumerate(PHASE_NAMES):
                prompt = DEFAULT_PHASE_PROMPTS[phase_id]
                vla.set_instruction(prompt)
                p_embs = []
                p_refs = []
                for batch_start in range(0, T, args.batch_size):
                    batch_frames = frames[batch_start : batch_start + args.batch_size]
                    batch_imgs = [
                        decode_image(f["image_bytes"], args.image_size)
                        for f in batch_frames
                    ]
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

        # Free JAX compilation/trace caches between episodes. Without this,
        # the JIT cache grows monotonically and eventually OOMs on GPUs with
        # tight memory headroom (observed at ep 5 on a 12GB Titan Xp with
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.92). Safe: only clears caches,
        # not model weights.
        if args.backend == "pi05":
            try:
                import jax

                jax.clear_caches()
            except Exception as e:
                logger.warning(f"jax.clear_caches() failed: {e}")

    logger.info("Embedding extraction complete.")


if __name__ == "__main__":
    main()
