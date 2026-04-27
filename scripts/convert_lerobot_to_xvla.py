#!/usr/bin/env python3
"""Convert LeRobot video dataset → X-VLA flat-parquet + JPG format.

Uses OpenCV for video decoding (avoids torchcodec/FFmpeg shared lib issues).

Usage:
    pixi run python scripts/convert_lerobot_to_xvla.py \
        --dataset siyulw2025/cableholder-all \
        --output /path/to/xvla_data \
        --episodes 0-179
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CACHE = Path.home() / ".cache/huggingface/hub"
CAMERAS = ["left_camera", "center_camera", "right_camera"]


def get_snapshot_root(dataset_id: str) -> Path:
    """Resolve a HF dataset repo_id → local cache snapshot path."""
    name = dataset_id.replace("/", "--")
    base = CACHE / f"datasets--{name}" / "snapshots"
    snapshots = sorted(base.iterdir())
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found for {dataset_id} in {base}")
    return snapshots[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="siyulw2025/cableholder-all")
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", default=None)
    args = p.parse_args()

    root = get_snapshot_root(args.dataset)
    info = json.loads((root / "meta" / "info.json").read_text())
    n_total = info["total_episodes"]
    fps = info["fps"]
    data_path_tpl = info["data_path"]  # e.g. "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    video_path_tpl = info["video_path"]  # e.g. "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

    # Episode metadata to know frame range per episode
    ep_meta = pq.read_table(str(root / "meta/episodes/chunk-000/file-000.parquet")).to_pandas()

    out_root = Path(args.output)
    ep_dir = out_root / "episodes"

    if args.episodes:
        eps = set()
        for part in args.episodes.split(","):
            if "-" in part:
                a, b = part.split("-")
                eps.update(range(int(a), int(b) + 1))
            else:
                eps.add(int(part))
        eps = sorted(eps)
    else:
        eps = sorted(ep_meta["episode_index"].tolist())

    print(f"Converting {len(eps)} episodes from {root} ...")

    for ep_idx in eps:
        row = ep_meta[ep_meta["episode_index"] == ep_idx].iloc[0]
        n_frames = int(row["length"])

        ep_out = ep_dir / f"episode_{ep_idx:04d}"
        ep_out.mkdir(parents=True, exist_ok=True)
        for cam in CAMERAS:
            (ep_out / "images" / cam).mkdir(parents=True, exist_ok=True)

        # Read data parquet
        data_path = root / data_path_tpl.format(chunk_index=0, file_index=ep_idx)
        df = pq.read_table(str(data_path)).to_pandas()

        # Drop video columns
        video_cols = [c for c in df.columns if c.startswith("observation.images")]
        df = df.drop(columns=video_cols, errors="ignore")

        # Open video captures
        vcaps = {}
        for cam in CAMERAS:
            cam_key = f"observation.images.{cam}"
            vp = root / video_path_tpl.format(video_key=cam_key, chunk_index=0, file_index=ep_idx)
            cap = cv2.VideoCapture(str(vp))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {vp}")
            vcaps[cam] = cap

        records = []

        for i in range(n_frames):
            # Read images from video
            img_paths = {}
            for cam in CAMERAS:
                cap = vcaps[cam]
                # Set to exact frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, img_bgr = cap.read()
                if not ret:
                    raise RuntimeError(f"Failed to read frame {i} from {cam}")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                fname = f"frame_{i:04d}.jpg"
                cv2.imwrite(str(ep_out / "images" / cam / fname), img_bgr)
                img_paths[cam] = f"episodes/episode_{ep_idx:04d}/images/{cam}/{fname}"

            # State and action from parquet
            row_data = df.iloc[i]
            rec = {
                **{f"state_{j}": float(row_data[f"observation.state"][j]) for j in range(26)},
                **{f"action_{j}": float(row_data["action"][j]) for j in range(7)},
                "image_path_left_camera": img_paths["left_camera"],
                "image_path_center_camera": img_paths["center_camera"],
                "image_path_right_camera": img_paths["right_camera"],
                "episode_id": ep_idx,
                "frame_index": i,
                "timestamp": float(i) / fps,
            }
            records.append(rec)

        for cap in vcaps.values():
            cap.release()

        pd.DataFrame(records).to_parquet(ep_out / "data.parquet", index=False)
        print(f"  ep {ep_idx:4d}: {n_frames:4d} frames")

    meta = {
        "dataset_name": "aic",
        "fps": fps,
        "datalist": sorted(
            [
                {
                    "parquet_path": str((ep_dir / f"episode_{ep:04d}" / "data.parquet").resolve()),
                    "image_root": str(out_root.resolve()),
                    "instruction": "insert the SFP cable into the port",
                    "fps": fps,
                }
                for ep in eps
            ],
            key=lambda x: x["parquet_path"],
        ),
    }

    meta_path = out_root / "aic_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMeta: {meta_path} ({len(meta['datalist'])} eps)")


if __name__ == "__main__":
    main()
