#!/usr/bin/env python3
"""Convert CheatCodeDataCollector output to LeRobot v3.0 format.

CheatCodeDataCollector (in aic_example_policies) saves episodes in a custom
flat format when run via the aic_model ROS node with ground_truth:=true.
This script converts that raw output into a LeRobotDataset consumable by
lerobot-train.

Input layout (written by CheatCodeDataCollector)::

    <input_dir>/
      dataset_index.jsonl          # one JSON line per episode
      episodes/
        <episode_id>/
          data.parquet             # per-step state + action (flat columns)
          images/
            left_camera/           # 000000.jpg, 000001.jpg, ...
            center_camera/
            right_camera/

State columns in data.parquet: state_0 .. state_25  (26D float32)
Action columns in data.parquet: action_0 .. action_6 (7D float32)

Output: LeRobot v3.0 dataset at <output_dir>/cheatcode/cable_insertion/

Usage::

    pixi run python aic_utils/sym_data/convert_cheatcode_to_lerobot.py \\
        --input_dir /home/yifeng/aic_data_raw \\
        --output_dir /home/yifeng/aic_data_sym

    # Only keep successful episodes (default: True)
    pixi run python aic_utils/sym_data/convert_cheatcode_to_lerobot.py \\
        --input_dir /home/yifeng/aic_data_raw \\
        --output_dir /home/yifeng/aic_data_sym \\
        --only_successful

    # Include failed episodes too
    pixi run python aic_utils/sym_data/convert_cheatcode_to_lerobot.py \\
        --input_dir /home/yifeng/aic_data_raw \\
        --output_dir /home/yifeng/aic_data_sym \\
        --no-only_successful
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---------------------------------------------------------------------------
# Constants — must match CheatCodeDataCollector and generate_synthetic.py
# ---------------------------------------------------------------------------

REPO_ID = "cheatcode/cable_insertion"
TASK_PROMPT = "Insert SFP cable into NIC port"
FPS = 20

_CAMERA_KEYS = ["left_camera", "center_camera", "right_camera"]
_STATE_DIM = 26
_ACTION_DIM = 7

# Image size written by CheatCodeDataCollector (_IMAGE_SCALE = 0.25 on 1152×1024)
IMG_H, IMG_W = 256, 288

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (_STATE_DIM,),
        "names": [
            "tcp_pos_x",
            "tcp_pos_y",
            "tcp_pos_z",
            "tcp_quat_x",
            "tcp_quat_y",
            "tcp_quat_z",
            "tcp_quat_w",
            "tcp_vel_lx",
            "tcp_vel_ly",
            "tcp_vel_lz",
            "tcp_vel_ax",
            "tcp_vel_ay",
            "tcp_vel_az",
            "tcp_err_x",
            "tcp_err_y",
            "tcp_err_z",
            "tcp_err_rx",
            "tcp_err_ry",
            "tcp_err_rz",
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ],
    },
    "action": {
        "dtype": "float32",
        "shape": (_ACTION_DIM,),
        "names": [
            "target_x",
            "target_y",
            "target_z",
            "target_qx",
            "target_qy",
            "target_qz",
            "target_qw",
        ],
    },
    "observation.images.left_camera": {
        "dtype": "image",
        "shape": (IMG_H, IMG_W, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.center_camera": {
        "dtype": "image",
        "shape": (IMG_H, IMG_W, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.right_camera": {
        "dtype": "image",
        "shape": (IMG_H, IMG_W, 3),
        "names": ["height", "width", "channel"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_index(input_dir: Path, only_successful: bool) -> list[dict]:
    """Read dataset_index.jsonl and return episodes sorted by timestamp."""
    index_path = input_dir / "dataset_index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(
            f"dataset_index.jsonl not found in {input_dir}. "
            "Run CheatCodeDataCollector first."
        )
    episodes = []
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    if only_successful:
        before = len(episodes)
        episodes = [e for e in episodes if e.get("success", False)]
        print(
            f"  Filtered to {len(episodes)} successful episodes "
            f"(dropped {before - len(episodes)} failed)"
        )

    # Stable ordering: sort by recorded timestamp
    episodes.sort(key=lambda e: e.get("timestamp", 0))
    return episodes


def _load_episode_data(ep_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read data.parquet → (state_array, action_array) both (N, dim) float32."""
    df = pd.read_parquet(ep_dir / "data.parquet")
    df = df.sort_values("frame_index").reset_index(drop=True)

    state_cols = [f"state_{i}" for i in range(_STATE_DIM)]
    action_cols = [f"action_{i}" for i in range(_ACTION_DIM)]

    states = df[state_cols].to_numpy(dtype=np.float32)
    actions = df[action_cols].to_numpy(dtype=np.float32)
    return states, actions


def _load_image(image_path: Path) -> np.ndarray:
    """Load a JPEG image as (H, W, 3) uint8 RGB numpy array."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert(
    input_dir: Path, output_dir: Path, only_successful: bool, task: str
) -> None:
    dataset_path = output_dir / REPO_ID
    if dataset_path.exists():
        raise FileExistsError(
            f"Output dataset already exists at {dataset_path}. "
            "Delete it or choose a different --output_dir."
        )

    episodes = _load_index(input_dir, only_successful)
    if not episodes:
        raise RuntimeError(
            "No episodes to convert. Check --input_dir and --only_successful."
        )

    print(f"Converting {len(episodes)} episodes from {input_dir}")
    print(f"Output → {dataset_path}")

    ds = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=FEATURES,
        root=output_dir,
        use_videos=False,
    )

    for i, ep_meta in enumerate(episodes):
        episode_id = ep_meta["episode_id"]
        ep_dir = input_dir / "episodes" / episode_id

        if not ep_dir.exists():
            print(f"  Warning: episode dir not found, skipping: {ep_dir}")
            continue

        states, actions = _load_episode_data(ep_dir)
        n_steps = len(states)

        for step in range(n_steps):
            frame: dict = {
                "task": task,
                "observation.state": states[step],
                "action": actions[step],
            }
            for cam_key in _CAMERA_KEYS:
                img_path = ep_dir / "images" / cam_key / f"{step:06d}.jpg"
                frame[f"observation.images.{cam_key}"] = _load_image(img_path)

            ds.add_frame(frame)

        ds.save_episode()

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i + 1}/{len(episodes)}] episodes converted")

    ds.finalize()
    print(
        f"\nDone. {ds.meta.total_episodes} episodes, "
        f"{ds.meta.total_frames} frames → {dataset_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CheatCodeDataCollector output to LeRobot v3.0 dataset."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory written by CheatCodeDataCollector (contains dataset_index.jsonl).",
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path.home() / "aic_data_sym"),
        help="Root for the output LeRobot dataset "
        "(dataset goes in <output_dir>/cheatcode/cable_insertion). "
        "Default: ~/aic_data_sym",
    )
    parser.add_argument(
        "--only_successful",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Only convert episodes where success=True (default: True).",
    )
    parser.add_argument(
        "--task",
        default=TASK_PROMPT,
        help=f'Task description string (default: "{TASK_PROMPT}").',
    )
    args = parser.parse_args()

    convert(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        only_successful=args.only_successful,
        task=args.task,
    )


if __name__ == "__main__":
    main()
