#!/usr/bin/env python3
"""Synthetic demonstration data generator for AIC cable-insertion policies.

Generates plausible trajectories that mimic CheatCode's motion pattern
(approach above port → align → descend to insert) with randomised port
positions and natural noise, then writes them directly as a LeRobotDataset
that can be consumed by ``lerobot-train`` with no further conversion.

No ROS or simulation is required — run this script as a standalone Python
program to bootstrap your training pipeline before real data is available.

Observation space
-----------------
State (26-D float32):
  [0:3]   tcp_pose position   (x, y, z)  in base_link frame
  [3:7]   tcp_pose orientation (qx, qy, qz, qw)
  [7:10]  tcp_velocity linear  (x, y, z)
  [10:13] tcp_velocity angular (x, y, z)
  [13:19] tcp_error (x, y, z, rx, ry, rz)
  [19:26] joint_positions (7 joints)

Action space
-----------
7-D float32 position-based target:
  [0:3]  target TCP position    (x, y, z)
  [3:7]  target TCP orientation (qx, qy, qz, qw)

Usage
-----
Via pixi task (recommended)::

    pixi run generate-data

Directly::

    python3 aic_utils/sym_data/generate_synthetic.py \\
        --output_dir ~/aic_data \\
        --num_episodes 200

The dataset is written to ``<output_dir>/synthetic/cable_insertion/``.
Load it for training with::

    pixi run train-act
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ID = "synthetic/cable_insertion"
TASK_PROMPT = "Insert SFP cable into NIC port"
FPS = 20

# Image resolution: 1152 × 1024 × 0.25 scale (matches RunACT / lerobot_robot_aic)
IMG_H, IMG_W = 256, 288  # (1024 * 0.25, 1152 * 0.25)

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (26,),
        "names": [
            "tcp_pos_x", "tcp_pos_y", "tcp_pos_z",
            "tcp_quat_x", "tcp_quat_y", "tcp_quat_z", "tcp_quat_w",
            "tcp_vel_lx", "tcp_vel_ly", "tcp_vel_lz",
            "tcp_vel_ax", "tcp_vel_ay", "tcp_vel_az",
            "tcp_err_x", "tcp_err_y", "tcp_err_z",
            "tcp_err_rx", "tcp_err_ry", "tcp_err_rz",
            "joint_0", "joint_1", "joint_2", "joint_3",
            "joint_4", "joint_5", "joint_6",
        ],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["target_x", "target_y", "target_z",
                  "target_qx", "target_qy", "target_qz", "target_qw"],
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

# UR5e home TCP pose (base_link frame, approximate)
HOME_POS = np.array([0.35, -0.20, 0.55], dtype=np.float32)
HOME_QUAT = np.array([0.0, 0.707, 0.707, 0.0], dtype=np.float32)  # qx,qy,qz,qw

HOME_JOINTS = np.array(
    [-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110, 0.0], dtype=np.float32
)

# Port position ranges (randomised per episode in base_link frame)
PORT_POS_MEAN = np.array([0.50, 0.05, 0.25], dtype=np.float32)
PORT_POS_STD = np.array([0.03, 0.03, 0.02], dtype=np.float32)

APPROACH_Z_OFFSET = 0.20   # above port before descending
INSERT_Z_OFFSET = -0.015   # final depth (plug fully inserted)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _norm_q(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-8 else q


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0, q1 = _norm_q(q0), _norm_q(q1)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        return _norm_q(q0 + t * (q1 - q0))
    theta = np.arccos(dot)
    s0 = np.sin((1 - t) * theta) / np.sin(theta)
    s1 = np.sin(t * theta) / np.sin(theta)
    return _norm_q(s0 * q0 + s1 * q1)


def _ease(t: float) -> float:
    """Smooth ease-in-out (cubic)."""
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------------------
# Synthetic image generation
# ---------------------------------------------------------------------------

def _make_image(
    cam_key: str,
    tcp_pos: np.ndarray,
    port_pos: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a synthetic RGB image (H×W×3 uint8).

    Renders a coloured gradient background, a bright ellipse where the port
    projects, and per-pixel Gaussian noise.  Not photorealistic — the intent
    is to give the model realistic array shapes and a coarse spatial signal
    so the pipeline can be validated end-to-end before real images arrive.
    """
    # Per-camera background colour makes the three views visually distinct
    base_color = {
        "observation.images.left_camera":   np.array([30, 15, 10], np.float32),
        "observation.images.center_camera": np.array([10, 30, 15], np.float32),
        "observation.images.right_camera":  np.array([10, 15, 30], np.float32),
    }.get(cam_key, np.array([20, 20, 20], np.float32))

    # Gradient (brighter towards bottom)
    gradient = np.linspace(0, 60, IMG_H, dtype=np.float32)[:, None, None]
    img = (base_color[None, None] + gradient).clip(0, 255).astype(np.uint8)
    img = np.broadcast_to(img, (IMG_H, IMG_W, 3)).copy()

    # Project port onto image plane (rough pinhole)
    rel = port_pos - tcp_pos
    focal = IMG_W * 1.5
    cx, cy = IMG_W // 2, IMG_H // 2
    denom = max(abs(rel[2]), 0.05)
    px = int(cx - focal * rel[0] / denom)
    py = int(cy - focal * rel[1] / denom)

    if 0 <= px < IMG_W and 0 <= py < IMG_H:
        r = max(4, int(6 * 0.3 / denom))
        # Outer ring (bright)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    iy, ix = py + dy, px + dx
                    if 0 <= iy < IMG_H and 0 <= ix < IMG_W:
                        img[iy, ix] = [200, 200, 50]
        # Inner hole (dark)
        ri = max(1, r // 3)
        for dy in range(-ri, ri + 1):
            for dx in range(-ri, ri + 1):
                if dx * dx + dy * dy <= ri * ri:
                    iy, ix = py + dy, px + dx
                    if 0 <= iy < IMG_H and 0 <= ix < IMG_W:
                        img[iy, ix] = [20, 20, 20]

    # Gaussian noise
    noise = rng.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Episode generator
# ---------------------------------------------------------------------------

def _generate_episode(
    ds: LeRobotDataset,
    steps_approach: int,
    steps_insert: int,
    rng: np.random.Generator,
    success_rate: float,
) -> None:
    port_pos = PORT_POS_MEAN + rng.standard_normal(3).astype(np.float32) * PORT_POS_STD
    port_quat = _norm_q(HOME_QUAT + rng.standard_normal(4).astype(np.float32) * 0.05)

    approach_pos = port_pos.copy()
    approach_pos[2] += APPROACH_Z_OFFSET
    insert_pos = port_pos.copy()
    insert_pos[2] += INSERT_Z_OFFSET

    tcp_pos = HOME_POS.copy()
    tcp_quat = HOME_QUAT.copy()
    joint_pos = HOME_JOINTS.copy()

    total_steps = steps_approach + steps_insert

    for step in range(total_steps):
        # ---- compute target ----
        if step < steps_approach:
            frac = _ease(step / max(steps_approach - 1, 1))
            target_pos = (1 - frac) * HOME_POS + frac * approach_pos
            target_quat = _slerp(HOME_QUAT, port_quat, frac)
        else:
            frac = _ease((step - steps_approach) / max(steps_insert - 1, 1))
            target_pos = (1 - frac) * approach_pos + frac * insert_pos
            target_quat = port_quat.copy()

        # ---- simulate lag + noise ----
        lag = 0.85
        tcp_pos_prev = tcp_pos.copy()
        tcp_pos = lag * tcp_pos + (1 - lag) * target_pos + rng.standard_normal(3).astype(np.float32) * 0.001
        tcp_quat = _norm_q(_slerp(tcp_quat, target_quat, 1 - lag + 0.01))
        tcp_vel_lin = (tcp_pos - tcp_pos_prev) * FPS
        tcp_vel_ang = rng.standard_normal(3).astype(np.float32) * 0.005
        tcp_error = np.concatenate([
            target_pos - tcp_pos + rng.standard_normal(3).astype(np.float32) * 0.0005,
            rng.standard_normal(3).astype(np.float32) * 0.002,
        ]).astype(np.float32)
        joint_pos = (HOME_JOINTS + rng.standard_normal(7).astype(np.float32) * 0.02).astype(np.float32)

        state = np.concatenate([
            tcp_pos, tcp_quat, tcp_vel_lin, tcp_vel_ang, tcp_error, joint_pos
        ]).astype(np.float32)
        action = np.concatenate([target_pos, target_quat]).astype(np.float32)

        frame: dict = {
            "task": TASK_PROMPT,
            "observation.state": state,
            "action": action,
        }
        for cam_key in FEATURES:
            if "images" in cam_key:
                frame[cam_key] = _make_image(cam_key, tcp_pos, port_pos, rng)

        ds.add_frame(frame)

    ds.save_episode()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic LeRobot dataset for AIC cable insertion."
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path.home() / "aic_data"),
        help="Root directory for the dataset (dataset goes in <output_dir>/synthetic/cable_insertion). "
             "Default: ~/aic_data",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=100,
        help="Number of episodes to generate (default: 100)",
    )
    parser.add_argument(
        "--steps_approach", type=int, default=60,
        help="Steps in the approach phase (default: 60, = 3 s at 20 Hz)",
    )
    parser.add_argument(
        "--steps_insert", type=int, default=80,
        help="Steps in the insertion phase (default: 80, = 4 s at 20 Hz)",
    )
    parser.add_argument(
        "--success_rate", type=float, default=0.95,
        help="Fraction of episodes labelled as successful (default: 0.95)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    dataset_path = output_dir / REPO_ID
    if dataset_path.exists():
        raise FileExistsError(
            f"Dataset already exists at {dataset_path}. "
            "Delete it or choose a different --output_dir."
        )

    rng = np.random.default_rng(args.seed)

    total_frames = args.num_episodes * (args.steps_approach + args.steps_insert)
    print(f"Generating {args.num_episodes} episodes "
          f"({args.steps_approach + args.steps_insert} steps each = {total_frames} total frames)")
    print(f"Dataset root: {dataset_path}")

    ds = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=FEATURES,
        root=output_dir,
        use_videos=False,
    )

    for ep in range(args.num_episodes):
        _generate_episode(
            ds=ds,
            steps_approach=args.steps_approach,
            steps_insert=args.steps_insert,
            rng=rng,
            success_rate=args.success_rate,
        )
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  [{ep + 1}/{args.num_episodes}] episodes written")

    ds.finalize()
    print(f"\nDone. {ds.meta.total_episodes} episodes, "
          f"{ds.meta.total_frames} frames → {dataset_path}")
    print("\nNext step:")
    print("  pixi run train-act")


if __name__ == "__main__":
    main()
