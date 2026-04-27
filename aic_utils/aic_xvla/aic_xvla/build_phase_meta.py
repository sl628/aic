"""Split the 180 labeled episodes into 4 phase-specific training sets for X-VLA.

For each phase (0-3), creates a directory of filtered parquet files containing
only the frames labeled for that phase, plus a meta.json pointing to them.

Usage:
    PYTHONPATH=$PWD/aic_utils/aic_xvla:$PYTHONPATH \
    /home/yifeng/aic/.pixi/envs/default/bin/python3 -m aic_xvla.build_phase_meta
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_phase_meta")

DATA_ROOT = Path("/home/yifeng/aic_xvla_data")
PHASE_NAMES = ["approach", "coarse_align", "fine_align", "insert"]
DEFAULT_INSTRUCTION = "insert the SFP cable into the port"

# Columns that X-VLA's handler reads from the parquet.
STATE_COLS = [f"state_{i}" for i in range(26)]
ACTION_COLS = [f"action_{i}" for i in range(7)]
IMAGE_COLS = [
    "image_path_left_camera",
    "image_path_center_camera",
    "image_path_right_camera",
]
META_COLS = ["episode_id", "frame_index", "timestamp"]
ALL_COLS = STATE_COLS + ACTION_COLS + IMAGE_COLS + META_COLS


def build_phase_data(
    episode_ids: list[str],
    labels: dict[str, list[int]],
    phase: int,
    out_dir: Path,
    meta_path: Path,
) -> None:
    """Create filtered parquets + meta.json for one phase."""
    out_dir.mkdir(parents=True, exist_ok=True)
    datalist = []

    for ep_id in episode_ids:
        ep_labels = labels.get(ep_id)
        if ep_labels is None:
            continue

        phase_mask = np.array(ep_labels) == phase
        if phase_mask.sum() < 5:  # skip if too few frames
            continue

        src = DATA_ROOT / "episodes" / ep_id / "data.parquet"
        table = pq.read_table(str(src), columns=ALL_COLS)
        filtered = table.filter(pa.array(phase_mask))

        out_path = str(out_dir / f"{ep_id}.parquet")
        pq.write_table(filtered, out_path)
        datalist.append({
            "parquet_path": out_path,
            "image_root": str(DATA_ROOT),
            "instruction": DEFAULT_INSTRUCTION,
            "fps": 20,
        })

    meta = {"dataset_name": "aic", "fps": 20, "datalist": datalist}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    total_frames = sum(
        sum(1 for lbl in labels.get(ep_id, []) if lbl == phase)
        for ep_id in episode_ids
    )
    log.info(
        "Phase %d (%s): %d episodes, %d frames → %s",
        phase, PHASE_NAMES[phase], len(datalist), total_frames, meta_path,
    )


def main() -> None:
    with open(DATA_ROOT / "episode_labels.json") as f:
        labels = json.load(f)

    with open(DATA_ROOT / "train_meta.json") as f:
        train_meta = json.load(f)
    with open(DATA_ROOT / "val_meta.json") as f:
        val_meta = json.load(f)

    train_ep_ids = sorted(Path(e["parquet_path"]).parent.name for e in train_meta["datalist"])
    val_ep_ids = sorted(Path(e["parquet_path"]).parent.name for e in val_meta["datalist"])

    for phase in range(4):
        # Training split
        build_phase_data(
            train_ep_ids, labels, phase,
            DATA_ROOT / f"phase_{phase}",
            DATA_ROOT / f"phase_{phase}_train_meta.json",
        )
        # Validation split
        build_phase_data(
            val_ep_ids, labels, phase,
            DATA_ROOT / f"phase_{phase}_val",
            DATA_ROOT / f"phase_{phase}_val_meta.json",
        )

    log.info("Done. Created 4 phase-split training/validation sets.")


if __name__ == "__main__":
    main()
