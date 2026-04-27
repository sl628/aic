"""Heuristic phase labeling for all 180 aic episodes.

Labels each frame as one of 4 phases based on position/orientation velocity
thresholds read from the 26-dim state vector:

    P0 (approach)  — moving toward target, no wrist rotation
    P1 (align)     — wrist rotates to align gripper with cable
    P2 (insert)    — arm descends / inserts cable into port
    P3 (settle)    — slow drift to rest pose

Output: JSON mapping episode_id → [phase_label_per_frame].
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("label_phases")

DATA_ROOT = Path("/home/yifeng/aic_xvla_data")
PHASE_NAMES = ["approach", "coarse_align", "fine_align", "insert"]


def _smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving-average smoothing."""
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def label_episode(parquet_path: str) -> list[int]:
    """Return a phase label (0-3) for every frame in one episode."""
    table = pq.read_table(
        parquet_path,
        columns=[f"state_{i}" for i in range(26)],
    )
    state = np.column_stack([
        np.asarray(table.column(f"state_{i}"), dtype=np.float64)
        for i in range(26)
    ])
    T = state.shape[0]
    if T < 10:
        return [0] * T

    pos_vel = np.linalg.norm(state[:, 7:10], axis=1)  # linear velocity
    ori_vel = np.linalg.norm(state[:, 10:13], axis=1)  # angular velocity
    z_vel = np.abs(state[:, 9])  # z linear velocity

    pos_vel = _smooth(pos_vel)
    ori_vel = _smooth(ori_vel)

    peak_pos = max(pos_vel.max(), 1e-8)
    peak_ori = max(ori_vel.max(), 1e-8)

    pos_norm = pos_vel / peak_pos
    ori_norm = ori_vel / peak_ori

    # Find the orientation-dominant region (coarse + fine align).
    # Phase 0 (approach): before orientation starts
    # Phase 1 (coarse-align): ori high (ramp up + peak)
    # Phase 2 (fine-align): ori low (ramp down, fine adjustments)
    # Phase 3 (insert): after orientation, pos dominates again

    orient_start = None
    orient_end = None
    for t in range(T):
        if ori_norm[t] > 0.15 and ori_norm[t] > pos_norm[t]:
            if orient_start is None:
                orient_start = t
            orient_end = t

    labels = np.full(T, 3, dtype=np.int64)  # default = insert

    if orient_start is None or orient_end is None or orient_end - orient_start < 5:
        # No clear orientation phase — everything is approach → insert.
        mid = T // 2
        labels[:mid] = 0
        labels[mid:] = 3
        return labels.tolist()

    # Find split point within orient region: where ori drops below 50% of its peak.
    peak_ori_val = ori_norm[orient_start:orient_end + 1].max()
    coarse_fine_split = orient_end
    for t in range(orient_start, orient_end + 1):
        if ori_norm[t] < 0.5 * peak_ori_val:
            coarse_fine_split = t
            break

    labels[:orient_start] = 0                     # approach
    labels[orient_start:coarse_fine_split] = 1    # coarse-align
    labels[coarse_fine_split:orient_end + 1] = 2  # fine-align
    labels[orient_end + 1:] = 3                   # insert

    return labels.tolist()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--out", default=str(DATA_ROOT / "episode_labels.json"))
    args = p.parse_args()

    root = Path(args.data_root)
    ep_dirs = sorted(root.glob("episodes/episode_*/data.parquet"))
    log.info("Found %d episodes", len(ep_dirs))

    all_labels: dict[str, list[int]] = {}
    counts = np.zeros(4, dtype=np.int64)

    for i, parquet_path in enumerate(ep_dirs):
        ep_id = parquet_path.parent.name
        labels = label_episode(str(parquet_path))
        all_labels[ep_id] = labels
        for p in range(4):
            counts[p] += labels.count(p)
        if (i + 1) % 20 == 0:
            log.info("  processed %d/%d episodes", i + 1, len(ep_dirs))

    total = int(counts.sum())
    pcts = counts / total * 100
    log.info(
        "Label distribution (%d frames):", total
    )
    for p in range(4):
        log.info("  P%d (%s): %d (%.1f%%)", p, PHASE_NAMES[p], int(counts[p]), pcts[p])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_labels, f)
    log.info("Saved labels to %s", out_path)


if __name__ == "__main__":
    main()
