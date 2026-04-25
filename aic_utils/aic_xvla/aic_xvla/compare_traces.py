"""Diff an offline replay trace against a closed-loop RunXVLA trace.

Both traces are JSONL where each line records one inference call.
Schema (a superset across the two sources):
  {
    "frame" or "step": int,
    "live_pos": [x, y, z],
    "live_quat_xyzw": [...],
    "instruction": str,
    "image_paths": [left, center, right],   # relative paths
    "pred_actions": [[x, y, z, qx, qy, qz, qw], ...],
    "gt_actions": [[...], ...]              # offline only
  }

`image_paths` are interpreted relative to:
  - offline: --offline-image-root  (default: dirname(--offline) — usually
    aic dataset root because the trace stores paths like
    "episodes/<id>/images/<cam>/000000.jpg")
  - closed:  --closed-image-root   (default: dirname(--closed))

We line them up by sequence (offline frame 0 ↔ closed step 0, etc.) and
report how the *predictions* differ given the *live_pos* drift between
the two runs. With --vis-dir we also render side-by-side images per cam
with the live_pos / pred[0] overlaid so you can eyeball whether the
visuals or the proprio are the source of divergence.

Usage:
    python -m aic_xvla.compare_traces \\
        --offline /home/yifeng/aic_xvla_overfit/replay_trace.jsonl \\
        --offline-image-root /home/yifeng/aic_data_one_ep \\
        --closed  /home/yifeng/aic_xvla_overfit/closedloop_trace.jsonl \\
        --closed-image-root /home/yifeng/aic_xvla_overfit \\
        --vis-dir /home/yifeng/aic_xvla_overfit/compare_vis \\
        --n 10
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _key(rec: dict) -> int:
    return rec.get("frame", rec.get("step", -1))


def _annotate(img: np.ndarray, lines: list[str]) -> np.ndarray:
    import cv2

    # Black header with white text; works regardless of image colorspace.
    h, w = img.shape[:2]
    pad = 18 * len(lines) + 8
    out = np.zeros((h + pad, w, 3), dtype=img.dtype)
    out[pad:] = img
    for i, text in enumerate(lines):
        cv2.putText(
            out,
            text,
            (4, 16 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def _render_pair(
    off_rec: dict,
    clo_rec: dict,
    off_root: str,
    clo_root: str,
    vis_dir: str,
    pair_idx: int,
) -> None:
    import cv2

    cams = ["left", "center", "right"]
    rows = []
    for cam_idx, cam in enumerate(cams):
        off_path = os.path.join(off_root, off_rec["image_paths"][cam_idx])
        clo_path = os.path.join(clo_root, clo_rec["image_paths"][cam_idx])
        off_img = cv2.imread(off_path)
        clo_img = cv2.imread(clo_path)
        if off_img is None or clo_img is None:
            print(f"  warn: could not read {off_path if off_img is None else clo_path}")
            continue
        # Match heights so they hstack cleanly.
        h = min(off_img.shape[0], clo_img.shape[0])
        off_img = cv2.resize(off_img, (int(off_img.shape[1] * h / off_img.shape[0]), h))
        clo_img = cv2.resize(clo_img, (int(clo_img.shape[1] * h / clo_img.shape[0]), h))
        off_lines = [
            f"OFFLINE frame={_key(off_rec)} cam={cam}",
            f"live_pos={np.round(off_rec['live_pos'], 4).tolist()}",
            f"pred[0]={np.round(off_rec['pred_actions'][0][:3], 4).tolist()}",
        ]
        clo_lines = [
            f"CLOSED   step={_key(clo_rec)} cam={cam}",
            f"live_pos={np.round(clo_rec['live_pos'], 4).tolist()}",
            f"pred[0]={np.round(clo_rec['pred_actions'][0][:3], 4).tolist()}",
        ]
        off_ann = _annotate(off_img, off_lines)
        clo_ann = _annotate(clo_img, clo_lines)
        rows.append(np.hstack([off_ann, clo_ann]))

    if not rows:
        return
    target_w = max(r.shape[1] for r in rows)
    rows = [
        cv2.copyMakeBorder(
            r, 0, 0, 0, target_w - r.shape[1], cv2.BORDER_CONSTANT, value=0
        )
        for r in rows
    ]
    grid = np.vstack(rows)
    out_path = os.path.join(vis_dir, f"pair_{pair_idx:03d}.jpg")
    cv2.imwrite(out_path, grid, [int(cv2.IMWRITE_JPEG_QUALITY), 85])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline", required=True)
    p.add_argument("--closed", required=True)
    p.add_argument("--offline-image-root", default=None)
    p.add_argument("--closed-image-root", default=None)
    p.add_argument("--n", type=int, default=10, help="number of pairs to print/render")
    p.add_argument(
        "--vis-dir", default=None, help="if set, write side-by-side image grids here"
    )
    args = p.parse_args()

    off = _load(args.offline)
    clo = _load(args.closed)
    n = min(len(off), len(clo), args.n)
    print(
        f"loaded offline={len(off)} closed-loop={len(clo)}; comparing first {n} entries\n"
    )

    off_root = args.offline_image_root or os.path.dirname(os.path.abspath(args.offline))
    clo_root = args.closed_image_root or os.path.dirname(os.path.abspath(args.closed))
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    print(
        f"{'i':>3}  {'off frame':>9}  {'clo step':>8}  "
        f"{'off live xyz':>32}  {'clo live xyz':>32}  "
        f"{'|Δlive|':>7}  {'off pred[0] xyz':>32}  {'clo pred[0] xyz':>32}  "
        f"{'|Δpred[0]|':>9}"
    )
    live_diffs = []
    pred_diffs = []
    for i in range(n):
        a, b = off[i], clo[i]
        a_live = np.array(a["live_pos"])
        b_live = np.array(b["live_pos"])
        a_pred = np.array(a["pred_actions"][0])
        b_pred = np.array(b["pred_actions"][0])
        live_diff = np.linalg.norm(a_live - b_live)
        pred_diff = np.linalg.norm(a_pred[:3] - b_pred[:3])
        live_diffs.append(live_diff)
        pred_diffs.append(pred_diff)
        print(
            f"{i:>3}  {_key(a):>9}  {_key(b):>8}  "
            f"{str(a_live.round(4).tolist()):>32}  {str(b_live.round(4).tolist()):>32}  "
            f"{live_diff:>7.4f}  "
            f"{str(a_pred[:3].round(4).tolist()):>32}  {str(b_pred[:3].round(4).tolist()):>32}  "
            f"{pred_diff:>9.4f}"
        )
        if args.vis_dir:
            _render_pair(a, b, off_root, clo_root, args.vis_dir, i)

    print("\n=== summary ===")
    print(f"  mean |Δlive|     : {np.mean(live_diffs):.4f} m   (input drift)")
    print(f"  mean |Δpred[0]|  : {np.mean(pred_diffs):.4f} m   (output divergence)")
    print(f"  off instruction  : {off[0]['instruction']!r}")
    print(f"  clo instruction  : {clo[0]['instruction']!r}")
    print(f"  off image paths  : {off[0]['image_paths']}")
    print(f"  clo image paths  : {clo[0]['image_paths']}")
    if args.vis_dir:
        print(f"  side-by-side    : wrote {n} grids to {args.vis_dir}/pair_*.jpg")


if __name__ == "__main__":
    main()
