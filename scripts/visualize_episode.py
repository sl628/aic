#!/usr/bin/env python3
"""Visualize collected episodes in Rerun.

Usage:
    source ~/rerun_env/bin/activate
    python scripts/visualize_episode.py --data_dir /home/yifeng/aic_data_sym --episode 0
    python scripts/visualize_episode.py --data_dir /home/yifeng/aic_data_sym --episode 0 5 10
    python scripts/visualize_episode.py --data_dir /home/yifeng/aic_data_sym --episode 0 --save ep0.rrd
    python scripts/visualize_episode.py --data_dir /home/yifeng/aic_data_sym --episode 0 --no-images
"""

import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rerun as rr
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Visualize AIC episode in Rerun")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--episode", type=int, nargs="+", default=[0])
    p.add_argument("--save", type=str, default="")
    p.add_argument(
        "--no-images", action="store_true", help="Skip camera images (faster)"
    )
    return p.parse_args()


def log_episode(data_dir: Path, ep_idx: int, log_images: bool):
    pf = data_dir / "data" / "chunk-000" / f"file-{ep_idx:03d}.parquet"
    if not pf.exists():
        print(f"  Parquet not found: {pf}, skipping")
        return

    table = pq.read_table(pf)
    d = table.to_pydict()
    T = len(d["frame_index"])
    print(f"  Episode {ep_idx}: {T} frames")

    pfx = f"ep{ep_idx:03d}"
    states = np.array(d["observation.state"], dtype=np.float32)
    actions = np.array(d["action"], dtype=np.float32)

    tcp_trail = []
    for t in range(T):
        rr.set_time_sequence("frame", t)
        rr.set_time_seconds("time", d["timestamp"][t])

        tcp = states[t, 0:3]
        act = actions[t, 0:3]
        tcp_trail.append(tcp.tolist())

        rr.log(
            f"{pfx}/world/tcp", rr.Points3D([tcp], radii=[0.005], colors=[[0, 200, 0]])
        )
        rr.log(
            f"{pfx}/world/target",
            rr.Points3D([act], radii=[0.004], colors=[[200, 0, 0]]),
        )
        rr.log(
            f"{pfx}/world/trail", rr.LineStrips3D([tcp_trail], colors=[[0, 200, 0, 80]])
        )

        rr.log(f"{pfx}/tcp/x", rr.Scalars(states[t, 0]))
        rr.log(f"{pfx}/tcp/y", rr.Scalars(states[t, 1]))
        rr.log(f"{pfx}/tcp/z", rr.Scalars(states[t, 2]))
        rr.log(f"{pfx}/tcp_err", rr.Scalars(float(np.linalg.norm(states[t, 13:16]))))
        rr.log(
            f"{pfx}/tcp_err_rot", rr.Scalars(float(np.linalg.norm(states[t, 16:19])))
        )
        rr.log(f"{pfx}/vel", rr.Scalars(float(np.linalg.norm(states[t, 7:10]))))

        for j in range(7):
            rr.log(f"{pfx}/joints/j{j}", rr.Scalars(states[t, 19 + j]))

        if log_images:
            for cam in ["center_camera", "left_camera", "right_camera"]:
                entry = d[f"observation.images.{cam}"][t]
                raw = entry["bytes"] if isinstance(entry, dict) else entry
                img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
                rr.log(f"{pfx}/cam/{cam}", rr.Image(img))


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    rr.init("aic_data", spawn=not args.save)
    if args.save:
        rr.save(args.save)

    for ep in args.episode:
        log_episode(data_dir, ep, log_images=not args.no_images)

    print("Done.")


if __name__ == "__main__":
    main()
