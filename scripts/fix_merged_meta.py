#!/usr/bin/env python3
"""Add missing video timestamps to the merged dataset's episode metadata."""
import sys
import types
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if "lerobot.policies" not in sys.modules:
    _dummy = types.ModuleType("lerobot.policies")
    _dummy.__path__ = [__import__("lerobot").__path__[0] + "/policies"]
    sys.modules["lerobot.policies"] = _dummy

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("siyulw2025/cableholder-all")
meta_root = Path(ds.root) / "meta"
ep_parquet = meta_root / "episodes" / "chunk-000" / "file-000.parquet"

ep_df = pd.read_parquet(ep_parquet)
print(f"Columns before: {[c for c in ep_df.columns if 'timestamp' in c or 'camera' in c]}")

fps = 20.0
cams = ["observation.images.left_camera", "observation.images.center_camera", "observation.images.right_camera"]

for cam in cams:
    for i, row in ep_df.iterrows():
        length = row["length"]
        ep_df.at[i, f"videos/{cam}/from_timestamp"] = 0.0
        ep_df.at[i, f"videos/{cam}/to_timestamp"] = length / fps

ep_df.to_parquet(ep_parquet, index=False)
print(f"Updated {len(ep_df)} episode metadata rows with video timestamps")
print(f"Columns after: {[c for c in ep_df.columns if 'timestamp' in c or 'camera' in c]}")
