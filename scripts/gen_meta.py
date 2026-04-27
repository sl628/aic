#!/usr/bin/env python3
"""Regenerate aic_meta.json for all episodes in the output dir."""
import json
from pathlib import Path

root = Path("/home/yifeng/aic_xvla_data")
ep_dir = root / "episodes"
eps = sorted(ep_dir.iterdir(), key=lambda p: int(p.name.split("_")[1]))

meta = {
    "dataset_name": "aic",
    "fps": 20,
    "datalist": [
        {
            "parquet_path": str((ep / "data.parquet").resolve()),
            "image_root": str(root.resolve()),
            "instruction": "insert the SFP cable into the port",
            "fps": 20,
        }
        for ep in eps
        if (ep / "data.parquet").exists()
    ],
}

meta_path = root / "aic_meta.json"
meta_path.write_text(json.dumps(meta, indent=2))
print(f"Written {len(meta['datalist'])} episodes to {meta_path}")
