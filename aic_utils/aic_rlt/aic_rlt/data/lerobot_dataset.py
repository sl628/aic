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

"""PyTorch dataset for RLT Phase 1 training over LeRobot v3.0 parquet data.

Reads pre-extracted XVLA embeddings (from prepare_embeddings.py) alongside
proprioceptive state and action chunks from parquet files.

Each sample is a sliding window of chunk_length consecutive frames within
an episode. The dataset yields:
    {
        "vla_embeddings": Tensor(num_tokens, embed_dim),  # single timestep
        "prop":           Tensor(26,),
        "action_chunk":   Tensor(chunk_length, 7),
    }
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from aic_rlt.vla.xvla_wrapper import quat_actions_to_rot6d

logger = logging.getLogger(__name__)


class LeRobotEmbeddingDataset(Dataset):
    """Dataset pairing pre-extracted VLA embeddings with proprioception + actions.

    Args:
        data_dir:      Root of the LeRobot v3.0 dataset (contains data/, meta/).
        embeddings_dir: Directory of per-episode .pt files from prepare_embeddings.py.
        chunk_length:  Number of consecutive frames per sample (action chunk size C).
        split:         Unused; all episodes in data_dir are loaded.
    """

    def __init__(
        self,
        data_dir: str,
        embeddings_dir: str,
        chunk_length: int = 10,
    ):
        self.data_dir = Path(data_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.chunk_length = chunk_length

        self._samples: List[Dict] = []   # list of {ep_idx, frame_start}
        self._episodes: Dict[int, Dict] = {}  # ep_idx → {prop, actions, embeddings}

        self._load_all_episodes()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_all_episodes(self) -> None:
        """Load all parquet files and pre-extracted embeddings into memory."""
        import pyarrow.parquet as pq

        parquet_files = sorted(self.data_dir.glob("data/**/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {self.data_dir}/data/")

        logger.info(f"Loading {len(parquet_files)} parquet files ...")

        # Collect all rows grouped by episode_index
        episode_rows: Dict[int, List] = {}
        for pf in parquet_files:
            table = pq.read_table(pf, columns=["episode_index", "frame_index",
                                                "observation.state", "action"])
            for row in table.to_pydict():
                pass
            # Convert to per-row dicts
            d = table.to_pydict()
            n = len(d["episode_index"])
            for i in range(n):
                ep = int(d["episode_index"][i])
                episode_rows.setdefault(ep, []).append({
                    "frame_index": int(d["frame_index"][i]),
                    "prop": np.array(d["observation.state"][i], dtype=np.float32),
                    "action": np.array(d["action"][i], dtype=np.float32),
                })

        # Sort frames within each episode and build sample index
        for ep_idx, rows in sorted(episode_rows.items()):
            rows.sort(key=lambda r: r["frame_index"])
            T = len(rows)

            props = np.stack([r["prop"] for r in rows])     # (T, 26)
            actions_quat = np.stack([r["action"] for r in rows])  # (T, 7)
            actions = quat_actions_to_rot6d(actions_quat)         # (T, 9)

            # Load pre-extracted embeddings
            emb_path = self.embeddings_dir / f"episode_{ep_idx:04d}.pt"
            if not emb_path.exists():
                logger.warning(f"Embedding file not found: {emb_path} — skipping episode {ep_idx}")
                continue

            emb_data = torch.load(emb_path, map_location="cpu", weights_only=False)
            embeddings = emb_data["vla_embeddings"]   # (T, num_tokens, embed_dim)

            if embeddings.shape[0] != T:
                logger.warning(
                    f"Episode {ep_idx}: embedding frames ({embeddings.shape[0]}) "
                    f"≠ parquet frames ({T}) — skipping"
                )
                continue

            # Optional: pre-extracted VLA reference action chunks (T, C, 7).
            # If present, Phase 2 offline RL will use these as ref_action_chunk
            # instead of copying demo actions, closing the train/deploy gap.
            ref_actions = emb_data.get("ref_actions", None)
            if ref_actions is not None:
                if ref_actions.shape[0] != T:
                    logger.warning(
                        f"Episode {ep_idx}: ref_actions frames ({ref_actions.shape[0]}) "
                        f"≠ parquet frames ({T}) — dropping ref_actions for this ep"
                    )
                    ref_actions = None
                elif ref_actions.shape[1] != self.chunk_length:
                    logger.warning(
                        f"Episode {ep_idx}: ref_actions chunk_length "
                        f"({ref_actions.shape[1]}) ≠ dataset chunk_length "
                        f"({self.chunk_length}) — dropping ref_actions"
                    )
                    ref_actions = None
                else:
                    ref_np = ref_actions.float().numpy()
                    # Convert 7D quat ref_actions → 9D rot6d if needed
                    if ref_np.shape[-1] == 7:
                        ref_np = quat_actions_to_rot6d(ref_np)
                    ref_actions = ref_np

            # Phase-conditioned embeddings: dict name→Tensor(T, N, D)
            phase_embeddings = emb_data.get("phase_embeddings", None)
            if phase_embeddings is not None:
                valid = True
                for pname, pemb in phase_embeddings.items():
                    if pemb.shape[0] != T:
                        logger.warning(
                            f"Episode {ep_idx}: phase_embeddings[{pname}] T "
                            f"mismatch ({pemb.shape[0]} vs {T}) — dropping"
                        )
                        valid = False
                        break
                if not valid:
                    phase_embeddings = None

            phase_ref_actions = emb_data.get("phase_ref_actions", None)
            if phase_ref_actions is not None:
                for pname, pref in phase_ref_actions.items():
                    if pref.shape[0] != T or pref.shape[1] != self.chunk_length:
                        logger.warning(
                            f"Episode {ep_idx}: phase_ref_actions[{pname}] "
                            f"shape mismatch — dropping"
                        )
                        phase_ref_actions = None
                        break
                if phase_ref_actions is not None:
                    phase_ref_actions = {
                        k: quat_actions_to_rot6d(v.float().numpy())
                        if v.shape[-1] == 7 else v.float().numpy()
                        for k, v in phase_ref_actions.items()
                    }

            self._episodes[ep_idx] = {
                "props": props,
                "actions": actions,
                "embeddings": embeddings,
                "ref_actions": ref_actions,
                "phase_embeddings": phase_embeddings,
                "phase_ref_actions": phase_ref_actions,
                "T": T,
            }

            # Create one sample per valid starting frame
            for t in range(T - self.chunk_length + 1):
                self._samples.append({"ep_idx": ep_idx, "frame_start": t})

        logger.info(
            f"Loaded {len(self._episodes)} episodes, "
            f"{len(self._samples)} samples (chunk_length={self.chunk_length})"
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self._samples[idx]
        ep = self._episodes[s["ep_idx"]]
        t = s["frame_start"]
        C = self.chunk_length

        # Single-frame VLA embedding (anchor frame of the chunk)
        vla_emb = ep["embeddings"][t]                    # (num_tokens, embed_dim)

        # Proprioception at anchor frame
        prop = torch.from_numpy(ep["props"][t])          # (26,)

        # Action chunk — C consecutive ground-truth actions
        action_chunk = torch.from_numpy(ep["actions"][t : t + C])  # (C, 7)

        return {
            "vla_embeddings": vla_emb,
            "prop": prop,
            "action_chunk": action_chunk,
        }
