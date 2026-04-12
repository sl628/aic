#!/usr/bin/env python3
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

"""Entry point for RLT training.

Usage:
    # Phase 1: Pretrain RL token on pre-extracted XVLA embeddings
    python train.py --mode pretrain_rl_token \
        --data_dir /home/yifeng/aic_data \
        --embeddings_dir /home/yifeng/aic_data/embeddings \
        --checkpoint_dir checkpoints/rlt

    # Phase 2: Online RL (requires a running robot/sim environment)
    python train.py --mode online_rl \
        --checkpoint_dir checkpoints/rlt \
        --load_checkpoint checkpoints/rlt/phase1_rl_token.pt \
        --vla_model_dir /home/yifeng/models/xvla-base

    Run prepare_embeddings.py first to generate the embeddings directory.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add parent to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic_rlt import RLTConfig, RLTTrainer
from aic_rlt.models.rl_token import RLTokenConfig
from aic_rlt.models.actor_critic import ActorCriticConfig
from aic_rlt.data.lerobot_dataset import LeRobotEmbeddingDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VLA loading stub
# ---------------------------------------------------------------------------

def load_vla(model_path: str, device: torch.device):
    """Load a pre-trained VLA backbone (π0 or similar).

    Returns a callable that:
        vla(obs) -> (vla_embeddings, action_chunk)
        - vla_embeddings: (1, N, D_vla) – internal layer embeddings (for RL token)
        - action_chunk:   (C, action_dim) – VLA's reference action prediction

    TODO: replace this stub with actual π0 / LeRobot model loading.
    For the AIC challenge, this will typically be the pretrained ACT or π0 policy
    loaded from HuggingFace, with hooks added to extract internal embeddings.
    """
    logger.warning(
        "VLA loading stub is active – returning random embeddings. "
        "Replace load_vla() with actual VLA model loading."
    )

    class _StubVLA:
        def __init__(self):
            self.embed_dim = 7848
            self.num_tokens = 540
            self.action_dim = 7
            self.chunk_length = 10

        def get_embeddings(self, obs) -> torch.Tensor:
            return torch.randn(1, self.num_tokens, self.embed_dim, device=device)

        def get_action_chunk(self, obs):
            import numpy as np
            return np.zeros((self.chunk_length, self.action_dim), dtype=np.float32)

    return _StubVLA()


# ---------------------------------------------------------------------------
# Demo dataset stub
# ---------------------------------------------------------------------------

class DemoDataset(torch.utils.data.Dataset):
    """Dataset of VLA embeddings extracted from demonstration episodes.

    Each item is a dict with key "vla_embeddings": (N, D_vla).

    In practice, generate this by:
      1. Loading π0 on your demonstration data
      2. Extracting internal embeddings at each timestep
      3. Saving to disk (e.g. as .pt files) and loading here
    """

    def __init__(self, demo_dir: str, vla, device: torch.device):
        self.demo_dir = Path(demo_dir)
        self.vla = vla
        self.device = device
        # Gather all saved embedding files
        self.files = sorted(self.demo_dir.glob("*.pt"))
        if not self.files:
            logger.warning(
                f"No .pt files found in {demo_dir}. "
                "Using synthetic data – replace with real demo embeddings."
            )
            self._synthetic = True
            self._length = 200
        else:
            self._synthetic = False

    def __len__(self):
        return self._length if self._synthetic else len(self.files)

    def __getitem__(self, idx):
        if self._synthetic:
            cfg = self.vla
            return {
                "vla_embeddings": torch.randn(cfg.num_tokens, cfg.embed_dim)
            }
        data = torch.load(self.files[idx], map_location="cpu")
        return {"vla_embeddings": data["vla_embeddings"]}


# ---------------------------------------------------------------------------
# Environment stub
# ---------------------------------------------------------------------------

class AICEnvWrapper:
    """Thin wrapper around the AIC environment for RLT training.

    In real usage this wraps the ROS 2 / MuJoCo / IsaacLab AIC env.
    Replace the stub methods with actual environment calls.
    """

    def __init__(self, vla, prop_dim: int = 26):
        self.vla = vla
        self.prop_dim = prop_dim
        self._obs = None

    def reset(self):
        """Reset environment and return initial observation."""
        import numpy as np
        self._obs = {"dummy": True}
        return self._obs

    def step(self, action_chunk):
        """Execute action chunk, return (next_obs, reward, done, info)."""
        import numpy as np
        next_obs = {"dummy": True}
        # Sparse reward: +1 for task success, -1 for failure, 0 otherwise
        reward = 0.0
        done = False
        info = {}
        self._obs = next_obs
        return next_obs, reward, done, info

    def get_prop_state(self, obs):
        import numpy as np
        return np.zeros(self.prop_dim, dtype=np.float32)

    def human_intervention(self):
        return None  # No intervention in automated runs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RLT Training")
    parser.add_argument(
        "--mode",
        choices=["pretrain_rl_token", "offline_rl", "online_rl", "full"],
        default="full",
        help="Training phase to run",
    )
    # Phase 1 (offline) arguments
    parser.add_argument("--data_dir", type=str, default="/home/yifeng/aic_data",
                        help="LeRobot v3.0 dataset root (contains data/, meta/)")
    parser.add_argument("--embeddings_dir", type=str, default="",
                        help="Directory of pre-extracted XVLA embeddings (.pt files)")
    # VLA backend (online RL)
    parser.add_argument("--vla_backend", type=str, default="xvla",
                        choices=["xvla", "pi05"],
                        help="VLA backbone for online RL: 'xvla' (default) or 'pi05'")
    parser.add_argument("--vla_model_dir", type=str, default="/home/yifeng/models/xvla-base",
                        help="XVLA model directory (used when --vla_backend=xvla)")
    parser.add_argument("--pi05_checkpoint", type=str,
                        default="/home/yifeng/workspace/pi05_base/pi05_base",
                        help="Pi0.5 checkpoint directory (used when --vla_backend=pi05)")
    parser.add_argument("--instruction", type=str,
                        default="Insert SFP cable into NIC port")
    # Shared arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/rlt")
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameter overrides
    parser.add_argument("--bc_coeff", type=float, default=1.0)
    parser.add_argument("--n_warmup_steps", type=int, default=2000)
    parser.add_argument("--total_env_steps", type=int, default=50000)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--rl_token_epochs", type=int, default=50)
    parser.add_argument("--chunk_length", type=int, default=10)
    # Phase 2 offline RL arguments
    parser.add_argument("--n_offline_epochs", type=int, default=100,
                        help="Offline gradient epochs for offline_rl mode")
    parser.add_argument("--reward_sigma", type=float, default=0.05,
                        help="Gaussian sigma (meters) for synthetic distance-to-goal reward")
    return parser.parse_args()


def _detect_embedding_dims(embeddings_dir: str) -> tuple:
    """Read the first .pt file to determine (vla_embed_dim, num_vla_tokens).

    Supports two formats:
      - Per-episode: episode_0000.pt with shape (T, num_tokens, embed_dim)
      - Per-frame: ep000_frame0000.pt with shape (num_tokens, embed_dim)
    """
    emb_dir = Path(embeddings_dir)
    # Try per-episode format first
    emb_files = sorted(emb_dir.glob("episode_*.pt"))
    if not emb_files:
        # Try per-frame format
        emb_files = sorted(emb_dir.glob("*.pt"))
    if not emb_files:
        raise FileNotFoundError(f"No .pt embedding files found in {embeddings_dir}")

    data = torch.load(emb_files[0], map_location="cpu", weights_only=False)
    emb = data["vla_embeddings"]
    if emb.ndim == 3:
        # Per-episode: (T, num_tokens, embed_dim)
        num_tokens = emb.shape[1]
        embed_dim = emb.shape[2]
        logger.info(f"Detected per-episode embeddings: T={emb.shape[0]}, num_tokens={num_tokens}, embed_dim={embed_dim}")
    elif emb.ndim == 2:
        # Per-frame: (num_tokens, embed_dim)
        num_tokens = emb.shape[0]
        embed_dim = emb.shape[1]
        logger.info(f"Detected per-frame embeddings: num_tokens={num_tokens}, embed_dim={embed_dim}")
    else:
        raise ValueError(f"Unexpected embedding shape: {emb.shape}")

    return embed_dim, num_tokens


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.mode in ("pretrain_rl_token", "full"):
        # --- Phase 1: offline pretraining ---

        if not args.embeddings_dir:
            raise ValueError(
                "--embeddings_dir is required for pretrain_rl_token. "
                "Run scripts/prepare_embeddings.py first."
            )

        # Auto-detect embedding dimensions from saved files
        vla_embed_dim, num_vla_tokens = _detect_embedding_dims(args.embeddings_dir)

        rl_token_cfg = RLTokenConfig(
            vla_embed_dim=vla_embed_dim,
            num_vla_tokens=num_vla_tokens,
        )
        actor_critic_cfg = ActorCriticConfig(
            rl_token_dim=rl_token_cfg.rl_token_dim,
            action_dim=7,
            prop_dim=26,
            chunk_length=args.chunk_length,
            hidden_dims=args.hidden_dims,
        )
        config = RLTConfig(
            rl_token=rl_token_cfg,
            actor_critic=actor_critic_cfg,
            bc_coeff=args.bc_coeff,
            n_warmup_steps=args.n_warmup_steps,
            total_env_steps=args.total_env_steps,
            rl_token_epochs=args.rl_token_epochs,
            checkpoint_dir=args.checkpoint_dir,
        )

        trainer = RLTTrainer(config=config, device=device)

        if args.load_checkpoint:
            trainer.load_checkpoint(args.load_checkpoint)

        demo_dataset = LeRobotEmbeddingDataset(
            data_dir=args.data_dir,
            embeddings_dir=args.embeddings_dir,
            chunk_length=args.chunk_length,
        )
        trainer.pretrain_rl_token(demo_dataset)
        trainer.save_checkpoint("phase1_rl_token")

    if args.mode == "offline_rl":
        # --- Phase 2 (offline): actor-critic training from demo data ---
        if not args.embeddings_dir:
            raise ValueError(
                "--embeddings_dir is required for offline_rl. "
                "Run scripts/prepare_embeddings.py first."
            )
        if not args.load_checkpoint:
            raise ValueError(
                "--load_checkpoint is required for offline_rl (Phase 1 checkpoint). "
                "Run --mode pretrain_rl_token first."
            )

        vla_embed_dim, num_vla_tokens = _detect_embedding_dims(args.embeddings_dir)

        rl_token_cfg = RLTokenConfig(
            vla_embed_dim=vla_embed_dim,
            num_vla_tokens=num_vla_tokens,
        )
        actor_critic_cfg = ActorCriticConfig(
            rl_token_dim=rl_token_cfg.rl_token_dim,
            action_dim=7,
            prop_dim=26,
            chunk_length=args.chunk_length,
            hidden_dims=args.hidden_dims,
        )
        config = RLTConfig(
            rl_token=rl_token_cfg,
            actor_critic=actor_critic_cfg,
            bc_coeff=args.bc_coeff,
            checkpoint_dir=args.checkpoint_dir,
        )

        trainer = RLTTrainer(config=config, device=device)
        trainer.load_checkpoint(args.load_checkpoint)

        demo_dataset = LeRobotEmbeddingDataset(
            data_dir=args.data_dir,
            embeddings_dir=args.embeddings_dir,
            chunk_length=args.chunk_length,
        )
        trainer.train_offline(
            demo_dataset,
            n_epochs=args.n_offline_epochs,
            reward_sigma=args.reward_sigma,
        )

    if args.mode in ("online_rl", "full"):
        # --- Phase 2: online RL with live VLA backend ---
        from aic_rlt.vla import create_vla_backend

        # Build backend kwargs from the selected backend
        if args.vla_backend == "xvla":
            backend_kwargs = dict(
                model_dir=args.vla_model_dir,
                instruction=args.instruction,
                chunk_length=args.chunk_length,
            )
        else:  # pi05
            backend_kwargs = dict(
                checkpoint_dir=args.pi05_checkpoint,
                instruction=args.instruction,
                chunk_length=args.chunk_length,
            )

        logger.info(f"Loading VLA backend: {args.vla_backend}")
        vla = create_vla_backend(args.vla_backend, device=device, **backend_kwargs)

        # Build RLT config from VLA dimensions (authoritative source)
        rl_token_cfg = RLTokenConfig(
            vla_embed_dim=vla.embed_dim,
            num_vla_tokens=vla.num_tokens,
        )
        actor_critic_cfg = ActorCriticConfig(
            rl_token_dim=rl_token_cfg.rl_token_dim,
            action_dim=7,
            prop_dim=26,
            chunk_length=args.chunk_length,
            hidden_dims=args.hidden_dims,
        )
        config = RLTConfig(
            rl_token=rl_token_cfg,
            actor_critic=actor_critic_cfg,
            bc_coeff=args.bc_coeff,
            n_warmup_steps=args.n_warmup_steps,
            total_env_steps=args.total_env_steps,
            checkpoint_dir=args.checkpoint_dir,
        )

        env = AICEnvWrapper(prop_dim=actor_critic_cfg.prop_dim)

        trainer = RLTTrainer(
            config=config,
            device=device,
            get_vla_embeddings=lambda obs: vla.get_embeddings(obs),
            get_vla_action_chunk=lambda obs: vla.get_action_chunk(obs),
            get_prop_state=lambda obs: env.get_prop_state(obs),
            env_step=lambda a: env.step(a),
            human_intervention=env.human_intervention,
        )

        if args.load_checkpoint:
            trainer.load_checkpoint(args.load_checkpoint)

        trainer.train(initial_obs_fn=env.reset)


if __name__ == "__main__":
    main()
