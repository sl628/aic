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

"""RLT training loop – Algorithm 1 of the paper.

This module implements the complete training procedure:

Phase 1 – RL Token adaptation (Section III-A):
    - (Optionally) fine-tune the VLA backbone with supervised fine-tuning
    - Train RL token encoder-decoder on a small demo dataset
    - Objective: autoregressive reconstruction loss L_ro (equation (2))

Phase 2 – Online RL with actor-critic (Section III-B):
    - Freeze VLA + RL token encoder
    - Roll out the VLA to collect warm-up transitions into replay buffer B
    - Alternate between:
        a) Rollout: collect (x_t, a_{t:t+C-1}, r_t, x_{t+1}) using actor
           (with optional human intervention)
        b) Update: sample batch from B, perform G gradient steps:
            - Critic update: TD backup (equation (3))
            - Actor update:  Q-maximization + BC regularizer (equations (4,5))

Hyperparameters (Table not in paper, from Appendix B):
    - Chunk length C = 10
    - Subsample stride 2 → 5 transitions per chunk stored
    - Update-to-data ratio = 5 (G = 5 * steps collected)
    - 2 critic updates per actor update
    - Discount γ per step; C-step return with γ^C terminal bootstrap
    - TD3 target networks (EMA of critic)
    - BC regularizer coefficient β
"""

import copy
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.actor_critic import Actor, ActorCriticConfig, Critic
from .models.rl_token import RLTokenConfig, RLTokenModel
from .replay_buffer import ReplayBuffer, Transition

logger = logging.getLogger(__name__)

# Phase ids used by structured reward + phase-conditioned prompts.
PHASE_APPROACH = 0
PHASE_ALIGN = 1
PHASE_INSERT = 2
PHASE_VERIFY = 3
PHASE_NAMES = ("approach", "align", "insert", "verify")

DEFAULT_PHASE_PROMPTS = (
    "move the SFP cable above the NIC port",
    "align the SFP connector with the port opening",
    "insert the SFP cable straight into the port",
    "verify the SFP cable is fully seated in the port",
)


@dataclass
class RewardConfig:
    """Structured reward configuration for SFP→NIC insertion.

    mode="legacy" reproduces the original distance-only Gaussian reward.
    mode="structured" enables the phase-aware multi-term reward.
    """

    mode: str = "legacy"  # "legacy" | "structured"

    # Term weights (structured only)
    w_pos: float = 1.0
    w_ori: float = 0.3
    w_depth: float = 0.5
    w_contact: float = 0.2
    w_phase_bonus: float = 1.0
    w_success: float = 10.0

    # Position Gaussian width (meters)
    sigma_pos: float = 0.03

    # Port frame override. If None, port is per-episode demo endpoint.
    # port_pos: (x,y,z); port_quat: (qx,qy,qz,qw)
    port_pos: Optional[Tuple[float, float, float]] = None
    port_quat: Optional[Tuple[float, float, float, float]] = None

    # Insertion axis in port frame (0=x, 1=y, 2=z). Depth measured along
    # this axis of the port frame; positive = deeper into the port.
    port_insert_axis: int = 2
    insert_depth_m: float = 0.03  # ramp range for depth reward

    # Contact term uses tcp_err as a wrench proxy (stiffness * err ≈ force).
    # Reward axial err in band; penalize lateral err.
    contact_axial_band: Tuple[float, float] = (0.001, 0.005)  # meters
    contact_lateral_scale: float = 0.005

    # Phase thresholds (all in meters / 1-|q·qg|)
    d_align_thresh: float = 0.03
    ori_align_thresh: float = 0.05
    d_verify_thresh: float = 0.005

    # Reward normalization: normalize chunk returns to zero mean / unit std
    # before storing in the replay buffer. Prevents Q-value divergence with
    # large reward scales.
    normalize: bool = True


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class RLTConfig:
    # ---- RL Token ----
    rl_token: RLTokenConfig = field(default_factory=RLTokenConfig)

    # ---- Actor / Critic ----
    actor_critic: ActorCriticConfig = field(default_factory=ActorCriticConfig)

    # ---- Replay Buffer ----
    replay_buffer_capacity: int = 200_000

    # ---- Warmup ----
    # Number of environment steps to collect with frozen VLA before RL starts
    n_warmup_steps: int = 2_000

    # ---- Online RL ----
    # Total environment steps
    total_env_steps: int = 50_000
    # Number of gradient updates per environment step (update-to-data ratio)
    updates_per_step: int = 5
    # Number of critic updates per actor update
    critic_updates_per_actor_update: int = 2
    # Batch size for gradient updates
    batch_size: int = 256
    # Discount factor (per-step)
    gamma: float = 0.99
    # TD3 target network EMA coefficient
    tau: float = 0.005
    # BC regularizer coefficient β (equation (5))
    bc_coeff: float = 5.0
    # TD3 target policy noise
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5

    # ---- RL Token pretraining ----
    rl_token_lr: float = 3e-4
    rl_token_epochs: int = 50
    rl_token_batch_size: int = 32

    # ---- Actor / Critic LR ----
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # ---- Reward shaping (offline RL) ----
    reward: RewardConfig = field(default_factory=RewardConfig)

    # ---- Wandb ----
    wandb_enabled: bool = False
    wandb_project: str = "aic-rlt"
    wandb_run_name: Optional[str] = None

    # ---- Logging / Saving ----
    log_interval: int = 100  # gradient steps
    save_interval: int = 1000  # gradient steps
    checkpoint_dir: str = "checkpoints/rlt"


# ---------------------------------------------------------------------------
# Reward / phase helpers
# ---------------------------------------------------------------------------


def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    """Quaternion [qx,qy,qz,qw] → 3x3 rotation matrix."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _infer_phases(
    props: np.ndarray,
    goal_pos: np.ndarray,
    goal_quat: np.ndarray,
    rcfg: RewardConfig,
) -> np.ndarray:
    """Label each frame with a phase id based on pose proximity to the goal.

    Logic (nested):
      verify  ← d ≤ d_verify_thresh
      inside the d_align_thresh ball:
        insert ← orientation already aligned AND axial-contact proxy is
                 non-trivial (|axial tcp_err| above the band lower bound)
        align  ← otherwise (close in position, but still rotating or free)
      approach ← outside the alignment ball

    No wrench is available in props[26], so axial tcp_err (projection onto the
    port insert axis) is used as a contact proxy — stiffness × axial_err ≈
    axial force. Lateral err does not count as "in contact" for this purpose.
    """
    T = props.shape[0]
    phases = np.full(T, PHASE_APPROACH, dtype=np.int64)
    R_port = _quat_xyzw_to_mat(goal_quat)
    port_axis = R_port[:, rcfg.port_insert_axis]  # world-frame unit vec
    contact_lo = float(rcfg.contact_axial_band[0])

    d = np.linalg.norm(props[:, 0:3] - goal_pos, axis=1)  # (T,)
    o_err = 1.0 - np.abs(props[:, 3:7] @ goal_quat)  # (T,)
    axial_err = np.abs(props[:, 13:16] @ port_axis)  # (T,)

    # Order matters: later assignments overwrite earlier ones.
    align_mask = d <= rcfg.d_align_thresh
    phases[align_mask] = PHASE_ALIGN

    insert_mask = (
        align_mask & (o_err <= rcfg.ori_align_thresh) & (axial_err > contact_lo)
    )
    phases[insert_mask] = PHASE_INSERT

    verify_mask = d <= rcfg.d_verify_thresh
    phases[verify_mask] = PHASE_VERIFY

    return phases


def _compute_structured_reward(
    props: np.ndarray,
    goal_pos: np.ndarray,
    goal_quat: np.ndarray,
    phases: np.ndarray,
    rcfg: RewardConfig,
) -> np.ndarray:
    """Per-frame scalar reward combining position, orientation, insertion
    depth, contact band, and phase-transition bonuses."""
    T = props.shape[0]
    R_port = _quat_xyzw_to_mat(goal_quat)
    port_axis = R_port[:, rcfg.port_insert_axis]  # (3,) world-frame unit vec

    pos = props[:, 0:3]  # (T, 3)
    quat = props[:, 3:7]  # (T, 4)
    err = props[:, 13:16]  # (T, 3) controller pose error (proxy for contact)

    # 1. Position Gaussian → [0, 1]
    d = np.linalg.norm(pos - goal_pos, axis=1)
    r_pos = np.exp(-(d**2) / (rcfg.sigma_pos**2))

    # 2. Orientation alignment → [0, 1]
    r_ori = np.abs(quat @ goal_quat)

    # 3. Insertion depth ramp → [0, 1]
    proj = (pos - goal_pos) @ port_axis
    r_depth = np.clip(-proj / max(rcfg.insert_depth_m, 1e-6), 0.0, 1.0)

    # 4. Contact term
    axial_scalar = err @ port_axis  # (T,)
    axial_err = np.abs(axial_scalar)
    lateral_vec = err - np.outer(axial_scalar, port_axis)  # (T, 3)
    lateral_err = np.linalg.norm(lateral_vec, axis=1)
    lo, hi = rcfg.contact_axial_band
    axial_in_band = ((axial_err >= lo) & (axial_err <= hi)).astype(np.float32)
    r_contact = np.clip(
        axial_in_band - lateral_err / max(rcfg.contact_lateral_scale, 1e-6),
        -1.0,
        1.0,
    )

    # Weighted sum
    rewards = (
        rcfg.w_pos * r_pos
        + rcfg.w_ori * r_ori
        + rcfg.w_depth * r_depth
        + rcfg.w_contact * r_contact
    ).astype(np.float32)

    # 5. Sparse phase-transition bonus
    phase_advanced = np.zeros(T, dtype=np.float32)
    phase_advanced[1:] = (phases[1:] > phases[:-1]).astype(np.float32)
    rewards += rcfg.w_phase_bonus * phase_advanced

    # 6. Success bonus on verify
    rewards += rcfg.w_success * (phases == PHASE_VERIFY).astype(np.float32)

    return rewards


# ---------------------------------------------------------------------------
# RLT Trainer
# ---------------------------------------------------------------------------


class RLTTrainer:
    """Implements Algorithm 1 of the RLT paper.

    The trainer is environment-agnostic: callers provide VLA and env
    interaction callbacks, keeping this class free of ROS / simulation deps.

    Expected callback signatures:

        get_vla_embeddings(obs) -> torch.Tensor  (B=1, N, D_vla)
            Run a single observation through the frozen VLA and return its
            internal layer embeddings.

        get_vla_action_chunk(obs) -> np.ndarray  (C, action_dim)
            Run VLA to get its reference action chunk prediction.

        get_prop_state(obs) -> np.ndarray  (prop_dim,)
            Extract proprioceptive features from obs.

        env_step(action_chunk) -> (obs_next, reward, done, info)
            Apply the *first* action from the chunk (or the whole chunk with
            the environment handling timing), return next obs + reward signal.

        human_intervention() -> Optional[np.ndarray]  (C, action_dim) or None
            Return human-corrected action chunk if operator intervenes, else None.

        episode_done() -> bool
            True if the operator signals task success/failure (terminal reward).
    """

    def __init__(
        self,
        config: RLTConfig,
        device: torch.device,
        # Callbacks – set before calling train()
        get_vla_embeddings: Optional[Callable] = None,
        get_vla_action_chunk: Optional[Callable] = None,
        get_prop_state: Optional[Callable] = None,
        env_step: Optional[Callable] = None,
        human_intervention: Optional[Callable] = None,
    ):
        self.config = config
        self.device = device

        self.get_vla_embeddings = get_vla_embeddings
        self.get_vla_action_chunk = get_vla_action_chunk
        self.get_prop_state = get_prop_state
        self.env_step = env_step
        self.human_intervention = human_intervention

        # ---- Models ----
        self.rl_token_model = RLTokenModel(config.rl_token).to(device)
        self.actor = Actor(config.actor_critic).to(device)
        self.critic = Critic(config.actor_critic).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ---- Optimizers ----
        self.rl_token_optimizer = torch.optim.AdamW(
            self.rl_token_model.parameters(), lr=config.rl_token_lr
        )
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=config.critic_lr
        )

        # ---- Replay buffer ----
        cfg = config.actor_critic
        self.replay_buffer = ReplayBuffer(
            capacity=config.replay_buffer_capacity,
            rl_token_dim=cfg.rl_token_dim,
            prop_dim=cfg.prop_dim,
            action_dim=cfg.action_dim,
            chunk_length=cfg.chunk_length,
            device=device,
        )

        self._total_gradient_steps = 0
        self._checkpoint_dir = Path(config.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._wandb = None
        if config.wandb_enabled:
            import wandb

            self._wandb = wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config={
                    "rl_token_dim": config.rl_token.rl_token_dim,
                    "vla_embed_dim": config.rl_token.vla_embed_dim,
                    "action_dim": config.actor_critic.action_dim,
                    "chunk_length": config.actor_critic.chunk_length,
                    "batch_size": config.batch_size,
                    "gamma": config.gamma,
                    "bc_coeff": config.bc_coeff,
                    "actor_lr": config.actor_lr,
                    "critic_lr": config.critic_lr,
                    "reward_mode": config.reward.mode,
                    "replay_buffer_capacity": config.replay_buffer_capacity,
                },
            )
            logger.info("Wandb initialized: project=%s", config.wandb_project)

    # ------------------------------------------------------------------
    # Phase 1: RL Token pretraining
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Phase 2 (Offline): Actor-Critic training on demo data
    # ------------------------------------------------------------------

    def train_offline(
        self,
        demo_dataset: "torch.utils.data.Dataset",
        n_epochs: int = 100,
        reward_sigma: Optional[float] = None,
    ) -> None:
        """Offline actor-critic training using demo data with synthetic rewards (TD3+BC).

        Since no simulator is available, rewards are computed from each demo episode's
        final TCP position (which equals the successful insertion pose for all 100 demos):

            r(t) = exp(-||pos(t) - pos(T)||² / reward_sigma²)

        This gives dense, well-shaped reward: 1.0 at the final frame, decaying
        exponentially as distance to goal increases (sigma=5cm by default).

        Transitions are pre-computed and stored in the replay buffer, then
        offline TD3+BC gradient updates are run for n_epochs.

        Args:
            demo_dataset: LeRobotEmbeddingDataset with _episodes attribute.
            n_epochs:     Gradient epochs (one epoch ≈ buffer_size / batch_size steps).
            reward_sigma: Legacy-mode Gaussian width (meters). If provided and
                          reward mode is "legacy", overrides config.reward.sigma_pos.
                          Ignored for "structured" mode.
        """
        logger.info(
            "=== Phase 2 (Offline): Actor-Critic Training (reward_mode=%s) ===",
            self.config.reward.mode,
        )

        # Legacy sigma override for backward compatibility with old CLI.
        if reward_sigma is not None and self.config.reward.mode == "legacy":
            self.config.reward.sigma_pos = float(reward_sigma)

        # Freeze RL token
        for p in self.rl_token_model.parameters():
            p.requires_grad = False
        self.rl_token_model.eval()

        # Pre-compute z_rl and populate replay buffer episode by episode
        logger.info("Encoding RL tokens and populating replay buffer ...")
        self._populate_replay_buffer_from_demos(demo_dataset)
        logger.info(f"Replay buffer size: {len(self.replay_buffer)} transitions")
        if self._wandb:
            self._wandb.log({"phase2/replay_buffer_size": len(self.replay_buffer)})

        # CSV logging for Phase 2
        log_dir = self._checkpoint_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / "phase2_metrics.csv"
        with open(csv_path, "w") as f:
            f.write("step,critic_loss,actor_loss\n")

        # Offline gradient updates
        steps_per_epoch = max(1, len(self.replay_buffer) // self.config.batch_size)
        n_steps = n_epochs * steps_per_epoch
        logger.info(
            f"Offline updates: {n_epochs} epochs × {steps_per_epoch} steps = {n_steps} total"
        )

        for step in range(n_steps):
            metrics = self._gradient_update_step()

            if (step + 1) % self.config.log_interval == 0:
                log_str = f"  offline step={step+1}/{n_steps}"
                for k, v in metrics.items():
                    log_str += f"  {k}={v:.4f}"
                logger.info(log_str)
                with open(csv_path, "a") as f:
                    c_loss = metrics.get("critic_loss", 0.0)
                    a_loss = metrics.get("actor_loss", 0.0)
                    f.write(f"{step+1},{c_loss:.6f},{a_loss:.6f}\n")
                if self._wandb:
                    self._wandb.log(
                        {
                            "phase2/step": step + 1,
                            "phase2/critic_loss": c_loss,
                            "phase2/actor_loss": a_loss,
                            "phase2/epoch": (step + 1) / steps_per_epoch,
                        },
                        step=step + 1,
                    )

            if (step + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"offline_step_{step+1}")

        logger.info("Offline training complete.")
        self.save_checkpoint("phase2_offline")

    def _populate_replay_buffer_from_demos(
        self,
        demo_dataset: "torch.utils.data.Dataset",
    ) -> None:
        """Encode all demo frames with the frozen RL token and add to replay buffer.

        Reward is either legacy (position-only Gaussian) or structured
        (position + orientation + depth + contact + phase bonuses), controlled
        by self.config.reward.mode. If a port pose is configured, it replaces
        the per-episode demo-endpoint goal with a fixed port-frame goal — this
        is what makes the policy target the *port*, not a world-coordinate
        point that happened to match the demos.
        """
        C = self.config.actor_critic.chunk_length
        encode_batch = 64  # frames per GPU batch for encoding
        rcfg = self.config.reward

        # Auto-size replay buffer so all demo transitions fit.
        total_transitions = sum(
            max(ep["T"] - C + 1, 0) for ep in demo_dataset._episodes.values()
        )
        needed = int(max(total_transitions * 1.1, self.replay_buffer.capacity))
        if needed > self.replay_buffer.capacity:
            logger.info(
                "Resizing replay buffer: %d → %d (dataset has %d transitions)",
                self.replay_buffer.capacity,
                needed,
                total_transitions,
            )
            cfg = self.config.actor_critic
            self.replay_buffer = ReplayBuffer(
                capacity=needed,
                rl_token_dim=cfg.rl_token_dim,
                prop_dim=cfg.prop_dim,
                action_dim=cfg.action_dim,
                chunk_length=cfg.chunk_length,
                device=self.device,
            )

        # Resolve port-frame goal (shared across episodes, if provided).
        fixed_port_pos: Optional[np.ndarray] = None
        fixed_port_quat: Optional[np.ndarray] = None
        if rcfg.port_pos is not None and rcfg.port_quat is not None:
            fixed_port_pos = np.asarray(rcfg.port_pos, dtype=np.float64)
            fixed_port_quat = np.asarray(rcfg.port_quat, dtype=np.float64)
            logger.info(
                "Using fixed port pose: pos=%s quat=%s",
                fixed_port_pos.tolist(),
                fixed_port_quat.tolist(),
            )
        else:
            logger.info(
                "No port pose configured — using per-episode demo endpoint "
                "(legacy behavior; policy will target world coords, not the port)."
            )

        # Pass 1: encode episodes and compute raw chunk returns.
        # Transitions are accumulated into a list so we can normalize rewards
        # across the entire dataset before committing to the replay buffer.
        pending: List[Tuple[Transition, int]] = []  # (transition, ep_idx)
        n_with_ref = 0
        n_without_ref = 0
        n_phase_matched = 0

        for ep_idx, ep_data in sorted(demo_dataset._episodes.items()):
            T = ep_data["T"]
            props: np.ndarray = ep_data["props"]  # (T, 26)
            actions: np.ndarray = ep_data["actions"]  # (T, 7)
            embeddings: torch.Tensor = ep_data[
                "embeddings"
            ]  # (T, num_tokens, embed_dim)
            ref_actions_ep: Optional[np.ndarray] = ep_data.get("ref_actions")
            phase_embeddings: Optional[dict] = ep_data.get("phase_embeddings")
            phase_ref_actions: Optional[dict] = ep_data.get("phase_ref_actions")

            # Goal pose: fixed port frame if configured, else demo endpoint
            if fixed_port_pos is not None:
                goal_pos = fixed_port_pos
                goal_quat = fixed_port_quat
            else:
                goal_pos = props[-1, 0:3].astype(np.float64)
                goal_quat = props[-1, 3:7].astype(np.float64)

            # Infer phases early — needed to select phase-matched embeddings.
            phases = _infer_phases(props, goal_pos, goal_quat, rcfg)
            has_phase_embs = phase_embeddings is not None and all(
                n in phase_embeddings for n in PHASE_NAMES
            )

            # Batch-encode all frames → z_rl (T, D_rl).
            # When phase embeddings are available, each frame uses the
            # embedding extracted with its phase-matched prompt.
            if has_phase_embs:
                # Build a per-frame embedding tensor by picking the right
                # phase variant for each frame.
                mixed_embs = torch.empty_like(embeddings)
                for t in range(T):
                    pname = PHASE_NAMES[phases[t]]
                    mixed_embs[t] = phase_embeddings[pname][t]
                emb_source = mixed_embs
                n_phase_matched += 1
            else:
                emb_source = embeddings

            z_rls_list = []
            for t0 in range(0, T, encode_batch):
                t1 = min(t0 + encode_batch, T)
                emb_batch = emb_source[t0:t1].to(self.device)  # (B, N, D)
                with torch.no_grad():
                    _, z_rl_batch = self.rl_token_model.encode(emb_batch)  # (B, D_rl)
                z_rls_list.append(z_rl_batch.cpu().numpy())
            z_rls = np.concatenate(z_rls_list, axis=0)  # (T, D_rl)

            # Per-frame rewards (phases already inferred above for embedding selection)
            if rcfg.mode == "structured":
                rewards = _compute_structured_reward(
                    props, goal_pos, goal_quat, phases, rcfg
                )
            else:  # "legacy": original position-only Gaussian
                d2 = np.sum((props[:, 0:3] - goal_pos) ** 2, axis=1)
                rewards = np.exp(-d2 / (rcfg.sigma_pos**2)).astype(np.float32)

            # C-step discounted return per chunk start
            gamma = float(self.config.gamma)
            chunk_returns = np.zeros(T, dtype=np.float32)
            for t in range(T):
                g = 1.0
                R = 0.0
                for k in range(C):
                    tt = min(t + k, T - 1)
                    R += g * float(rewards[tt])
                    g *= gamma
                chunk_returns[t] = R

            if ref_actions_ep is not None:
                n_with_ref += 1
            else:
                n_without_ref += 1

            has_phase_refs = phase_ref_actions is not None and all(
                n in phase_ref_actions for n in PHASE_NAMES
            )

            n_added = 0
            for t in range(T - C + 1):
                action_chunk = actions[t : t + C]
                # Pick reference: phase-matched > default VLA > demo copy
                if has_phase_refs:
                    pname = PHASE_NAMES[phases[t]]
                    ref_chunk = phase_ref_actions[pname][t]
                elif ref_actions_ep is not None:
                    ref_chunk = ref_actions_ep[t]
                else:
                    ref_chunk = action_chunk.copy()
                t_next = min(t + 1, T - 1)
                done = float(t == T - C)

                pending.append(
                    (
                        Transition(
                            z_rl=z_rls[t],
                            prop=props[t],
                            action_chunk=action_chunk,
                            ref_action_chunk=ref_chunk,
                            reward=float(chunk_returns[t]),
                            next_z_rl=z_rls[t_next],
                            next_prop=props[t_next],
                            done=done,
                        ),
                        ep_idx,
                    )
                )
                n_added += 1

            ph_counts = [int(np.sum(phases == p)) for p in range(4)]
            logger.info(
                "  Episode %03d: T=%d, transitions=%d, phases=%s (appr/align/insert/verify), "
                "reward min/mean/max=%.3f/%.3f/%.3f",
                ep_idx,
                T,
                n_added,
                ph_counts,
                float(rewards.min()),
                float(rewards.mean()),
                float(rewards.max()),
            )

        # Pass 2: normalize rewards and commit to replay buffer.
        raw_rewards = np.array([tr.reward for tr, _ in pending], dtype=np.float32)
        if rcfg.normalize and len(raw_rewards) > 1:
            r_mean = float(raw_rewards.mean())
            r_std = float(raw_rewards.std())
            if r_std < 1e-8:
                r_std = 1.0
            logger.info(
                "Reward normalization: mean=%.3f std=%.3f → shifting to zero mean, unit std",
                r_mean,
                r_std,
            )
            normed_rewards = (raw_rewards - r_mean) / r_std
        else:
            normed_rewards = raw_rewards

        for i, (tr, _ep) in enumerate(pending):
            tr.reward = float(normed_rewards[i])
            self.replay_buffer.add(tr)

        logger.info(
            "Replay reward stats over %d transitions (after normalization=%s): "
            "min=%.3f p50=%.3f mean=%.3f p95=%.3f max=%.3f",
            len(normed_rewards),
            rcfg.normalize,
            float(normed_rewards.min()),
            float(np.median(normed_rewards)),
            float(normed_rewards.mean()),
            float(np.percentile(normed_rewards, 95)),
            float(normed_rewards.max()),
        )
        n_total_eps = n_with_ref + n_without_ref
        if n_without_ref > 0:
            logger.warning(
                "Phase 2: %d/%d episodes have NO pre-extracted VLA ref_actions; "
                "falling back to demo actions as the reference.",
                n_without_ref,
                n_total_eps,
            )
        else:
            logger.info(
                "Phase 2: using pre-extracted VLA ref_actions for all %d episodes.",
                n_with_ref,
            )
        if n_phase_matched > 0:
            logger.info(
                "Phase 2: %d/%d episodes used phase-matched embeddings+refs "
                "(approach/align/insert/verify prompts).",
                n_phase_matched,
                n_total_eps,
            )
        else:
            logger.info(
                "Phase 2: no phase-conditioned embeddings found; using single-prompt "
                "embeddings. Re-run prepare_embeddings.py --extract_phase_prompts to fix."
            )

    def pretrain_rl_token(self, demo_dataset: "torch.utils.data.Dataset") -> None:
        """Train RL token encoder-decoder on demonstration data (equation (2)).

        Args:
            demo_dataset: dataset that yields (vla_embeddings,) tensors
                          vla_embeddings: (N, D_vla) per sample
        """
        logger.info("=== Phase 1: RL Token Pretraining ===")
        loader = torch.utils.data.DataLoader(
            demo_dataset,
            batch_size=self.config.rl_token_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        # CSV logging for Phase 1
        log_dir = self._checkpoint_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / "phase1_loss.csv"
        with open(csv_path, "w") as f:
            f.write("epoch,loss\n")

        self.rl_token_model.train()
        for epoch in range(self.config.rl_token_epochs):
            total_loss = 0.0
            for batch in loader:
                vla_embeds = batch["vla_embeddings"].to(self.device)  # (B, N, D_vla)
                z_rl, _ = self.rl_token_model.encode(vla_embeds)
                loss = self.rl_token_model.reconstruction_loss(vla_embeds, z_rl)
                self.rl_token_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.rl_token_model.parameters(), 1.0)
                self.rl_token_optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            logger.info(
                f"  RL Token epoch {epoch + 1}/{self.config.rl_token_epochs}  loss={avg_loss:.4f}"
            )
            with open(csv_path, "a") as f:
                f.write(f"{epoch + 1},{avg_loss:.6f}\n")
            if self._wandb:
                self._wandb.log(
                    {
                        "phase1/epoch": epoch + 1,
                        "phase1/recon_loss": avg_loss,
                    },
                    step=epoch + 1,
                )

        # Freeze RL token for online RL
        for p in self.rl_token_model.parameters():
            p.requires_grad = False
        self.rl_token_model.eval()
        logger.info("RL token frozen for online RL phase.")

    # ------------------------------------------------------------------
    # Phase 2: Online RL
    # ------------------------------------------------------------------

    def _encode_rl_state(self, obs) -> Tuple[np.ndarray, np.ndarray]:
        """Run VLA encoder → RL token, and extract proprioceptive state."""
        with torch.no_grad():
            vla_embeds = self.get_vla_embeddings(obs)  # (1, N, D_vla)
            _, z_rl_sg = self.rl_token_model.encode(vla_embeds)  # (1, D_rl)
        z_rl = z_rl_sg.squeeze(0).cpu().numpy()  # (D_rl,)
        prop = self.get_prop_state(obs)  # (prop_dim,)
        return z_rl, prop

    def _get_rl_state_tensor(self, z_rl: np.ndarray, prop: np.ndarray):
        z = torch.from_numpy(z_rl).unsqueeze(0).to(self.device)
        p = torch.from_numpy(prop).unsqueeze(0).to(self.device)
        return z, p

    def _collect_transition(
        self,
        obs,
        step: int,
        n_warmup: int,
    ) -> Tuple[bool, object]:
        """Collect one chunk-level transition and add to replay buffer.

        Returns:
            done: whether episode ended
            next_obs: the observation after executing the chunk
        """
        C = self.config.actor_critic.chunk_length
        D = self.config.actor_critic.action_dim
        subsample_stride = 2  # store every other step in chunk

        z_rl, prop = self._encode_rl_state(obs)

        # VLA reference action chunk (always computed; may be zeroed by dropout in actor)
        ref_chunk = self.get_vla_action_chunk(obs)  # (C, D)

        # Check for human intervention
        human_chunk = self.human_intervention() if self.human_intervention else None

        if human_chunk is not None:
            # During intervention: store human actions; VLA reference included
            action_chunk = human_chunk
        elif step < n_warmup:
            # Warmup phase: use VLA reference action directly
            action_chunk = ref_chunk
        else:
            # Online RL: sample from actor
            z_t, p_t = self._get_rl_state_tensor(z_rl, prop)
            ref_t = torch.from_numpy(ref_chunk).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_t, _ = self.actor.sample(z_t, p_t, ref_t, training=True)
            action_chunk = action_t.squeeze(0).cpu().numpy()  # (C, D)

        # Execute chunk in environment (environment handles the C steps internally
        # and returns the observation after the last step + aggregate reward)
        next_obs, reward, done, _ = self.env_step(action_chunk)

        next_z_rl, next_prop = self._encode_rl_state(next_obs)

        # Subsample and store transitions (stride=2 → C//stride transitions)
        for i in range(0, C, subsample_stride):
            end = min(i + subsample_stride, C)
            sub_action = action_chunk[i:end]
            sub_ref = ref_chunk[i:end]
            # Pad to length C//2 if needed (or store as-is; adjust buffer chunk_length if using)
            transition = Transition(
                z_rl=z_rl,
                prop=prop,
                action_chunk=action_chunk,  # full chunk for Q-function
                ref_action_chunk=ref_chunk,
                reward=reward,
                next_z_rl=next_z_rl,
                next_prop=next_prop,
                done=done,
            )
            self.replay_buffer.add(transition)
            break  # store one transition per chunk (full chunk as in paper)

        return done, next_obs

    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> float:
        """Critic TD update (equation (3)).

        Target Q̂ = sum_{t'=t}^{t+C-1} γ^{t'-t} r_{t'} + γ^C * E_{a'~π}[Q'(x', a')]
        (We simplify: single reward + γ^C bootstrap since chunk reward is aggregate.)
        """
        cfg = self.config
        C = cfg.actor_critic.chunk_length

        z_rl = batch["z_rl"]
        prop = batch["prop"]
        action = batch["action_chunk"]
        ref_action = batch["ref_action_chunk"]
        reward = batch["reward"]
        next_z_rl = batch["next_z_rl"]
        next_prop = batch["next_prop"]
        done = batch["done"]

        with torch.no_grad():
            # Sample next action from actor (with target policy noise, TD3)
            next_ref_t = torch.zeros_like(
                action
            )  # no reference at next state during target
            next_mu, next_log_std = self.actor.forward(
                next_z_rl, next_prop, next_ref_t, training=False
            )
            noise = (torch.randn_like(next_mu) * cfg.target_policy_noise).clamp(
                -cfg.target_noise_clip, cfg.target_noise_clip
            )
            next_action = next_mu + noise

            # Minimum twin Q target
            q_next = self.critic_target.min_q(next_z_rl, next_prop, next_action)
            q_target = reward + (cfg.gamma**C) * (1.0 - done) * q_next  # (B,)

        # Critic loss (all networks)
        qs = self.critic.forward(z_rl, prop, action)
        critic_loss = sum(F.mse_loss(q, q_target) for q in qs)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        return critic_loss.item()

    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> float:
        """Actor update: maximize Q + BC regularizer (equation (5)).

        L_a(θ) = E[ -Q_ψ(x, a_{1:C}) + β * ||a_{1:C} - ā_{1:C}||² ]
        """
        z_rl = batch["z_rl"]
        prop = batch["prop"]
        ref_action = batch["ref_action_chunk"]

        action, _ = self.actor.sample(z_rl, prop, ref_action, training=True)

        q_val = self.critic.min_q(z_rl, prop, action)

        # BC regularizer toward VLA reference action chunk (equation (5))
        bc_loss = F.mse_loss(action, ref_action)

        actor_loss = -q_val.mean() + self.config.bc_coeff * bc_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        return actor_loss.item()

    def _soft_update_target(self) -> None:
        """EMA update of target critic (TD3 target network)."""
        tau = self.config.tau
        for p_target, p in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            p_target.data.mul_(1.0 - tau).add_(tau * p.data)

    def _gradient_update_step(self) -> Dict[str, float]:
        """One round of gradient updates (G times called per env step)."""
        if len(self.replay_buffer) < self.config.batch_size:
            return {}

        metrics = {}
        batch = self.replay_buffer.sample(self.config.batch_size)

        # Critic updates (2 per actor update)
        for _ in range(self.config.critic_updates_per_actor_update):
            c_loss = self._update_critic(batch)
            self._soft_update_target()
        metrics["critic_loss"] = c_loss

        # Actor update
        a_loss = self._update_actor(batch)
        metrics["actor_loss"] = a_loss

        self._total_gradient_steps += 1
        return metrics

    def train(self, initial_obs_fn: Callable) -> None:
        """Main training loop (Algorithm 1).

        Args:
            initial_obs_fn: callable() → obs, resets the environment and returns
                            the first observation of a new episode.
        """
        cfg = self.config
        logger.info("=== Phase 2: Online RL ===")

        obs = initial_obs_fn()
        env_step_count = 0

        while env_step_count < cfg.total_env_steps:
            # --- Rollout ---
            done, obs = self._collect_transition(
                obs, step=env_step_count, n_warmup=cfg.n_warmup_steps
            )
            env_step_count += 1

            if done:
                obs = initial_obs_fn()

            # --- Gradient updates (update-to-data ratio) ---
            for _ in range(cfg.updates_per_step):
                metrics = self._gradient_update_step()

            # --- Logging ---
            if (
                self._total_gradient_steps % cfg.log_interval == 0
                and self._total_gradient_steps > 0
            ):
                log_str = f"  step={env_step_count}"
                for k, v in metrics.items():
                    log_str += f"  {k}={v:.4f}"
                logger.info(log_str)

            # --- Checkpointing ---
            if (
                self._total_gradient_steps % cfg.save_interval == 0
                and self._total_gradient_steps > 0
            ):
                self.save_checkpoint(f"step_{self._total_gradient_steps}")

        logger.info("Training complete.")
        self.save_checkpoint("final")

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: str = "latest") -> None:
        ckpt_path = self._checkpoint_dir / f"{tag}.pt"
        torch.save(
            {
                "rl_token_model": self.rl_token_model.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "total_gradient_steps": self._total_gradient_steps,
            },
            ckpt_path,
        )
        logger.info(f"Checkpoint saved → {ckpt_path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.rl_token_model.load_state_dict(ckpt["rl_token_model"])
        # Actor/critic may have different action_dim than checkpoint (e.g.
        # 7D quat checkpoint loaded into 9D rot6d model). Load what fits,
        # skip what doesn't — actor/critic will be retrained from scratch.
        for name, model, opt in [
            ("actor", self.actor, self.actor_optimizer),
            ("critic", self.critic, self.critic_optimizer),
            ("critic_target", self.critic_target, None),
        ]:
            if name in ckpt:
                try:
                    model.load_state_dict(ckpt[name])
                    if opt and f"{name}_optimizer" in ckpt:
                        opt.load_state_dict(ckpt[f"{name}_optimizer"])
                except RuntimeError as e:
                    logger.warning(
                        "Skipping %s weights from checkpoint (shape mismatch, "
                        "likely action_dim change): %s",
                        name,
                        e,
                    )
        self._total_gradient_steps = ckpt.get("total_gradient_steps", 0)
        logger.info(f"Checkpoint loaded ← {path}")
