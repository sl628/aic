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

"""RLT inference policy for the AIC cable insertion challenge.

Implements "RL Token: Bootstrapping Online RL with Vision-Language-Action Models"
(Xu et al., Physical Intelligence, 2025).

Supports two VLA backends via policy_args.vla_backend:
  "xvla"  — lerobot/xvla-base (Florence-2, PyTorch, default)
  "pi05"  — openpi pi0.5 (PaliGemma, JAX)

At inference time this policy:
  1. Runs the frozen VLA backbone to extract internal embeddings
  2. Encodes them into the RL token z_rl with the frozen encoder
  3. Concatenates z_rl with proprioceptive state → RL state x
  4. Feeds x + VLA reference action chunk into the trained actor to get
     a refined action chunk ā_{1:C}
  5. Executes actions from the chunk at the backend's control rate

Load a trained checkpoint produced by aic_rlt/scripts/train.py.

Usage (ROS 2) — XVLA backend (default):
    pixi run ros2 run aic_model aic_model \
        --ros-args \
        -p use_sim_time:=true \
        -p policy:=aic_example_policies.ros.RunRLT \
        -p policy_args.checkpoint_path:=/path/to/checkpoints/rlt/phase2_offline.pt \
        -p policy_args.vla_model_dir:=/home/yifeng/models/xvla-base \
        "-p policy_args.instruction:=Insert SFP cable into NIC port"

Usage (ROS 2) — Pi0.5 backend:
    pixi run ros2 run aic_model aic_model \
        --ros-args \
        -p use_sim_time:=true \
        -p policy:=aic_example_policies.ros.RunRLT \
        -p policy_args.vla_backend:=pi05 \
        -p policy_args.checkpoint_path:=/path/to/checkpoints/rlt/phase2_offline.pt \
        -p policy_args.pi05_checkpoint:=/home/yifeng/workspace/pi05_base/pi05_base \
        "-p policy_args.instruction:=insert the cable into the port"
"""

import time
from typing import Optional

import numpy as np
import torch
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, Wrench
from rclpy.node import Node

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode

# RLT components
from aic_rlt.models.rl_token import RLTokenModel, RLTokenConfig
from aic_rlt.models.actor_critic import Actor, ActorCriticConfig
from aic_rlt.vla import create_vla_backend
from aic_rlt.trainer import DEFAULT_PHASE_PROMPTS, PHASE_NAMES

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_VLA_BACKEND = "xvla"
DEFAULT_VLA_MODEL_DIR = "/home/yifeng/models/xvla-base"  # XVLA
DEFAULT_PI05_CKPT = "/home/yifeng/workspace/pi05_base/pi05_base"  # Pi0.5
DEFAULT_INSTRUCTION = "Insert SFP cable into NIC port"
DEFAULT_CHECKPOINT = ""

# Phase-conditioned prompt defaults (from trainer.py shared constants).
# Override via ROS params policy_args.prompt_{approach,align,insert,verify}.
_PHASE_PROMPT_DEFAULTS = {
    name: DEFAULT_PHASE_PROMPTS[i] for i, name in enumerate(PHASE_NAMES)
}

# Phase-estimator thresholds (must match RewardConfig defaults in trainer).
_PHASE_D_ALIGN = 0.03  # m, approach → align
_PHASE_ORI_ALIGN = 0.05  # 1 - |q·qg|, align → insert
_PHASE_ERR_CONTACT = 0.001  # tcp_err norm (m), align → insert
_PHASE_D_VERIFY = 0.005  # m, insert → verify


class RunRLT(Policy):
    """RLT inference policy supporting XVLA and Pi0.5 VLA backends.

    Architecture (backend-agnostic):
        VLA backbone (frozen) → embeddings (1, N, D_vla)
        RL Token Encoder (frozen) → z_rl (D_rl)
        Actor MLP (frozen) → action chunk (C, action_dim)
        → MotionUpdate (Cartesian velocity) → aic_controller
    """

    # Action chunk length C (must match training)
    CHUNK_LENGTH: int = 10
    # Action dimension: target TCP pose xyz(3) + 6D rotation r1,r2(6) = 9
    ACTION_DIM: int = 9
    # Proprioceptive state dimension
    PROP_DIM: int = 26

    # Control rates per backend (Hz)
    _CONTROL_HZ = {"xvla": 50.0, "pi05": 20.0}

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Declare then read ROS parameters ----
        _params = [
            ("policy_args.vla_backend", DEFAULT_VLA_BACKEND),
            ("policy_args.checkpoint_path", DEFAULT_CHECKPOINT),
            ("policy_args.vla_model_dir", DEFAULT_VLA_MODEL_DIR),
            ("policy_args.pi05_checkpoint", DEFAULT_PI05_CKPT),
            ("policy_args.instruction", DEFAULT_INSTRUCTION),
            # Port pose for phase estimator (xyz + quat xyzw). Empty list
            # disables phase switching; the VLA keeps the static instruction.
            ("policy_args.port_pose_xyzquat", []),
            # Per-phase prompts (can be overridden from launch file)
            ("policy_args.prompt_approach", _PHASE_PROMPT_DEFAULTS["approach"]),
            ("policy_args.prompt_align", _PHASE_PROMPT_DEFAULTS["align"]),
            ("policy_args.prompt_insert", _PHASE_PROMPT_DEFAULTS["insert"]),
            ("policy_args.prompt_verify", _PHASE_PROMPT_DEFAULTS["verify"]),
            # Debug: bypass actor and execute VLA reference actions directly.
            ("policy_args.vla_only", False),
        ]
        for name, default in _params:
            try:
                parent_node.declare_parameter(name, default)
            except Exception:
                pass  # already declared by a previous init

        vla_backend = parent_node.get_parameter("policy_args.vla_backend").value
        checkpoint_path = parent_node.get_parameter("policy_args.checkpoint_path").value
        vla_model_dir = parent_node.get_parameter("policy_args.vla_model_dir").value
        pi05_checkpoint = parent_node.get_parameter("policy_args.pi05_checkpoint").value
        instruction = parent_node.get_parameter("policy_args.instruction").value

        # Phase prompts + port pose (optional)
        self._phase_prompts = {
            "approach": parent_node.get_parameter("policy_args.prompt_approach").value,
            "align": parent_node.get_parameter("policy_args.prompt_align").value,
            "insert": parent_node.get_parameter("policy_args.prompt_insert").value,
            "verify": parent_node.get_parameter("policy_args.prompt_verify").value,
        }
        self._vla_only = parent_node.get_parameter("policy_args.vla_only").value
        if self._vla_only:
            self.get_logger().info(
                "VLA-only mode: bypassing actor, executing VLA refs directly"
            )

        port_pose = list(
            parent_node.get_parameter("policy_args.port_pose_xyzquat").value
        )
        if len(port_pose) == 7:
            self._port_pos = np.asarray(port_pose[0:3], dtype=np.float64)
            self._port_quat = np.asarray(port_pose[3:7], dtype=np.float64)
        else:
            self._port_pos = None
            self._port_quat = None
        self._current_phase: Optional[str] = None

        self.control_hz = self._CONTROL_HZ.get(vla_backend, 50.0)

        if not checkpoint_path:
            self.get_logger().warn(
                "No checkpoint path provided via policy_args.checkpoint_path. "
                "Models will be randomly initialized – only useful for testing."
            )

        # ---- Load VLA backend ----
        self._vla = self._load_vla_backend(
            vla_backend, vla_model_dir, pi05_checkpoint, instruction
        )

        # ---- Build RLT models using VLA dimensions ----
        rl_token_cfg = RLTokenConfig(
            vla_embed_dim=self._vla.embed_dim,
            num_vla_tokens=self._vla.num_tokens,
        )

        # Validate against checkpoint if provided
        if checkpoint_path:
            self._validate_checkpoint_dims(checkpoint_path, rl_token_cfg)

        actor_critic_cfg = ActorCriticConfig(
            rl_token_dim=rl_token_cfg.rl_token_dim,
            prop_dim=self.PROP_DIM,
            action_dim=self.ACTION_DIM,
            chunk_length=self.CHUNK_LENGTH,
        )

        self.rl_token_model = RLTokenModel(rl_token_cfg).to(self.device)
        self.actor = Actor(actor_critic_cfg).to(self.device)

        # ---- Load checkpoint ----
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        # Freeze all RLT parameters at inference
        for model in (self.rl_token_model, self.actor):
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

        self.get_logger().info(
            f"RunRLT initialized on {self.device}. "
            f"Backend: {vla_backend} "
            f"(num_tokens={self._vla.num_tokens}, embed_dim={self._vla.embed_dim}). "
            f"Checkpoint: {checkpoint_path or '(none)'}"
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _load_vla_backend(
        self,
        backend: str,
        vla_model_dir: str,
        pi05_checkpoint: str,
        instruction: str,
    ):
        """Instantiate the appropriate VLA backend."""
        self.get_logger().info(f"Loading VLA backend: {backend}")

        if backend == "xvla":
            from pathlib import Path

            if not Path(vla_model_dir).exists():
                self.get_logger().warn(
                    f"XVLA model directory not found: {vla_model_dir}. "
                    "Using zero-output stub. "
                    'Download with: python -c "from huggingface_hub import snapshot_download; '
                    f"snapshot_download('lerobot/xvla-base', local_dir='{vla_model_dir}')\""
                )
                return self._make_stub_backend()

            return create_vla_backend(
                "xvla",
                device=self.device,
                model_dir=vla_model_dir,
                instruction=instruction,
                image_size=256,
                chunk_length=self.CHUNK_LENGTH,
            )

        elif backend == "pi05":
            return create_vla_backend(
                "pi05",
                device=self.device,
                checkpoint_dir=pi05_checkpoint,
                chunk_length=self.CHUNK_LENGTH,
                instruction=instruction,
            )

        else:
            raise ValueError(
                f"Unknown vla_backend '{backend}'. "
                "Set policy_args.vla_backend to 'xvla' or 'pi05'."
            )

    def _make_stub_backend(self):
        """Zero-output stub backend for connectivity testing without a VLA."""
        # Use XVLA default dims so any checkpoint trained on XVLA will load
        from aic_rlt.vla.base import VLABackend

        device = self.device
        chunk_length = self.CHUNK_LENGTH
        action_dim = self.ACTION_DIM

        class _StubBackend(VLABackend):
            embed_dim = 1024
            num_tokens = 115

            def get_embeddings(self, obs):
                return torch.zeros(1, self.num_tokens, self.embed_dim, device=device)

            def get_action_chunk(self, obs):
                return np.zeros((chunk_length, action_dim), dtype=np.float32)

        self.get_logger().warn("Using zero-output stub VLA — actions will be zeros.")
        return _StubBackend()

    def _validate_checkpoint_dims(
        self, checkpoint_path: str, cfg: RLTokenConfig
    ) -> None:
        """Warn if checkpoint VLA dims don't match the loaded backend."""
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            w = ckpt["rl_token_model"]["input_proj.weight"]  # (enc_dim, vla_embed_dim)
            ckpt_vla_dim = w.shape[1]
            if ckpt_vla_dim != cfg.vla_embed_dim:
                self.get_logger().error(
                    f"Checkpoint vla_embed_dim={ckpt_vla_dim} does not match "
                    f"VLA backend embed_dim={cfg.vla_embed_dim}. "
                    "The checkpoint was likely trained with a different VLA backend. "
                    "Loading will proceed but inference results will be wrong."
                )
        except Exception as e:
            self.get_logger().warn(f"Could not validate checkpoint dims: {e}")

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.rl_token_model.load_state_dict(ckpt["rl_token_model"])
        self.actor.load_state_dict(ckpt["actor"])
        self.get_logger().info(f"RLT checkpoint loaded from {path}")

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def _extract_prop_state(self, obs: Observation) -> np.ndarray:
        tcp_pose = obs.controller_state.tcp_pose
        tcp_vel = obs.controller_state.tcp_velocity
        return np.array(
            [
                tcp_pose.position.x,
                tcp_pose.position.y,
                tcp_pose.position.z,
                tcp_pose.orientation.x,
                tcp_pose.orientation.y,
                tcp_pose.orientation.z,
                tcp_pose.orientation.w,
                tcp_vel.linear.x,
                tcp_vel.linear.y,
                tcp_vel.linear.z,
                tcp_vel.angular.x,
                tcp_vel.angular.y,
                tcp_vel.angular.z,
                *obs.controller_state.tcp_error,
                *obs.joint_states.position[:7],
            ],
            dtype=np.float32,
        )

    def _encode_rl_state(self, obs: Observation):
        """Returns z_rl (1, D_rl), prop (1, prop_dim), ref_chunk (1, C, D) or None."""
        if self._vla.actions_are_bc_targets:
            # Backend's actions match the BC target distribution (e.g. XVLA):
            # fetch both embeddings and ref actions in one forward pass.
            vla_embeds, ref_np = self._vla.get_embeddings_and_actions(obs)
        else:
            # Backend's actions are NOT BC targets (e.g. Pi0.5 Option B):
            # skip the action-generation path entirely — it's both wasted
            # compute and the outputs would mismatch the actor's expectations.
            vla_embeds = self._vla.get_embeddings(obs)
            ref_np = None

        with torch.no_grad():
            _, z_rl = self.rl_token_model.encode(
                vla_embeds.to(self.device)
            )  # (1, D_rl)

        if ref_np is not None:
            ref_t = torch.from_numpy(ref_np).unsqueeze(0).to(self.device)  # (1, C, D)
        else:
            ref_t = None

        prop_t = (
            torch.from_numpy(self._extract_prop_state(obs)).unsqueeze(0).to(self.device)
        )  # (1, prop_dim)

        return z_rl, prop_t, ref_t

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Phase estimation (port-relative) + prompt switching
    # ------------------------------------------------------------------

    def _estimate_phase(self, prop: np.ndarray) -> str:
        """Pick a phase name from TCP pose + port pose. Falls back to 'approach'
        when no port pose is configured."""
        if self._port_pos is None or self._port_quat is None:
            return "approach"
        pos = prop[0:3]
        quat = prop[3:7]
        err_norm = float(np.linalg.norm(prop[13:16]))
        d = float(np.linalg.norm(pos - self._port_pos))
        ori_sim = abs(float(np.dot(quat, self._port_quat)))
        ori_err = 1.0 - ori_sim
        if d <= _PHASE_D_VERIFY:
            return "verify"
        if (
            d <= _PHASE_D_ALIGN
            and ori_err <= _PHASE_ORI_ALIGN
            and err_norm > _PHASE_ERR_CONTACT
        ):
            return "insert"
        if d <= _PHASE_D_ALIGN:
            return "align"
        return "approach"

    def _maybe_switch_prompt(self, phase: str) -> None:
        """Call backend.set_instruction when the phase changes."""
        if self._port_pos is None:
            return
        if phase == self._current_phase:
            return
        prompt = self._phase_prompts.get(phase)
        if not prompt:
            return
        try:
            self._vla.set_instruction(prompt)
            self.get_logger().info(f"Phase → {phase}: prompt='{prompt}'")
            self._current_phase = phase
        except Exception as e:
            self.get_logger().warn(f"set_instruction failed on phase={phase}: {e}")

    # Max position delta per control step (m). Prevents large jumps that
    # generate torques exceeding Gazebo's effort limits.
    _MAX_POS_DELTA: float = 0.005  # 5 mm per step @ 20 Hz = 10 cm/s max
    _MAX_ROT_DELTA: float = 0.05  # ~3° per step @ 20 Hz = 60°/s max

    def _clamp_action(self, action: np.ndarray, obs: "Observation") -> np.ndarray:
        """Clamp action so the commanded pose is close to the current TCP pose.

        CheatCode works because it interpolates smoothly over 100 steps.
        Without clamping, the actor can command a pose far from the current
        one → huge joint torques → effort clamped to 0 → arm collapses.
        """
        current_pos = np.array(
            [
                obs.controller_state.tcp_pose.position.x,
                obs.controller_state.tcp_pose.position.y,
                obs.controller_state.tcp_pose.position.z,
            ],
            dtype=np.float32,
        )

        # Clamp position delta
        pos_delta = action[0:3] - current_pos
        pos_norm = np.linalg.norm(pos_delta)
        if pos_norm > self._MAX_POS_DELTA:
            action = action.copy()
            action[0:3] = current_pos + pos_delta * (self._MAX_POS_DELTA / pos_norm)

        # Clamp rotation delta (in 6D space — limit the norm of the change)
        from aic_rlt.vla.xvla_wrapper import quat_to_rot6d

        q = obs.controller_state.tcp_pose.orientation
        current_rot6d = quat_to_rot6d(np.array([q.x, q.y, q.z, q.w], dtype=np.float32))
        rot_delta = action[3:9] - current_rot6d
        rot_norm = np.linalg.norm(rot_delta)
        if rot_norm > self._MAX_ROT_DELTA:
            action = action.copy() if not action.flags.writeable else action
            action[3:9] = current_rot6d + rot_delta * (self._MAX_ROT_DELTA / rot_norm)

        return action

    def _action_to_motion_update(self, action: np.ndarray) -> MotionUpdate:
        """Convert a 9-dim action [xyz, r1, r2] to a MotionUpdate.

        Action semantics (from actor output):
            action[0:3] — target TCP position (x, y, z) in base_link frame
            action[3:9] — 6D rotation: r1(3) + r2(3), first two columns of R
        """
        from aic_rlt.vla.xvla_wrapper import rot6d_to_quat

        quat = rot6d_to_quat(action[3:9])  # [qx, qy, qz, qw]

        motion_update = MotionUpdate()
        motion_update.header.frame_id = "base_link"
        motion_update.header.stamp = self.get_clock().now().to_msg()
        motion_update.pose = Pose(
            position=Point(x=float(action[0]), y=float(action[1]), z=float(action[2])),
            orientation=Quaternion(
                x=float(quat[0]),
                y=float(quat[1]),
                z=float(quat[2]),
                w=float(quat[3]),
            ),
        )
        # Match CheatCode defaults (policy.py:set_pose_target)
        motion_update.target_stiffness = (
            np.diag([90.0, 90.0, 90.0, 50.0, 50.0, 50.0]).flatten().tolist()
        )
        motion_update.target_damping = (
            np.diag([50.0, 50.0, 50.0, 20.0, 20.0, 20.0]).flatten().tolist()
        )
        motion_update.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        motion_update.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        motion_update.trajectory_generation_mode.mode = (
            TrajectoryGenerationMode.MODE_POSITION
        )
        return motion_update

    # ------------------------------------------------------------------
    # Main policy entry point
    # ------------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ) -> bool:
        self.get_logger().info(f"RunRLT.insert_cable() start. Task: {task}")

        # Clear JAX compile caches at each trial boundary. Observed during
        # multi-trial aic_engine eval: GPU memory grew across trials until
        # trial 2 or 3 hit CUDA_ERROR_OUT_OF_MEMORY during pi0.5 SigLIP
        # forward. Same pattern we already mitigate in prepare_embeddings
        # (commit 5851cf4). Only runs on backends that use JAX; plain PyTorch
        # backends (e.g. XVLA) shouldn't need this and it's a harmless no-op.
        if not getattr(self._vla, "actions_are_bc_targets", True):
            try:
                import jax  # lazy — don't force JAX import in non-JAX backends

                jax.clear_caches()
            except Exception as e:
                self.get_logger().warn(f"jax.clear_caches() failed: {e}")

        dt = 1.0 / self.control_hz
        timeout_sec = 60.0
        start_time = time.time()

        action_chunk: Optional[np.ndarray] = None
        chunk_step = self.CHUNK_LENGTH  # force re-query on first iteration

        while time.time() - start_time < timeout_sec:
            loop_start = time.time()

            obs = get_observation()
            if obs is None:
                self.get_logger().warn("No observation received, skipping step.")
                time.sleep(dt)
                continue

            # Re-query actor at the start of each new chunk
            if chunk_step >= self.CHUNK_LENGTH:
                # Update phase-conditioned prompt from current proprioception
                # *before* we encode — so the VLA sees the right prompt.
                prop_np = self._extract_prop_state(obs)
                self._maybe_switch_prompt(self._estimate_phase(prop_np))

                z_rl, prop, ref_chunk = self._encode_rl_state(obs)
                if self._vla_only:
                    # Execute VLA reference actions directly (debug mode)
                    action_chunk = ref_chunk.squeeze(0).cpu().numpy()  # (C, D)
                else:
                    with torch.no_grad():
                        action_chunk_t = self.actor.get_mean(
                            z_rl, prop, ref_chunk
                        )  # (1, C, D)
                    action_chunk = action_chunk_t.squeeze(0).cpu().numpy()  # (C, D)
                chunk_step = 0

            action = action_chunk[chunk_step]
            chunk_step += 1

            action = self._clamp_action(action, obs)
            move_robot(motion_update=self._action_to_motion_update(action))
            send_feedback(f"RLT step {chunk_step}/{self.CHUNK_LENGTH}")

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, dt - elapsed))

        self.get_logger().info("RunRLT.insert_cable() finished.")
        return True
