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

At inference time this policy:
  1. Runs the frozen VLA backbone to extract internal embeddings
  2. Encodes them into the RL token z_rl with the frozen encoder
  3. Concatenates z_rl with proprioceptive state → RL state x
  4. Feeds x + VLA reference action chunk into the trained actor to get
     a refined action chunk ā_{1:C}
  5. Executes actions from the chunk at 50 Hz

Load a trained checkpoint produced by aic_rlt/scripts/train.py.

Usage (ROS 2):
    pixi run ros2 run aic_model aic_model \
        --ros-args \
        -p use_sim_time:=true \
        -p policy:=aic_example_policies.ros.RunRLT \
        -p policy_args.checkpoint_path:=/path/to/checkpoints/rlt/phase2_offline.pt \
        -p policy_args.vla_model_dir:=/home/yifeng/models/xvla-base \
        -p policy_args.instruction:="Insert SFP cable into NIC port"
"""

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import cv2
from geometry_msgs.msg import Twist, Vector3
from rclpy.node import Node

# Default XVLA model directory (override via policy_args.vla_model_dir)
DEFAULT_VLA_MODEL_DIR = "/home/yifeng/models/xvla-base"
DEFAULT_INSTRUCTION = "Insert SFP cable into NIC port"

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from geometry_msgs.msg import Wrench

# RLT components
from aic_rlt.models.rl_token import RLTokenModel, RLTokenConfig
from aic_rlt.models.actor_critic import Actor, ActorCriticConfig


# ---------------------------------------------------------------------------
# Default checkpoint path (override via ROS parameter policy_args.checkpoint_path)
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = ""


class RunRLT(Policy):
    """RLT inference policy.

    This class loads:
      - A frozen VLA backbone (π0 or similar) to extract internal embeddings
        and produce the VLA reference action chunk.
      - The trained RL token encoder (frozen at inference).
      - The trained RL actor (lightweight MLP).

    All three are frozen at inference time; only the actor's output is used
    for robot control.
    """

    # Control loop frequency (Hz) – paper uses 50 Hz
    CONTROL_HZ: float = 50.0
    # Action chunk length C (must match training)
    CHUNK_LENGTH: int = 10
    # Action dimension (6D velocity delta + gripper)
    ACTION_DIM: int = 7
    # Proprioceptive state dimension (TCP pose 7 + vel 6 + error 6 + joints 7)
    PROP_DIM: int = 26
    # Image scale factor (must match training data)
    IMAGE_SCALE: float = 0.25

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Read parameters from ROS (set via policy_args)
        checkpoint_path = DEFAULT_CHECKPOINT
        vla_model_dir = DEFAULT_VLA_MODEL_DIR
        instruction = DEFAULT_INSTRUCTION
        try:
            checkpoint_path = parent_node.get_parameter("policy_args.checkpoint_path").value
        except Exception:
            pass
        try:
            vla_model_dir = parent_node.get_parameter("policy_args.vla_model_dir").value
        except Exception:
            pass
        try:
            instruction = parent_node.get_parameter("policy_args.instruction").value
        except Exception:
            pass

        if not checkpoint_path:
            self.get_logger().warn(
                "No checkpoint path provided via policy_args.checkpoint_path. "
                "Models will be randomly initialized – only useful for testing."
            )

        # ---- Build model components ----
        rl_token_cfg = RLTokenConfig()
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

        # Freeze all parameters at inference
        for model in (self.rl_token_model, self.actor):
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

        # ---- VLA backbone ----
        self._vla = self._load_vla(vla_model_dir, instruction)

        self.get_logger().info(
            f"RunRLT initialized on {self.device}. "
            f"Checkpoint: {checkpoint_path or '(none)'}"
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.rl_token_model.load_state_dict(ckpt["rl_token_model"])
        self.actor.load_state_dict(ckpt["actor"])
        self.get_logger().info(f"RLT checkpoint loaded from {path}")

    def _load_vla(self, model_dir: str, instruction: str):
        """Load XVLA (lerobot/xvla-base) as the frozen VLA backbone.

        Wraps XVLAWrapper in an adapter that accepts ROS Observation messages
        and extracts the center camera image + proprioceptive state internally.

        Falls back to a zero-output stub if the model directory does not exist,
        so the pipeline can be tested for ROS connectivity before XVLA is available.

        The returned object has:
            vla.get_embeddings(obs) -> torch.Tensor (1, N, D_vla)
            vla.get_action_chunk(obs) -> np.ndarray (C, action_dim)
        """
        if not Path(model_dir).exists():
            self.get_logger().warn(
                f"XVLA model directory not found: {model_dir}. "
                "Using zero-output stub. "
                "Download with: python -c \"from huggingface_hub import snapshot_download; "
                f"snapshot_download('lerobot/xvla-base', local_dir='{model_dir}')\""
            )

            class _StubVLA:
                def __init__(self, device, cfg: RLTokenConfig, C: int, D: int):
                    self.device = device
                    self.N = cfg.num_vla_tokens
                    self.D_vla = cfg.vla_embed_dim
                    self.C = C
                    self.D = D

                def get_embeddings(self, obs) -> torch.Tensor:
                    return torch.zeros(1, self.N, self.D_vla, device=self.device)

                def get_action_chunk(self, obs) -> np.ndarray:
                    return np.zeros((self.C, self.D), dtype=np.float32)

            return _StubVLA(self.device, RLTokenConfig(), self.CHUNK_LENGTH, self.ACTION_DIM)

        self.get_logger().info(f"Loading XVLAWrapper from {model_dir} ...")
        from aic_rlt.vla.xvla_wrapper import XVLAWrapper

        xvla = XVLAWrapper(
            model_dir=model_dir,
            device=self.device,
            instruction=instruction,
            image_size=256,
            chunk_length=self.CHUNK_LENGTH,
        )
        self.get_logger().info("XVLAWrapper loaded.")

        # Adapter: accepts ROS Observation, extracts center image + prop state
        extract_prop = self._extract_prop_state

        class _XVLAAdapter:
            def __init__(self, wrapper: XVLAWrapper, prop_fn):
                self._xvla = wrapper
                self._prop_fn = prop_fn

            def get_embeddings(self, obs) -> torch.Tensor:
                img_np = np.frombuffer(obs.center_image.data, dtype=np.uint8).reshape(
                    obs.center_image.height, obs.center_image.width, 3
                )
                # (num_tokens, 1024) → unsqueeze batch dim → (1, num_tokens, 1024)
                emb = self._xvla.get_embeddings(img_np)
                return emb.unsqueeze(0)

            def get_action_chunk(self, obs) -> np.ndarray:
                img_np = np.frombuffer(obs.center_image.data, dtype=np.uint8).reshape(
                    obs.center_image.height, obs.center_image.width, 3
                )
                prop = self._prop_fn(obs)
                return self._xvla.get_action_chunk(img_np, prop)

        return _XVLAAdapter(xvla, extract_prop)

    # ------------------------------------------------------------------
    # Observation processing
    # ------------------------------------------------------------------

    @staticmethod
    def _img_to_tensor(raw_img, device: torch.device, scale: float) -> torch.Tensor:
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )
        if scale != 1.0:
            img_np = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )

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
        """Returns z_rl (1, D_rl), prop (1, prop_dim), ref_chunk (1, C, D) tensors."""
        # VLA embeddings → RL token
        vla_embeds = self._vla.get_embeddings(obs)           # (1, N, D_vla)
        with torch.no_grad():
            _, z_rl = self.rl_token_model.encode(vla_embeds)  # (1, D_rl)

        # VLA reference action chunk
        ref_np = self._vla.get_action_chunk(obs)              # (C, D)
        ref_t = torch.from_numpy(ref_np).unsqueeze(0).to(self.device)  # (1, C, D)

        # Proprioception
        prop_np = self._extract_prop_state(obs)
        prop_t = torch.from_numpy(prop_np).unsqueeze(0).to(self.device)  # (1, prop_dim)

        return z_rl, prop_t, ref_t

    # ------------------------------------------------------------------
    # Action execution helpers
    # ------------------------------------------------------------------

    def _action_to_motion_update(self, action: np.ndarray) -> MotionUpdate:
        """Convert a 7-dim delta velocity action to a MotionUpdate message."""
        twist = Twist(
            linear=Vector3(x=float(action[0]), y=float(action[1]), z=float(action[2])),
            angular=Vector3(x=float(action[3]), y=float(action[4]), z=float(action[5])),
        )
        motion_update = MotionUpdate()
        motion_update.velocity = twist
        motion_update.header.frame_id = "base_link"
        motion_update.header.stamp = self.get_clock().now().to_msg()
        motion_update.target_stiffness = np.diag(
            [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
        ).flatten()
        motion_update.target_damping = np.diag(
            [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
        ).flatten()
        motion_update.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        motion_update.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        motion_update.trajectory_generation_mode.mode = (
            TrajectoryGenerationMode.MODE_VELOCITY
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

        dt = 1.0 / self.CONTROL_HZ
        timeout_sec = 60.0
        start_time = time.time()

        # Action chunk buffer: we query the actor once per chunk, then
        # execute individual actions within the chunk at 50 Hz.
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
                z_rl, prop, ref_chunk = self._encode_rl_state(obs)
                with torch.no_grad():
                    action_chunk_t = self.actor.get_mean(z_rl, prop, ref_chunk)  # (1, C, D)
                action_chunk = action_chunk_t.squeeze(0).cpu().numpy()  # (C, D)
                chunk_step = 0

            # Execute current step within chunk
            action = action_chunk[chunk_step]  # (D,)
            chunk_step += 1

            motion_update = self._action_to_motion_update(action)
            move_robot(motion_update=motion_update)
            send_feedback(f"RLT step {chunk_step}/{self.CHUNK_LENGTH}")

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, dt - elapsed))

        self.get_logger().info("RunRLT.insert_cable() finished.")
        return True
