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

"""XVLA (lerobot/xvla-base) backend for RLT.

Wraps XVLAWrapper so RunRLT can use it via the VLABackend interface.
Handles ROS Observation → image/prop extraction internally.
"""

import logging

import numpy as np
import torch

from .base import VLABackend
from .xvla_wrapper import XVLAWrapper

logger = logging.getLogger(__name__)

# Index range of joint positions within 26-dim proprioceptive state:
# state = [tcp_pos(3), tcp_quat(4), tcp_vel_l(3), tcp_vel_a(3),
#           tcp_err(3), tcp_err_r(3), joint_0..6(7)]
_JOINT_START = 19
_JOINT_END = 26


class XVLABackend(VLABackend):
    """XVLA (Florence-2 based) VLA backend.

    Uses the XVLA center-camera image for both embedding extraction and
    action generation.  Prop state is extracted from the ROS Observation
    for the action chunk call (XVLA uses joints for conditioning).

    Parameters
    ----------
    model_dir:    path to the downloaded xvla-base checkpoint directory
    device:       torch device
    instruction:  language prompt for the task
    image_size:   square size to resize images to (default 256)
    chunk_length: number of actions to return per chunk
    """

    def __init__(
        self,
        model_dir: str,
        device: torch.device,
        instruction: str = "Insert SFP cable into NIC port",
        image_size: int = 256,
        chunk_length: int = 10,
    ):
        self._wrapper = XVLAWrapper(
            model_dir=model_dir,
            device=device,
            instruction=instruction,
            image_size=image_size,
            chunk_length=chunk_length,
        )
        self.device = device
        self.chunk_length = chunk_length
        self.action_dim: int = 9  # 3 xyz + 6 rot6d (see xvla_wrapper.ee6d_to_xyz_rot6d)

        # Probe dimensions from a dummy call
        dummy = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        emb = self._wrapper.get_embeddings(dummy)  # (num_tokens, embed_dim) on CPU
        self.num_tokens: int = emb.shape[0]
        self.embed_dim: int = emb.shape[1]
        logger.info(
            "XVLABackend ready: num_tokens=%d, embed_dim=%d, action_dim=%d",
            self.num_tokens,
            self.embed_dim,
            self.action_dim,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_center_image(obs) -> np.ndarray:
        """Extract center camera image as (H, W, 3) uint8 numpy array."""
        raw = obs.center_image
        return np.frombuffer(raw.data, dtype=np.uint8).reshape(raw.height, raw.width, 3)

    @staticmethod
    def _extract_prop_state(obs) -> np.ndarray:
        """Extract 26D proprioceptive state from AIC observation."""
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

    # ------------------------------------------------------------------
    # VLABackend interface
    # ------------------------------------------------------------------

    def set_instruction(self, instruction: str) -> None:
        """Swap the active prompt (forwards to the underlying XVLAWrapper)."""
        self._wrapper.set_instruction(instruction)

    def get_embeddings(self, obs) -> torch.Tensor:
        """(1, num_tokens, embed_dim) on device."""
        img = self._extract_center_image(obs)
        emb = self._wrapper.get_embeddings(img)  # (N, D) on CPU
        return emb.unsqueeze(0).to(self.device)  # (1, N, D)

    def get_action_chunk(self, obs) -> np.ndarray:
        """(chunk_length, action_dim) float32."""
        img = self._extract_center_image(obs)
        prop = self._extract_prop_state(obs)
        return self._wrapper.get_action_chunk(img, prop)  # (C, D)

    # get_embeddings_and_actions uses default (two separate calls)
    # because XVLAWrapper has no single-pass API. If you add one,
    # override here to avoid the double forward pass.
