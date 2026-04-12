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

"""XVLA wrapper for RLT: embedding extraction and reference action generation.

Wraps lerobot/xvla-base to provide:
  - get_embeddings(): Florence-2 encoder features for RL token pretraining
  - get_action_chunk(): TCP action chunks (7-dim) for RLT actor conditioning

Action space conversion (lossless):
  Our 7-dim: (x, y, z, qx, qy, qz, qw)  — TCP position + quaternion
  XVLA ee6d: (x, y, z, r1x,r1y,r1z, r2x,r2y,r2z, gripper, [zeros×10])
  Conversion via rotation matrix first-two-columns ↔ Gram-Schmidt ↔ quaternion.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from transformers import AutoTokenizer

from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_STATE

logger = logging.getLogger(__name__)

# ImageNet normalization applied by XVLA preprocessor
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# BART tokenizer used by Florence-2 inside XVLA
_TOKENIZER_NAME = "facebook/bart-large"
_TOKEN_MAX_LEN = 50

# Index range of joint positions within 26-dim proprioceptive state
# observation.state = [tcp_pos(3), tcp_quat(4), tcp_vel_l(3), tcp_vel_a(3),
#                       tcp_err(3), tcp_err_r(3), joint_0..6(7)]
_JOINT_START = 19
_JOINT_END = 26  # joint_0 through joint_6 → 7 values


# ---------------------------------------------------------------------------
# Rotation conversion utilities
# ---------------------------------------------------------------------------

def quat_to_ee6d(xyz: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Convert position + quaternion to XVLA ee6d format (20-dim, zero-padded).

    Args:
        xyz:  (3,) float32 — TCP position
        quat: (4,) float32 — quaternion [qx, qy, qz, qw]

    Returns:
        (20,) float32 — XVLA ee6d: [x,y,z, r1(3), r2(3), gripper=0, zeros×10]
    """
    import transforms3d.quaternions as tq
    # transforms3d uses [qw, qx, qy, qz] convention
    qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
    R = tq.quat2mat([qw, qx, qy, qz])         # (3, 3)
    r1 = R[:, 0].astype(np.float32)            # first column
    r2 = R[:, 1].astype(np.float32)            # second column
    ee6d = np.zeros(20, dtype=np.float32)
    ee6d[0:3] = xyz.astype(np.float32)
    ee6d[3:6] = r1
    ee6d[6:9] = r2
    ee6d[9] = 0.0                              # gripper (not in our action space)
    # indices 10-19 remain zero (second arm / padding)
    return ee6d


def ee6d_to_quat_xyz(ee6d: np.ndarray) -> tuple:
    """Convert XVLA ee6d (20-dim) to position + quaternion.

    Args:
        ee6d: (20,) float32 — XVLA action output

    Returns:
        xyz:  (3,) float32
        quat: (4,) float32 — quaternion [qx, qy, qz, qw]
    """
    import transforms3d.quaternions as tq
    xyz = ee6d[0:3].astype(np.float32)
    r1 = ee6d[3:6].astype(np.float64)
    r2 = ee6d[6:9].astype(np.float64)
    # Gram-Schmidt orthonormalization
    r1 = r1 / (np.linalg.norm(r1) + 1e-8)
    r2 = r2 - np.dot(r2, r1) * r1
    r2 = r2 / (np.linalg.norm(r2) + 1e-8)
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)         # (3, 3), columns are basis vectors
    # transforms3d mat2quat returns [qw, qx, qy, qz]
    q_wxyz = tq.mat2quat(R)
    quat = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)  # [qx,qy,qz,qw]
    return xyz, quat


# ---------------------------------------------------------------------------
# XVLA Wrapper
# ---------------------------------------------------------------------------

class XVLAWrapper:
    """Wraps XVLAPolicy for RLT embedding extraction and reference action generation.

    Usage:
        vla = XVLAWrapper("/home/yifeng/models/xvla-base", device, "Insert SFP cable")
        embeddings = vla.get_embeddings(image_np)         # (num_tokens, 1024)
        ref_chunk  = vla.get_action_chunk(image_np, prop) # (C, 7)
    """

    def __init__(
        self,
        model_dir: str,
        device: torch.device,
        instruction: str,
        image_size: int = 256,
        chunk_length: int = 10,
    ):
        self.device = device
        self.instruction = instruction
        self.image_size = image_size
        self.chunk_length = chunk_length

        logger.info(f"Loading XVLAPolicy from {model_dir} ...")
        self.policy = XVLAPolicy.from_pretrained(model_dir)
        self.policy.eval().to(device)

        logger.info(f"Loading tokenizer ({_TOKENIZER_NAME}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)

        # Tokenize the fixed instruction once
        tok = self.tokenizer(
            instruction,
            max_length=_TOKEN_MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = tok["input_ids"].to(device)   # (1, TOKEN_MAX_LEN)
        logger.info("XVLAWrapper ready.")

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    def _preprocess_image(self, image_np: np.ndarray) -> torch.Tensor:
        """(H, W, 3) uint8 RGB → (1, 3, image_size, image_size) float32."""
        img = torch.from_numpy(image_np).permute(2, 0, 1)            # (3, H, W) uint8
        img = TF.resize(img, [self.image_size, self.image_size],
                         interpolation=TF.InterpolationMode.BILINEAR)
        img = img.float() / 255.0
        img = TF.normalize(img, _IMAGENET_MEAN, _IMAGENET_STD)
        return img.unsqueeze(0)                                       # (1, 3, H, W)

    # ------------------------------------------------------------------
    # Embedding extraction (for RLT Phase 1)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_embeddings(self, image_np: np.ndarray) -> torch.Tensor:
        """Extract Florence-2 encoder features for RL token pretraining.

        Args:
            image_np: (H, W, 3) uint8 RGB

        Returns:
            (num_tokens, 1024) float32 on CPU
        """
        img_t = self._preprocess_image(image_np).to(self.device)
        pixel_values = img_t.unsqueeze(1)                             # (1, 1, 3, H, W)
        image_mask = torch.ones(1, 1, dtype=torch.bool, device=self.device)

        out = self.policy.model.forward_vlm(
            input_ids=self.input_ids,
            pixel_values=pixel_values,
            image_mask=image_mask,
        )
        return out["vlm_features"].squeeze(0).cpu()                   # (num_tokens, 1024)

    # ------------------------------------------------------------------
    # Reference action chunk (for RLT actor conditioning at inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action_chunk(
        self,
        image_np: np.ndarray,
        prop: np.ndarray,
    ) -> np.ndarray:
        """Run XVLA inference and return a reference action chunk in 7-dim TCP format.

        Args:
            image_np: (H, W, 3) uint8 RGB
            prop:     (26,) float32 proprioceptive state

        Returns:
            (chunk_length, 7) float32 — [x, y, z, qx, qy, qz, qw]
        """
        img_t = self._preprocess_image(image_np).to(self.device)

        # XVLA expects 8-dim state: 7 joints + 1 gripper
        # Extract joint_0..6 from prop[19:26]; set gripper = 0
        joints = prop[_JOINT_START:_JOINT_END].astype(np.float32)
        state_8 = np.zeros(8, dtype=np.float32)
        state_8[:7] = joints
        state_t = torch.from_numpy(state_8).unsqueeze(0).to(self.device)   # (1, 8)

        batch = {
            "observation.images.image": img_t,     # (1, 3, H, W)
            OBS_LANGUAGE_TOKENS: self.input_ids,   # (1, TOKEN_MAX_LEN)
            OBS_STATE: state_t,                    # (1, 8)
        }

        # Returns (1, chunk_size, 20) — XVLA's full chunk
        actions = self.policy._get_action_chunk(batch)
        actions_np = actions.squeeze(0).cpu().numpy()   # (chunk_size, 20)

        # Take first chunk_length steps and convert ee6d → (xyz, quat)
        result = np.zeros((self.chunk_length, 7), dtype=np.float32)
        n = min(self.chunk_length, len(actions_np))
        for i in range(n):
            xyz, quat = ee6d_to_quat_xyz(actions_np[i])
            result[i, 0:3] = xyz
            result[i, 3:7] = quat
        return result
