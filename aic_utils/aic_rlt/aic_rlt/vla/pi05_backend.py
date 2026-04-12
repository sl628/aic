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

"""Pi0.5 (openpi) VLA backend for RLT.

Loads the Pi0.5 model via the openpi library (JAX), extracts prefix embeddings
(SigLIP vision + PaliGemma text tokens), and produces reference action chunks.

The model is completely frozen — RLT never modifies Pi0.5 weights.

Requires:
  - openpi repository at /home/yifeng/workspace/openpi (or override OPENPI_ROOT env)
  - openpi venv installed (uv sync)
  - Downloaded pi05_base checkpoint
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .base import VLABackend

logger = logging.getLogger(__name__)

OPENPI_ROOT = Path(os.environ.get("OPENPI_ROOT", "/home/yifeng/workspace/openpi"))


def _setup_openpi_path():
    """Add openpi source and its venv site-packages to sys.path."""
    openpi_src = OPENPI_ROOT / "src"
    openpi_client_src = OPENPI_ROOT / "packages" / "openpi-client" / "src"
    venv_site = OPENPI_ROOT / ".venv" / "lib"

    site_packages = None
    if venv_site.exists():
        for p in venv_site.iterdir():
            sp = p / "site-packages"
            if sp.exists():
                site_packages = sp
                break

    paths_to_add = [str(openpi_src), str(openpi_client_src)]
    if site_packages:
        paths_to_add.append(str(site_packages))

    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)


class Pi05Backend(VLABackend):
    """Pi0.5 (PaliGemma-based) VLA backend.

    Uses a single forward pass for both embedding extraction and action
    generation — more efficient than calling them separately.

    Parameters
    ----------
    checkpoint_dir : str
        Path to the downloaded pi05_base checkpoint directory.
    device : torch.device
        Torch device (Pi0.5 runs in JAX but tensors are moved here).
    chunk_length : int
        Number of actions to return per chunk.
    action_dim : int
        Number of action dimensions to keep from Pi0.5's 32D output.
    openpi_config : str
        OpenPI config name (default "pi05_aloha").
    instruction : str
        Language prompt for the task.
    num_denoise_steps : int
        Flow matching denoising steps (fewer = faster).
    camera_mapping : dict
        AIC camera name → Pi0.5 image slot mapping.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        device: torch.device,
        chunk_length: int = 10,
        action_dim: int = 7,
        openpi_config: str = "pi05_aloha",
        instruction: str = "insert the cable into the port",
        num_denoise_steps: int = 10,
        camera_mapping: dict | None = None,
    ):
        self.device = device
        self.chunk_length = chunk_length
        self.action_dim = action_dim
        self._checkpoint_dir = checkpoint_dir
        self._openpi_config = openpi_config
        self._instruction = instruction
        self._num_denoise_steps = num_denoise_steps
        self._camera_mapping = camera_mapping or {
            "center": "base_0_rgb",
            "left": "left_wrist_0_rgb",
            "right": "right_wrist_0_rgb",
        }

        # Lazy-loaded state
        self._policy = None
        self._model_obj = None
        self._loaded = False

        # Set after loading
        self.embed_dim: int = 2048
        self.num_tokens: int = 0

        # Load immediately
        self._ensure_loaded()

    def _ensure_loaded(self):
        if self._loaded:
            return

        logger.info("Loading Pi0.5 model from %s ...", self._checkpoint_dir)
        _setup_openpi_path()

        import jax
        import jax.numpy as jnp
        from openpi.training import config as openpi_config
        from openpi.policies import policy_config
        from openpi.models import model as _model

        train_config = openpi_config.get_config(self._openpi_config)
        checkpoint_dir = Path(self._checkpoint_dir)

        self._policy = policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
            default_prompt=self._instruction,
        )
        self._model_obj = self._policy._model

        # Probe dimensions
        batch_size = 1
        dummy_img = jnp.zeros((batch_size, 224, 224, 3), dtype=jnp.float32)
        dummy_mask = jnp.ones((batch_size,), dtype=jnp.bool_)
        obs = _model.Observation(
            images={k: dummy_img for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
            image_masks={k: dummy_mask for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
            state=jnp.zeros((batch_size, 32), dtype=jnp.float32),
            tokenized_prompt=jnp.zeros((batch_size, 200), dtype=jnp.int32),
            tokenized_prompt_mask=jnp.zeros((batch_size, 200), dtype=jnp.bool_),
        )
        obs = _model.preprocess_observation(None, obs, train=False)
        prefix_tokens, _, _ = self._model_obj.embed_prefix(obs)
        self.num_tokens = int(prefix_tokens.shape[1])
        self.embed_dim = int(prefix_tokens.shape[2])

        self._loaded = True
        logger.info(
            "Pi05Backend ready: num_tokens=%d, embed_dim=%d", self.num_tokens, self.embed_dim
        )

    def _obs_to_pi05_input(self, obs) -> dict:
        """Convert AIC ROS Observation to pi0.5 input dict."""
        import cv2

        images = {}
        for aic_key, pi05_key in self._camera_mapping.items():
            if aic_key == "left":
                raw_img = obs.left_image
            elif aic_key == "center":
                raw_img = obs.center_image
            elif aic_key == "right":
                raw_img = obs.right_image
            else:
                raise ValueError(f"Unknown camera key: {aic_key}")

            img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
                raw_img.height, raw_img.width, 3
            )
            img_np = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_AREA)
            images[pi05_key] = img_np

        joint_positions = list(obs.joint_states.position[:6])
        gripper = [obs.joint_states.position[6]] if len(obs.joint_states.position) > 6 else [0.0]
        state = np.array(joint_positions + gripper, dtype=np.float32)

        return {"state": state, "images": images, "prompt": self._instruction}

    def _run_forward_with_embeddings(self, obs) -> tuple[np.ndarray, np.ndarray]:
        """Run pi0.5 forward pass → (prefix_embeddings, actions).

        Returns:
            prefix_embeddings: (N, D_vla) float32 numpy
            actions: (action_horizon, action_dim) float32 numpy
        """
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        self._ensure_loaded()

        pi05_input = self._obs_to_pi05_input(obs)
        inputs = dict(pi05_input)
        inputs = self._policy._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        observation = _model.Observation.from_dict(inputs)
        observation = _model.preprocess_observation(None, observation, train=False)

        # Extract prefix embeddings (bfloat16 → float32 via JAX cast)
        prefix_tokens, _, _ = self._model_obj.embed_prefix(observation)
        prefix_f32 = prefix_tokens[0].astype(jnp.float32)
        prefix_embeds = np.array(jax.device_get(prefix_f32), dtype=np.float32)

        # Get actions via flow matching
        rng = jax.random.key(0)
        actions_jax = self._model_obj.sample_actions(
            rng, observation, num_steps=self._num_denoise_steps
        )

        # Unnormalize actions
        actions_np = np.array(jax.device_get(actions_jax[0].astype(jnp.float32)), dtype=np.float32)
        outputs = {"actions": actions_np, "state": np.array(jax.device_get(inputs["state"][0].astype(jnp.float32)), dtype=np.float32)}
        outputs = self._policy._output_transform(outputs)
        actions_np = outputs["actions"][:, :self.action_dim]

        return prefix_embeds, actions_np

    # ------------------------------------------------------------------
    # VLABackend interface
    # ------------------------------------------------------------------

    def get_embeddings(self, obs) -> torch.Tensor:
        """(1, num_tokens, embed_dim) on device."""
        prefix_embeds, _ = self._run_forward_with_embeddings(obs)
        return torch.from_numpy(prefix_embeds).unsqueeze(0).to(self.device)

    def get_action_chunk(self, obs) -> np.ndarray:
        """(chunk_length, action_dim) float32."""
        _, actions = self._run_forward_with_embeddings(obs)
        return actions[:self.chunk_length].astype(np.float32)

    def get_embeddings_and_actions(self, obs) -> tuple:
        """Single Pi0.5 forward pass — more efficient than two separate calls."""
        prefix_embeds, actions = self._run_forward_with_embeddings(obs)
        embeddings = torch.from_numpy(prefix_embeds).unsqueeze(0).to(self.device)
        action_chunk = actions[:self.chunk_length].astype(np.float32)
        return embeddings, action_chunk
