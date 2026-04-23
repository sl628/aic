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
    """Make openpi and openpi_client importable.

    Two supported environments:
      - aic pixi `pi05` env: openpi + openpi-client are installed editable;
        this function is a no-op (detected by successful `import openpi`).
      - fallback (historical): openpi lives at OPENPI_ROOT with its own
        Python 3.11 uv venv; we prepend openpi's sources + venv site-packages
        to sys.path. Only safe when the host interpreter is also 3.11 and
        openpi isn't already installed — otherwise cp311 C extensions shadow
        the host's compatible ones (e.g. tensorstore) and import breaks.
    """
    try:
        import openpi  # noqa: F401
        import openpi_client  # noqa: F401

        return  # already importable from the current env; do not taint sys.path
    except ImportError:
        pass

    openpi_src = OPENPI_ROOT / "src"
    openpi_client_src = OPENPI_ROOT / "packages" / "openpi-client" / "src"
    venv_site = OPENPI_ROOT / ".venv" / "lib"

    site_packages = None
    if venv_site.exists():
        # Only use the fallback venv if its Python matches the host interpreter.
        host_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
        for p in venv_site.iterdir():
            if p.name != host_tag:
                continue
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
        asset_id: str = "ur5e",
    ):
        self.device = device
        self.chunk_length = chunk_length
        self.action_dim = action_dim
        # Option B: pi0.5 is used as a frozen feature extractor only. Its
        # joint-space predictions are OOD for aic's workspace and were not
        # used as BC targets during training. Inference should skip the
        # denoise loop and pass ref_chunk=None to the actor.
        self.actions_are_bc_targets = False
        self._checkpoint_dir = checkpoint_dir
        self._openpi_config = openpi_config
        self._asset_id = asset_id
        self._instruction = instruction
        self._num_denoise_steps = num_denoise_steps
        # Per openpi ur5 recipe: 2 cams used (base + single wrist); right is masked.
        # aic records 3 cams (left/center/right) — center=base, left=wrist, right unused.
        self._camera_mapping = camera_mapping or {
            "center": "base_0_rgb",
            "left": "left_wrist_0_rgb",
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

        logger.info(
            "Loading Pi0.5 model from %s (asset_id=%s) ...",
            self._checkpoint_dir,
            self._asset_id,
        )
        _setup_openpi_path()

        import dataclasses
        import jax.numpy as jnp
        from openpi.training import config as openpi_config
        from openpi.training import checkpoints as _checkpoints
        from openpi.policies import policy_config
        from openpi.models import model as _model

        # Build a TrainConfig whose DataConfig points at our chosen asset_id
        # (e.g. "ur5e") with UR5-appropriate transforms. We start from pi05_aloha
        # (for its Pi0Config(pi05=True) model settings and its repack/transform
        # structure), then replace the data factory wholesale with a UR5 one.
        base_config = openpi_config.get_config(self._openpi_config)
        from ._ur5_transforms import make_ur5_data_config_cls

        Pi05UR5DataConfig = make_ur5_data_config_cls()
        # Point assets_dir at the checkpoint's own assets/ so the internal
        # create_base_config doesn't warn about a missing norm_stats path
        # derived from pi05_aloha's default train-time assets layout.
        ckpt_assets_dir = str(Path(self._checkpoint_dir) / "assets")
        ur5_data_factory = Pi05UR5DataConfig(
            assets=openpi_config.AssetsConfig(
                assets_dir=ckpt_assets_dir,
                asset_id=self._asset_id,
            ),
        )
        train_config = dataclasses.replace(base_config, data=ur5_data_factory)

        checkpoint_dir = Path(self._checkpoint_dir)

        # Load norm_stats from the ur5e sub-folder of the shipped assets, since
        # the default asset_id="trossen" (inherited from pi05_aloha) doesn't apply.
        norm_stats = _checkpoints.load_norm_stats(
            checkpoint_dir / "assets",
            self._asset_id,
        )

        self._policy = policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
            default_prompt=self._instruction,
            norm_stats=norm_stats,
        )
        self._model_obj = self._policy._model

        # Probe prefix dims with a real instruction so embed_prefix returns
        # language-bearing tokens (not image-only as a zero-prompt probe would).
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        probe_input = {
            "joints": np.zeros(6, dtype=np.float32),
            "gripper": np.zeros(1, dtype=np.float32),
            "base_rgb": dummy_img,
            "wrist_rgb": dummy_img,
            "prompt": self._instruction,
        }
        inputs = self._policy._input_transform(probe_input)
        import jax

        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        probe_obs = _model.Observation.from_dict(inputs)
        probe_obs = _model.preprocess_observation(None, probe_obs, train=False)
        prefix_tokens, _, _ = self._model_obj.embed_prefix(probe_obs)
        self.num_tokens = int(prefix_tokens.shape[1])
        self.embed_dim = int(prefix_tokens.shape[2])

        self._loaded = True
        logger.info(
            "Pi05Backend ready: num_tokens=%d, embed_dim=%d, action_dim=%d",
            self.num_tokens,
            self.embed_dim,
            self.action_dim,
        )

    def _obs_to_pi05_input(self, obs) -> dict:
        """Convert AIC ROS Observation to the UR5Inputs schema.

        UR5Inputs (see _ur5_transforms.py) expects:
          joints:     (6,) radians
          gripper:    (1,) [0..1]
          base_rgb:   (H,W,3) uint8 — center camera → pi0.5 base_0_rgb
          wrist_rgb:  (H,W,3) uint8 — left camera   → pi0.5 left_wrist_0_rgb
          prompt:     instruction string
        """
        import cv2

        def _extract(aic_key: str):
            if aic_key == "left":
                raw = obs.left_image
            elif aic_key == "center":
                raw = obs.center_image
            elif aic_key == "right":
                raw = obs.right_image
            else:
                raise ValueError(f"Unknown camera key: {aic_key}")
            img_np = np.frombuffer(raw.data, dtype=np.uint8).reshape(
                raw.height, raw.width, 3
            )
            return cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_AREA)

        # Fixed roles: center=base, left=wrist (UR5 uses 2 cams; right is unused)
        base_rgb = _extract("center")
        wrist_rgb = _extract("left")

        joints = np.asarray(obs.joint_states.position[:6], dtype=np.float32)
        gripper = np.asarray(
            (
                [obs.joint_states.position[6]]
                if len(obs.joint_states.position) > 6
                else [0.0]
            ),
            dtype=np.float32,
        )

        return {
            "joints": joints,
            "gripper": gripper,
            "base_rgb": base_rgb,
            "wrist_rgb": wrist_rgb,
            "prompt": self._instruction,
        }

    def frame_to_backend_input(
        self,
        joints: np.ndarray,
        gripper: np.ndarray,
        base_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
    ) -> dict:
        """Offline-extraction adapter: parquet frame → UR5Inputs schema.

        Used by prepare_embeddings.py so extraction doesn't have to construct a
        ROS Observation from parquet rows. Returns the same dict as
        _obs_to_pi05_input so _run_forward_with_embeddings can consume it.
        """
        return {
            "joints": np.asarray(joints, dtype=np.float32),
            "gripper": np.asarray(gripper, dtype=np.float32),
            "base_rgb": np.asarray(base_rgb),
            "wrist_rgb": np.asarray(wrist_rgb),
            "prompt": self._instruction,
        }

    def _run_forward_with_embeddings(
        self,
        obs,
        *,
        want_actions: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Run pi0.5 forward pass → (prefix_embeddings, actions_or_None).

        `obs` may be a ROS Observation (online path) OR an already-built UR5Inputs
        dict (offline extraction path, constructed via frame_to_backend_input).

        When ``want_actions=False``, skips the ``sample_actions`` flow-matching
        denoise loop entirely — ~2× faster and the right choice for
        feature-extractor-only usage (Option B: pi0.5 provides embeddings,
        BC targets come from demos).

        Returns:
            prefix_embeddings: (N, D_vla) float32 numpy
            actions: (action_horizon, action_dim) float32 numpy, or None if not requested
        """
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        self._ensure_loaded()

        if isinstance(obs, dict):
            pi05_input = obs
        else:
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

        if not want_actions:
            return prefix_embeds, None

        # Get actions via flow matching
        rng = jax.random.key(0)
        actions_jax = self._model_obj.sample_actions(
            rng, observation, num_steps=self._num_denoise_steps
        )

        # Unnormalize actions
        actions_np = np.array(
            jax.device_get(actions_jax[0].astype(jnp.float32)), dtype=np.float32
        )
        outputs = {
            "actions": actions_np,
            "state": np.array(
                jax.device_get(inputs["state"][0].astype(jnp.float32)), dtype=np.float32
            ),
        }
        outputs = self._policy._output_transform(outputs)
        actions_np = outputs["actions"][:, : self.action_dim]

        return prefix_embeds, actions_np

    # ------------------------------------------------------------------
    # VLABackend interface
    # ------------------------------------------------------------------

    def set_instruction(self, instruction: str) -> None:
        """Swap the active Pi0.5 prompt.

        Pi0.5's create_trained_policy stores the default_prompt internally,
        but `_obs_to_pi05_input` re-emits it per call via ``self._instruction``,
        so updating the attribute is sufficient for subsequent inferences.
        """
        if instruction != self._instruction:
            self._instruction = instruction

    def get_embeddings(self, obs) -> torch.Tensor:
        """(1, num_tokens, embed_dim) on device. Skips the denoise loop."""
        prefix_embeds, _ = self._run_forward_with_embeddings(obs, want_actions=False)
        return torch.from_numpy(prefix_embeds).unsqueeze(0).to(self.device)

    def get_action_chunk(self, obs) -> np.ndarray:
        """(chunk_length, action_dim) float32."""
        _, actions = self._run_forward_with_embeddings(obs)
        return actions[: self.chunk_length].astype(np.float32)

    def get_embeddings_and_actions(self, obs) -> tuple:
        """Single Pi0.5 forward pass — more efficient than two separate calls."""
        prefix_embeds, actions = self._run_forward_with_embeddings(obs)
        embeddings = torch.from_numpy(prefix_embeds).unsqueeze(0).to(self.device)
        action_chunk = actions[: self.chunk_length].astype(np.float32)
        return embeddings, action_chunk
