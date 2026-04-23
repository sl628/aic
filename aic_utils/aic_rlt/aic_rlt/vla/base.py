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

"""Abstract VLA backend interface for RLT.

All VLA backends must accept AIC ROS Observation messages and return:
  - get_embeddings(obs)  -> torch.Tensor (1, num_tokens, embed_dim) on device
  - get_action_chunk(obs) -> np.ndarray (chunk_length, action_dim)

Backends also expose:
  - embed_dim:   int   (per-token embedding dimensionality)
  - num_tokens:  int   (number of tokens per observation)
  - action_dim:  int   (dimensionality of actions the backend emits)
  - actions_are_bc_targets: bool
      Whether the actions returned by get_action_chunk(...) are in the same
      distribution/space as the BC targets the RLT actor was trained against.
      True for backends whose native action output matches aic's demos (e.g.
      XVLA is TCP-native, same as aic's recorded demo actions).
      False for backends whose actions don't match training BC targets (e.g.
      Pi0.5 under "Option B" — used as a frozen feature extractor only; its
      joint-space predictions are OOD for aic's workspace and were not used
      as BC targets). When False, inference should call get_embeddings(...)
      and pass ref_chunk=None to the actor.

These dimensions are used to construct RLTokenConfig / ActorCriticConfig.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch


class VLABackend(ABC):
    """Abstract base class for VLA backends used by RLT.

    Subclasses wrap a specific VLA model (XVLA, Pi0.5, ACT, …) and expose a
    uniform interface so RunRLT and the training scripts are backend-agnostic.
    """

    # Subclasses must set these after the model is loaded.
    embed_dim: int
    num_tokens: int
    action_dim: int
    # Default: most backends' actions ARE the BC target distribution (true for
    # TCP-native backends like XVLA). Pi0.5/Option-B overrides this to False.
    actions_are_bc_targets: bool = True

    @abstractmethod
    def get_embeddings(self, obs) -> torch.Tensor:
        """Extract internal VLA embeddings from a ROS Observation.

        Args:
            obs: aic_model_interfaces.msg.Observation

        Returns:
            torch.Tensor of shape (1, num_tokens, embed_dim) on the backend's device.
        """

    @abstractmethod
    def get_action_chunk(self, obs) -> np.ndarray:
        """Run VLA inference to produce the reference action chunk.

        Args:
            obs: aic_model_interfaces.msg.Observation

        Returns:
            np.ndarray of shape (chunk_length, action_dim), float32.
        """

    def set_instruction(self, instruction: str) -> None:
        """Swap the active language prompt (e.g. for phase-conditioned RLT).

        Default is a no-op so backends that don't support runtime prompt
        switching don't break. Backends that cache tokenized prompts should
        override this to re-tokenize and update their internal state.
        """
        # no-op default
        return None

    def get_embeddings_and_actions(self, obs) -> tuple:
        """Get embeddings and action chunk in one call.

        Default: two separate calls.  Override when a single forward pass
        returns both (e.g. Pi0.5 — avoids redundant computation).

        Returns:
            (embeddings, action_chunk):
                embeddings:   torch.Tensor (1, num_tokens, embed_dim)
                action_chunk: np.ndarray   (chunk_length, action_dim)
        """
        return self.get_embeddings(obs), self.get_action_chunk(obs)
