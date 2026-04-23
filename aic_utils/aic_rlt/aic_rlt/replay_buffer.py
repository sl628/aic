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

"""Replay buffer for off-policy online RL (Section V of the RLT paper).

The buffer stores transitions of the form:
    (x_t, a_{t:t+C-1}, ā_{t:t+C-1}, r_t, x_{t+1})

where:
  x_t   = (z_rl_t, s^p_t)  – RL state (RL token + proprioception)
  a      – executed action chunk (C steps)
  ā      – VLA reference action chunk (may be zero if reference-action-dropout was applied)
  r_t    – scalar reward
  x_{t+1} – next RL state

This buffer supports:
  - VLA warmup episodes (from VLA policy rollouts before RL starts)
  - Online RL episodes (actor rollouts)
  - Human intervention corrections (overwrite executed action)

Subsample stride=2 is applied at *add time* (every other step within a chunk)
so that the stored transitions span the correct horizon for the chunked TD target.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch


@dataclass
class Transition:
    """A single stored transition (all numpy arrays, float32)."""

    z_rl: np.ndarray  # (D_rl,)
    prop: np.ndarray  # (prop_dim,)
    action_chunk: np.ndarray  # (C, action_dim)  – executed (or human) actions
    ref_action_chunk: np.ndarray  # (C, action_dim) – VLA reference (may be zeros)
    reward: float
    next_z_rl: np.ndarray  # (D_rl,)
    next_prop: np.ndarray  # (prop_dim,)
    done: bool


class ReplayBuffer:
    """Fixed-size circular replay buffer.

    Stores transitions as contiguous numpy arrays for fast batch sampling.
    """

    def __init__(
        self,
        capacity: int,
        rl_token_dim: int,
        prop_dim: int,
        action_dim: int,
        chunk_length: int,
        device: torch.device,
    ):
        self.capacity = capacity
        self.rl_token_dim = rl_token_dim
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.chunk_length = chunk_length
        self.device = device

        self._ptr: int = 0
        self._size: int = 0

        C: int = chunk_length
        D: int = action_dim
        self._z_rl: np.ndarray = np.zeros((capacity, rl_token_dim), dtype=np.float32)
        self._prop: np.ndarray = np.zeros((capacity, prop_dim), dtype=np.float32)
        self._action_chunk: np.ndarray = np.zeros((capacity, C, D), dtype=np.float32)
        self._ref_action_chunk: np.ndarray = np.zeros((capacity, C, D), dtype=np.float32)
        self._reward: np.ndarray = np.zeros((capacity,), dtype=np.float32)
        self._next_z_rl: np.ndarray = np.zeros((capacity, rl_token_dim), dtype=np.float32)
        self._next_prop: np.ndarray = np.zeros((capacity, prop_dim), dtype=np.float32)
        self._done: np.ndarray = np.zeros((capacity,), dtype=np.float32)

    def add(self, transition: Transition) -> None:
        idx: int = self._ptr
        self._z_rl[idx] = transition.z_rl
        self._prop[idx] = transition.prop
        self._action_chunk[idx] = transition.action_chunk
        self._ref_action_chunk[idx] = transition.ref_action_chunk
        self._reward[idx] = transition.reward
        self._next_z_rl[idx] = transition.next_z_rl
        self._next_prop[idx] = transition.next_prop
        self._done[idx] = float(transition.done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch of transitions.

        Returns a dict of tensors on self.device.
        """
        assert (
            self._size >= batch_size
        ), f"Buffer has {self._size} transitions, need {batch_size}"
        idxs: np.ndarray = np.random.randint(0, self._size, size=batch_size)

        def to_tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr[idxs]).to(self.device)

        return {
            "z_rl": to_tensor(self._z_rl),
            "prop": to_tensor(self._prop),
            "action_chunk": to_tensor(self._action_chunk),
            "ref_action_chunk": to_tensor(self._ref_action_chunk),
            "reward": to_tensor(self._reward),
            "next_z_rl": to_tensor(self._next_z_rl),
            "next_prop": to_tensor(self._next_prop),
            "done": to_tensor(self._done),
        }

    def __len__(self) -> int:
        return self._size

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            z_rl=self._z_rl[: self._size],
            prop=self._prop[: self._size],
            action_chunk=self._action_chunk[: self._size],
            ref_action_chunk=self._ref_action_chunk[: self._size],
            reward=self._reward[: self._size],
            next_z_rl=self._next_z_rl[: self._size],
            next_prop=self._next_prop[: self._size],
            done=self._done[: self._size],
        )

    def load(self, path: str) -> None:
        data: Dict[str, np.ndarray] = dict(np.load(path))
        n: int = data["z_rl"].shape[0]
        assert (
            n <= self.capacity
        ), f"Saved buffer ({n}) exceeds capacity ({self.capacity})"
        for key in (
            "z_rl",
            "prop",
            "action_chunk",
            "ref_action_chunk",
            "reward",
            "next_z_rl",
            "next_prop",
            "done",
        ):
            buffer_attr: np.ndarray = getattr(self, f"_{key}")
            buffer_attr[:n] = data[key]
        self._size = n
        self._ptr = n % self.capacity
