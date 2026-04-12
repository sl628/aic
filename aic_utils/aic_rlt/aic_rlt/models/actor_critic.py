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

"""Lightweight actor and critic MLPs for online RL (Section III-B of the RLT paper).

The actor and critic both receive the RL state x = (z_rl, s^p):
  - z_rl: RL token (D_rl-dim), produced by the frozen RL token encoder
  - s^p:  proprioceptive state (joint positions, TCP pose/velocity, etc.)

Actor (equation (4)):
    π_θ(ā_{1:C} | x, ā_{1:C}) = N(μ_θ(x, ā_{1:C}), σ²I)
    - Gaussian policy conditioned on state AND the VLA reference action chunk ā
    - Outputs the mean of a factored Gaussian over action chunks
    - Reference action dropout at 50% to prevent over-reliance (Section III-B)

Critic (equation (3)):
    Q_ψ(x, a_{1:C}) – ensemble of 2 Q-networks (TD3-style)
    - Takes state x and the full action chunk a_{1:C} as input
    - Returns scalar value estimate
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ActorCriticConfig:
    # RL token dimension (output of RLTokenModel.encode)
    rl_token_dim: int = 2048
    # Proprioceptive state dimension (TCP pose 7 + TCP vel 6 + TCP error 6 + joints 7 = 26)
    prop_dim: int = 26
    # Action dimension (6D velocity delta + gripper = 7)
    action_dim: int = 7
    # Action chunk length C
    chunk_length: int = 10
    # Hidden layer sizes for actor and critic
    # Paper uses 256×2 for Ethernet/charger/zip-tie and 512×3 for screw
    hidden_dims: List[int] = None
    # Dropout for actor/critic (not used in paper but good practice)
    dropout: float = 0.0
    # Reference action dropout probability (Section III-B, 50%)
    ref_action_dropout: float = 0.5
    # Minimum log std for actor Gaussian
    log_std_min: float = -5.0
    # Maximum log std for actor Gaussian
    log_std_max: float = 2.0

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    dropout: float = 0.0,
    activate_final: bool = False,
) -> nn.Sequential:
    layers = []
    in_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.LayerNorm(h))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    if activate_final:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Gaussian actor π_θ(ā_{1:C} | x, ā_{1:C}).

    Input:
        state x = concat(z_rl, s^p)                 shape (B, D_rl + prop_dim)
        ref_action_chunk ā_{1:C} (VLA reference)     shape (B, C*action_dim)

    Output:
        action chunk mean μ                           shape (B, C, action_dim)
        action chunk log std                          shape (B, C, action_dim)
    """

    def __init__(self, config: ActorCriticConfig):
        super().__init__()
        self.config = config
        C, D = config.chunk_length, config.action_dim

        # State encoder (z_rl + s^p)
        state_dim = config.rl_token_dim + config.prop_dim
        # Reference action encoder (VLA action chunk, flattened)
        ref_dim = C * D

        # Shared trunk takes concatenation of state + (optionally zeroed) ref action
        input_dim = state_dim + ref_dim
        output_dim = C * D  # mean for each action in chunk

        self.trunk = build_mlp(
            input_dim=input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=output_dim,
            dropout=config.dropout,
        )
        # Separate head for log_std (state-independent, learned parameter)
        self.log_std = nn.Parameter(torch.zeros(C * D))

    def forward(
        self,
        z_rl: torch.Tensor,
        prop: torch.Tensor,
        ref_action_chunk: Optional[torch.Tensor],
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_rl:              (B, D_rl)
            prop:              (B, prop_dim)
            ref_action_chunk:  (B, C, action_dim) or None
            training:          whether to apply reference action dropout

        Returns:
            mu:      (B, C, action_dim)
            log_std: (B, C, action_dim)
        """
        B = z_rl.size(0)
        C, D = self.config.chunk_length, self.config.action_dim

        # Flatten and optionally dropout reference action chunk
        if ref_action_chunk is not None:
            ref_flat = ref_action_chunk.reshape(B, -1)  # (B, C*D)
            if training and self.config.ref_action_dropout > 0:
                # Reference action dropout: zero out entire chunk for a fraction of samples
                # (Section III-B: replace with zeros for random subset of transitions)
                mask = torch.bernoulli(
                    torch.full((B, 1), 1.0 - self.config.ref_action_dropout, device=z_rl.device)
                )
                ref_flat = ref_flat * mask
        else:
            ref_flat = torch.zeros(B, C * D, device=z_rl.device)

        x = torch.cat([z_rl, prop, ref_flat], dim=-1)  # (B, D_rl + prop_dim + C*D)
        mu = self.trunk(x).reshape(B, C, D)             # (B, C, action_dim)

        log_std = self.log_std.clamp(self.config.log_std_min, self.config.log_std_max)
        log_std = log_std.reshape(1, C, D).expand(B, -1, -1)

        return mu, log_std

    def sample(
        self,
        z_rl: torch.Tensor,
        prop: torch.Tensor,
        ref_action_chunk: Optional[torch.Tensor],
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action chunk and return (action, log_prob)."""
        mu, log_std = self.forward(z_rl, prop, ref_action_chunk, training=training)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()  # (B, C, action_dim)
        log_prob = dist.log_prob(action).sum(dim=(-1, -2))  # (B,)
        return action, log_prob

    def get_mean(
        self,
        z_rl: torch.Tensor,
        prop: torch.Tensor,
        ref_action_chunk: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Deterministic action (mean), used at test time."""
        mu, _ = self.forward(z_rl, prop, ref_action_chunk, training=False)
        return mu


# ---------------------------------------------------------------------------
# Critic (Q-network ensemble)
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Single Q-network: Q(x, a_{1:C}) → scalar."""

    def __init__(self, config: ActorCriticConfig):
        super().__init__()
        C, D = config.chunk_length, config.action_dim
        state_dim = config.rl_token_dim + config.prop_dim
        action_dim = C * D
        self.net = build_mlp(
            input_dim=state_dim + action_dim,
            hidden_dims=config.hidden_dims,
            output_dim=1,
            dropout=config.dropout,
        )

    def forward(self, z_rl: torch.Tensor, prop: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_rl:   (B, D_rl)
            prop:   (B, prop_dim)
            action: (B, C, action_dim) or (B, C*action_dim)

        Returns:
            q: (B,)
        """
        B = z_rl.size(0)
        a_flat = action.reshape(B, -1)
        x = torch.cat([z_rl, prop, a_flat], dim=-1)
        return self.net(x).squeeze(-1)  # (B,)


class Critic(nn.Module):
    """Twin Q-network ensemble (TD3-style, equation (3) in paper).

    Returns the minimum of two Q-values to reduce overestimation bias.
    """

    def __init__(self, config: ActorCriticConfig, num_critics: int = 2):
        super().__init__()
        self.q_networks = nn.ModuleList([QNetwork(config) for _ in range(num_critics)])

    def forward(
        self, z_rl: torch.Tensor, prop: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """Return all Q-values (one per network)."""
        return tuple(q(z_rl, prop, action) for q in self.q_networks)

    def min_q(
        self, z_rl: torch.Tensor, prop: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Return element-wise minimum Q across all networks (B,)."""
        qs = self.forward(z_rl, prop, action)
        return torch.stack(qs, dim=0).min(dim=0).values
