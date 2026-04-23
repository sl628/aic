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

"""RL Token encoder-decoder (Section III-A of the RLT paper).

The encoder compresses the VLA's internal per-layer transformer embeddings
into a single compact readout vector z_rl (the "RL token").  A paired decoder
is trained jointly to autoregressively reconstruct those embeddings from z_rl,
acting as a bottleneck that forces z_rl to capture all task-relevant information.

Architecture (Figure 2 of the paper):
  - VLA embeddings: N×D_vla tokens (stop-gradient)
  - Learnable readout token e_r appended to the sequence
  - Lightweight Transformer encoder g_φ processes the augmented sequence
  - The output at the readout position is z_rl ∈ R^{D_rl}  (default 2048)
  - Transformer decoder d_φ reconstructs original per-layer embeddings from z_rl
  - Reconstruction loss: autoregressive L2 (equation (2) in the paper)
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RLTokenConfig:
    # Dimensionality of each VLA embedding token
    vla_embed_dim: int = (
        7848  # SigLIP 400M (1152) + Gemma 4B (2048×3 layers ≈ 7848 aggregate)
    )
    # Number of VLA embedding tokens per observation
    num_vla_tokens: int = 540  # N (image + language tokens combined)
    # Output RL token dimensionality
    rl_token_dim: int = 2048
    # Encoder transformer
    encoder_num_heads: int = 8
    encoder_num_layers: int = 4
    encoder_ffn_dim: int = 1024
    encoder_dropout: float = 0.1
    # Decoder transformer
    decoder_num_heads: int = 8
    decoder_num_layers: int = 4
    decoder_ffn_dim: int = 1024
    decoder_dropout: float = 0.1
    # Projection dims for encoder input (reduces vla_embed_dim → encoder_dim)
    encoder_dim: int = 512


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


class RLTokenModel(nn.Module):
    """Encoder-decoder that produces and can reconstruct the RL token.

    Usage:
        # During VLA fine-tuning / RL token training
        z_rl, z_rl_sg = model.encode(vla_embeddings)  # z_rl_sg is stop-grad
        loss = model.reconstruction_loss(vla_embeddings, z_rl)

        # During online RL (encoder only, VLA & encoder frozen)
        with torch.no_grad():
            z_rl, _ = model.encode(vla_embeddings)
    """

    def __init__(self, config: RLTokenConfig):
        super().__init__()
        self.config = config
        D_enc = config.encoder_dim
        D_rl = config.rl_token_dim
        D_vla = config.vla_embed_dim

        # --- Input projection: VLA embeddings → encoder working dimension ---
        self.input_proj = nn.Linear(D_vla, D_enc)

        # --- Learnable readout token ---
        self.readout_embed = nn.Parameter(torch.randn(1, 1, D_enc) * 0.02)

        # --- Positional encoding (covers num_vla_tokens + 1 readout) ---
        self.pos_enc = SinusoidalPositionalEncoding(
            D_enc, max_len=config.num_vla_tokens + 1, dropout=config.encoder_dropout
        )

        # --- Encoder transformer ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_enc,
            nhead=config.encoder_num_heads,
            dim_feedforward=config.encoder_ffn_dim,
            dropout=config.encoder_dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=config.encoder_num_layers, enable_nested_tensor=False
        )

        # --- Readout → RL token projection ---
        self.readout_proj = nn.Linear(D_enc, D_rl)

        # --- Decoder: takes z_rl and autoregressively reconstructs VLA embeddings ---
        # The decoder receives z_rl as a memory vector and reconstructs each token
        self.z_rl_to_memory = nn.Linear(D_rl, D_enc)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=D_enc,
            nhead=config.decoder_num_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.decoder_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=config.decoder_num_layers
        )

        # --- Output projection: encoder dim → VLA embed dim for reconstruction ---
        self.output_proj = nn.Linear(D_enc, D_vla)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, vla_embeddings: torch.Tensor):
        """Produce the RL token from VLA internal embeddings.

        Args:
            vla_embeddings: (B, N, D_vla) – stop-gradient applied inside

        Returns:
            z_rl:    (B, D_rl) – differentiable RL token
            z_rl_sg: (B, D_rl) – stop-gradient copy (use as RL state)
        """
        B = vla_embeddings.size(0)

        # Project VLA embeddings to encoder dimension
        x = self.input_proj(vla_embeddings.detach())  # always stop-grad VLA embeddings

        # Append learnable readout token
        readout = self.readout_embed.expand(B, -1, -1)  # (B, 1, D_enc)
        x = torch.cat([x, readout], dim=1)  # (B, N+1, D_enc)

        # Positional encoding + encoder
        x = self.pos_enc(x)
        x = self.encoder(x)  # (B, N+1, D_enc)

        # Extract readout position (last token)
        readout_out = x[:, -1, :]  # (B, D_enc)
        z_rl = self.readout_proj(readout_out)  # (B, D_rl)
        z_rl_sg = z_rl.detach()

        return z_rl, z_rl_sg

    def decode(self, z_rl: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Reconstruct VLA embeddings from z_rl.

        Args:
            z_rl:       (B, D_rl)
            num_tokens: N (number of VLA tokens to reconstruct)

        Returns:
            reconstructed: (B, N, D_vla)
        """
        B = z_rl.size(0)
        D_enc = self.config.encoder_dim

        # z_rl → memory for cross-attention
        memory = self.z_rl_to_memory(z_rl).unsqueeze(1)  # (B, 1, D_enc)

        # Learned positional queries (one per token to reconstruct)
        # We use the positional encoding at positions 0..N-1
        queries = torch.zeros(B, num_tokens, D_enc, device=z_rl.device)
        queries = self.pos_enc.pe[:, :num_tokens].expand(B, -1, -1) + queries

        # Autoregressive mask (causal) for decoder
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            num_tokens, device=z_rl.device
        )

        decoded = self.decoder(queries, memory, tgt_mask=causal_mask)  # (B, N, D_enc)
        reconstructed = self.output_proj(decoded)  # (B, N, D_vla)
        return reconstructed

    def reconstruction_loss(
        self,
        vla_embeddings: torch.Tensor,
        z_rl: torch.Tensor,
        layer_projections: Optional[List[nn.Linear]] = None,
    ) -> torch.Tensor:
        """Autoregressive reconstruction loss (equation (2) in paper).

        L_ro = E[ sum_i || h_φ(d_φ(z_{i-1}, z_{1:i-1}))_i - z_i ||^2 ]

        Args:
            vla_embeddings: (B, N, D_vla) – original VLA embeddings (targets)
            z_rl:           (B, D_rl)     – RL token (may be differentiable)
            layer_projections: optional per-layer linear heads (unused by default)

        Returns:
            scalar loss
        """
        N = vla_embeddings.size(1)
        reconstructed = self.decode(z_rl, num_tokens=N)  # (B, N, D_vla)
        # Shift: predict token i from tokens 0..i-1; skip first query (no target)
        targets = vla_embeddings.detach()
        loss = F.mse_loss(reconstructed, targets)
        return loss
