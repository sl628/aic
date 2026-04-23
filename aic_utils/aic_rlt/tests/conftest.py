import pytest
import torch

from aic_rlt.models.actor_critic import ActorCriticConfig
from aic_rlt.models.rl_token import RLTokenConfig
from aic_rlt.trainer import RewardConfig


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def ac_config():
    return ActorCriticConfig(
        rl_token_dim=64,
        prop_dim=26,
        action_dim=9,
        chunk_length=4,
        hidden_dims=[32],
    )


@pytest.fixture
def rl_token_config():
    return RLTokenConfig(
        vla_embed_dim=32,
        num_vla_tokens=8,
        rl_token_dim=64,
        encoder_dim=32,
        encoder_num_heads=4,
        encoder_num_layers=1,
        encoder_ffn_dim=64,
        decoder_num_heads=4,
        decoder_num_layers=1,
        decoder_ffn_dim=64,
    )


@pytest.fixture
def reward_config():
    return RewardConfig()
