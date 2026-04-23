import torch

from aic_rlt.models.rl_token import RLTokenModel

B = 4


def test_encode_output_shape(rl_token_config):
    model = RLTokenModel(rl_token_config)
    emb = torch.randn(B, rl_token_config.num_vla_tokens, rl_token_config.vla_embed_dim)
    z_rl, z_rl_sg = model.encode(emb)
    assert z_rl.shape == (B, rl_token_config.rl_token_dim)
    assert z_rl_sg.shape == z_rl.shape


def test_encode_stop_gradient(rl_token_config):
    model = RLTokenModel(rl_token_config)
    emb = torch.randn(B, rl_token_config.num_vla_tokens, rl_token_config.vla_embed_dim)
    _, z_rl_sg = model.encode(emb)
    assert not z_rl_sg.requires_grad


def test_decode_output_shape(rl_token_config):
    model = RLTokenModel(rl_token_config)
    z_rl = torch.randn(B, rl_token_config.rl_token_dim)
    N = rl_token_config.num_vla_tokens
    out = model.decode(z_rl, num_tokens=N)
    assert out.shape == (B, N, rl_token_config.vla_embed_dim)


def test_reconstruction_loss_scalar(rl_token_config):
    model = RLTokenModel(rl_token_config)
    emb = torch.randn(B, rl_token_config.num_vla_tokens, rl_token_config.vla_embed_dim)
    z_rl, _ = model.encode(emb)
    loss = model.reconstruction_loss(emb, z_rl)
    assert loss.dim() == 0
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_reconstruction_loss_decreases(rl_token_config):
    model = RLTokenModel(rl_token_config)
    emb = torch.randn(B, rl_token_config.num_vla_tokens, rl_token_config.vla_embed_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    z_rl, _ = model.encode(emb)
    loss_initial = model.reconstruction_loss(emb, z_rl).item()

    for _ in range(20):
        z_rl, _ = model.encode(emb)
        loss = model.reconstruction_loss(emb, z_rl)
        opt.zero_grad()
        loss.backward()
        opt.step()

    z_rl, _ = model.encode(emb)
    loss_final = model.reconstruction_loss(emb, z_rl).item()
    assert loss_final < loss_initial
