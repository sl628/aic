import torch

from aic_rlt.models.actor_critic import Actor, Critic, build_mlp


B, D_rl, PROP, ACT, C = 4, 64, 26, 9, 4


def test_build_mlp_output_shape():
    mlp = build_mlp(input_dim=10, hidden_dims=[32, 32], output_dim=5)
    out = mlp(torch.randn(3, 10))
    assert out.shape == (3, 5)


def test_actor_forward_shape(ac_config):
    actor = Actor(ac_config)
    z_rl = torch.randn(B, D_rl)
    prop = torch.randn(B, PROP)
    ref = torch.randn(B, C, ACT)
    mu, log_std = actor(z_rl, prop, ref)
    assert mu.shape == (B, C, ACT)
    assert log_std.shape == (B, C, ACT)


def test_actor_forward_no_ref(ac_config):
    actor = Actor(ac_config)
    mu, log_std = actor(torch.randn(B, D_rl), torch.randn(B, PROP), None)
    assert mu.shape == (B, C, ACT)


def test_actor_sample_shape(ac_config):
    actor = Actor(ac_config)
    action, log_prob = actor.sample(
        torch.randn(B, D_rl), torch.randn(B, PROP), torch.randn(B, C, ACT)
    )
    assert action.shape == (B, C, ACT)
    assert log_prob.shape == (B,)


def test_actor_get_mean_deterministic(ac_config):
    actor = Actor(ac_config)
    actor.eval()
    z = torch.randn(B, D_rl)
    p = torch.randn(B, PROP)
    r = torch.randn(B, C, ACT)
    m1 = actor.get_mean(z, p, r)
    m2 = actor.get_mean(z, p, r)
    assert torch.allclose(m1, m2)


def test_actor_ref_dropout_all(ac_config):
    ac_config.ref_action_dropout = 1.0
    actor = Actor(ac_config)
    z = torch.randn(B, D_rl)
    p = torch.randn(B, PROP)
    ref = torch.randn(B, C, ACT)
    mu_with_ref, _ = actor(z, p, ref, training=True)
    mu_no_ref, _ = actor(z, p, None, training=True)
    assert torch.allclose(mu_with_ref, mu_no_ref, atol=1e-6)


def test_critic_forward_shape(ac_config):
    critic = Critic(ac_config)
    z = torch.randn(B, D_rl)
    p = torch.randn(B, PROP)
    a = torch.randn(B, C, ACT)
    qs = critic(z, p, a)
    assert len(qs) == 2
    for q in qs:
        assert q.shape == (B,)


def test_critic_min_q_shape(ac_config):
    critic = Critic(ac_config)
    q = critic.min_q(torch.randn(B, D_rl), torch.randn(B, PROP), torch.randn(B, C, ACT))
    assert q.shape == (B,)
