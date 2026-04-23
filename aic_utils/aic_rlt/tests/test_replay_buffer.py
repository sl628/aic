import numpy as np
import torch

from aic_rlt.replay_buffer import ReplayBuffer, Transition

D_RL, PROP, ACT, C = 64, 26, 9, 4


def _make_transition(seed=0):
    rng = np.random.RandomState(seed)
    return Transition(
        z_rl=rng.randn(D_RL).astype(np.float32),
        prop=rng.randn(PROP).astype(np.float32),
        action_chunk=rng.randn(C, ACT).astype(np.float32),
        ref_action_chunk=rng.randn(C, ACT).astype(np.float32),
        reward=float(rng.randn()),
        next_z_rl=rng.randn(D_RL).astype(np.float32),
        next_prop=rng.randn(PROP).astype(np.float32),
        done=False,
    )


def test_add_and_len():
    buf = ReplayBuffer(100, D_RL, PROP, ACT, C, torch.device("cpu"))
    assert len(buf) == 0
    for i in range(5):
        buf.add(_make_transition(i))
    assert len(buf) == 5


def test_sample_keys_and_shapes():
    buf = ReplayBuffer(100, D_RL, PROP, ACT, C, torch.device("cpu"))
    for i in range(10):
        buf.add(_make_transition(i))
    batch = buf.sample(4)
    assert set(batch.keys()) == {
        "z_rl",
        "prop",
        "action_chunk",
        "ref_action_chunk",
        "reward",
        "next_z_rl",
        "next_prop",
        "done",
    }
    assert batch["z_rl"].shape == (4, D_RL)
    assert batch["action_chunk"].shape == (4, C, ACT)
    assert batch["reward"].shape == (4,)


def test_circular_overwrite():
    cap = 5
    buf = ReplayBuffer(cap, D_RL, PROP, ACT, C, torch.device("cpu"))
    for i in range(12):
        buf.add(_make_transition(i))
    assert len(buf) == cap


def test_save_load_roundtrip(tmp_path):
    buf = ReplayBuffer(100, D_RL, PROP, ACT, C, torch.device("cpu"))
    for i in range(7):
        buf.add(_make_transition(i))

    path = str(tmp_path / "buf.npz")
    buf.save(path)

    buf2 = ReplayBuffer(100, D_RL, PROP, ACT, C, torch.device("cpu"))
    buf2.load(path)
    assert len(buf2) == 7
    np.testing.assert_array_equal(buf._z_rl[:7], buf2._z_rl[:7])
    np.testing.assert_array_equal(buf._reward[:7], buf2._reward[:7])
