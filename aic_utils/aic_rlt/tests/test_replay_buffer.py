import inspect
import numpy as np
import torch
from typing import get_type_hints

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


def test_transition_dataclass_type_hints():
    """Test that Transition dataclass has proper type annotations."""
    type_hints = get_type_hints(Transition)
    
    # Check that all fields have type annotations
    expected_fields = {
        'z_rl', 'prop', 'action_chunk', 'ref_action_chunk', 
        'reward', 'next_z_rl', 'next_prop', 'done'
    }
    assert set(type_hints.keys()) == expected_fields
    
    # Check specific type annotations
    assert type_hints['z_rl'] == np.ndarray
    assert type_hints['prop'] == np.ndarray
    assert type_hints['action_chunk'] == np.ndarray
    assert type_hints['ref_action_chunk'] == np.ndarray
    assert type_hints['reward'] == float
    assert type_hints['next_z_rl'] == np.ndarray
    assert type_hints['next_prop'] == np.ndarray
    assert type_hints['done'] == bool


def test_replay_buffer_init_type_hints():
    """Test that ReplayBuffer.__init__ has proper type annotations."""
    type_hints = get_type_hints(ReplayBuffer.__init__)
    
    # Check parameter type annotations
    assert type_hints['capacity'] == int
    assert type_hints['rl_token_dim'] == int
    assert type_hints['prop_dim'] == int
    assert type_hints['action_dim'] == int
    assert type_hints['chunk_length'] == int
    assert type_hints['device'] == torch.device
    # __init__ methods don't typically have explicit return type annotations


def test_replay_buffer_method_type_hints():
    """Test that ReplayBuffer methods have proper type annotations."""
    # Test add method
    add_hints = get_type_hints(ReplayBuffer.add)
    assert add_hints['transition'] == Transition
    assert add_hints['return'] == type(None)
    
    # Test sample method
    sample_hints = get_type_hints(ReplayBuffer.sample)
    assert sample_hints['batch_size'] == int
    # sample method returns Dict[str, torch.Tensor]
    import typing
    expected_return = typing.Dict[str, torch.Tensor]
    assert sample_hints['return'] == expected_return
    
    # Test __len__ method
    len_hints = get_type_hints(ReplayBuffer.__len__)
    assert len_hints['return'] == int
    
    # Test save method
    save_hints = get_type_hints(ReplayBuffer.save)
    assert save_hints['path'] == str
    assert save_hints['return'] == type(None)
    
    # Test load method
    load_hints = get_type_hints(ReplayBuffer.load)
    assert load_hints['path'] == str
    assert load_hints['return'] == type(None)


def test_type_annotated_variables_work_correctly():
    """Test that type-annotated variables in methods work correctly at runtime."""
    buf = ReplayBuffer(10, D_RL, PROP, ACT, C, torch.device("cpu"))
    
    # Test add method with type-annotated variables
    transition = _make_transition(42)
    buf.add(transition)  # This exercises the type-annotated idx variable
    
    assert len(buf) == 1
    
    # Test sample method with type-annotated variables  
    for i in range(5):
        buf.add(_make_transition(i))
    
    batch = buf.sample(3)  # This exercises type-annotated idxs and to_tensor closure
    assert isinstance(batch, dict)
    assert batch['z_rl'].shape[0] == 3
    
    # Test load method with type-annotated variables
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        buf.save(f.name)
        
        buf2 = ReplayBuffer(10, D_RL, PROP, ACT, C, torch.device("cpu"))
        buf2.load(f.name)  # This exercises type-annotated data, n, buffer_attr variables
        
        assert len(buf2) == len(buf)
        
        import os
        os.unlink(f.name)


def test_numpy_array_attributes_type_hinted():
    """Test that numpy array buffer attributes are properly type-hinted."""
    buf = ReplayBuffer(5, D_RL, PROP, ACT, C, torch.device("cpu"))
    
    # These attributes should be type-hinted as np.ndarray and work correctly
    assert isinstance(buf._z_rl, np.ndarray)
    assert isinstance(buf._prop, np.ndarray) 
    assert isinstance(buf._action_chunk, np.ndarray)
    assert isinstance(buf._ref_action_chunk, np.ndarray)
    assert isinstance(buf._reward, np.ndarray)
    assert isinstance(buf._next_z_rl, np.ndarray)
    assert isinstance(buf._next_prop, np.ndarray)
    assert isinstance(buf._done, np.ndarray)
    
    # Verify shapes are correct (validates the type-hinted C, D variables worked)
    assert buf._action_chunk.shape == (5, C, ACT)
    assert buf._ref_action_chunk.shape == (5, C, ACT)
