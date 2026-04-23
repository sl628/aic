import numpy as np
import pytest
import transforms3d.quaternions as tq

from aic_rlt.vla.xvla_wrapper import (
    _gram_schmidt_r1r2,
    ee6d_to_quat_xyz,
    ee6d_to_xyz_rot6d,
    quat_actions_to_rot6d,
    quat_to_ee6d,
    quat_to_rot6d,
    rot6d_to_quat,
)


def _random_unit_quat(rng):
    q = rng.randn(4).astype(np.float32)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return q


class TestGramSchmidt:
    def test_orthonormal(self):
        r1 = np.array([1.0, 0.5, 0.0], dtype=np.float32)
        r2 = np.array([0.3, 1.0, 0.1], dtype=np.float32)
        r1o, r2o = _gram_schmidt_r1r2(r1, r2)
        np.testing.assert_allclose(np.linalg.norm(r1o), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(r2o), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.dot(r1o, r2o), 0.0, atol=1e-6)

    def test_preserves_r1_direction(self):
        r1 = np.array([3.0, 0.0, 0.0], dtype=np.float32)
        r2 = np.array([0.0, 2.0, 0.0], dtype=np.float32)
        r1o, _ = _gram_schmidt_r1r2(r1, r2)
        np.testing.assert_allclose(r1o, [1.0, 0.0, 0.0], atol=1e-6)


class TestQuatEe6dRoundtrip:
    def test_identity(self):
        xyz = np.array([0.5, -0.1, 0.3], dtype=np.float32)
        quat = np.array([0, 0, 0, 1], dtype=np.float32)
        ee6d = quat_to_ee6d(xyz, quat)
        assert ee6d.shape == (20,)
        xyz_out, quat_out = ee6d_to_quat_xyz(ee6d)
        np.testing.assert_allclose(xyz_out, xyz, atol=1e-5)
        np.testing.assert_allclose(np.abs(np.dot(quat_out, quat)), 1.0, atol=1e-5)

    def test_random_roundtrip(self):
        rng = np.random.RandomState(42)
        for _ in range(10):
            xyz = rng.randn(3).astype(np.float32)
            quat = _random_unit_quat(rng)
            ee6d = quat_to_ee6d(xyz, quat)
            xyz_out, quat_out = ee6d_to_quat_xyz(ee6d)
            np.testing.assert_allclose(xyz_out, xyz, atol=1e-5)
            np.testing.assert_allclose(np.abs(np.dot(quat_out, quat)), 1.0, atol=1e-4)

    def test_ee6d_gripper_zero(self):
        ee6d = quat_to_ee6d(np.zeros(3), np.array([0, 0, 0, 1.0]))
        assert ee6d[9] == 0.0
        np.testing.assert_array_equal(ee6d[10:], 0.0)


class TestRot6dConversions:
    def test_quat_to_rot6d_shape(self):
        quat = np.array([0, 0, 0, 1], dtype=np.float32)
        r6 = quat_to_rot6d(quat)
        assert r6.shape == (6,)

    def test_rot6d_roundtrip(self):
        rng = np.random.RandomState(7)
        for _ in range(10):
            quat = _random_unit_quat(rng)
            r6 = quat_to_rot6d(quat)
            quat_back = rot6d_to_quat(r6)
            np.testing.assert_allclose(np.abs(np.dot(quat_back, quat)), 1.0, atol=1e-5)

    def test_rot6d_columns_orthonormal(self):
        quat = _random_unit_quat(np.random.RandomState(0))
        r6 = quat_to_rot6d(quat)
        r1, r2 = r6[:3], r6[3:]
        np.testing.assert_allclose(np.linalg.norm(r1), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(r2), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.dot(r1, r2), 0.0, atol=1e-5)


class TestEe6dToXyzRot6d:
    def test_shape(self):
        ee6d = quat_to_ee6d(np.zeros(3), np.array([0, 0, 0, 1.0]))
        out = ee6d_to_xyz_rot6d(ee6d)
        assert out.shape == (9,)

    def test_xyz_preserved(self):
        xyz = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ee6d = quat_to_ee6d(xyz, np.array([0, 0, 0, 1.0]))
        out = ee6d_to_xyz_rot6d(ee6d)
        np.testing.assert_allclose(out[:3], xyz, atol=1e-5)

    def test_rot_columns_orthonormal(self):
        rng = np.random.RandomState(3)
        quat = _random_unit_quat(rng)
        ee6d = quat_to_ee6d(rng.randn(3).astype(np.float32), quat)
        out = ee6d_to_xyz_rot6d(ee6d)
        r1, r2 = out[3:6], out[6:9]
        np.testing.assert_allclose(np.dot(r1, r2), 0.0, atol=1e-5)


class TestQuatActionsToRot6d:
    def test_batch_shape(self):
        actions = np.random.randn(5, 7).astype(np.float32)
        actions[:, 3:7] /= np.linalg.norm(actions[:, 3:7], axis=1, keepdims=True)
        out = quat_actions_to_rot6d(actions)
        assert out.shape == (5, 9)

    def test_multidim_batch(self):
        actions = np.random.randn(3, 4, 7).astype(np.float32)
        actions[..., 3:7] /= np.linalg.norm(actions[..., 3:7], axis=-1, keepdims=True)
        out = quat_actions_to_rot6d(actions)
        assert out.shape == (3, 4, 9)

    def test_xyz_preserved(self):
        rng = np.random.RandomState(1)
        actions = rng.randn(3, 7).astype(np.float32)
        actions[:, 3:7] /= np.linalg.norm(actions[:, 3:7], axis=1, keepdims=True)
        out = quat_actions_to_rot6d(actions)
        np.testing.assert_allclose(out[:, :3], actions[:, :3], atol=1e-5)


class TestVLAFactory:
    def test_unknown_backend_raises(self):
        import torch
        from aic_rlt.vla import create_vla_backend

        with pytest.raises(ValueError, match="Unknown VLA backend"):
            create_vla_backend("nonexistent", device=torch.device("cpu"))
