import numpy as np

from aic_rlt.trainer import (
    PHASE_APPROACH,
    PHASE_VERIFY,
    RewardConfig,
    _compute_structured_reward,
    _infer_phases,
    _quat_xyzw_to_mat,
)


def _make_props(T, pos, quat=None, err=None):
    """Build a (T, 26) props array with given TCP pos/quat/err."""
    props = np.zeros((T, 26), dtype=np.float32)
    props[:, 0:3] = pos
    if quat is not None:
        props[:, 3:7] = quat
    else:
        props[:, 6] = 1.0  # identity quat (0,0,0,1)
    if err is not None:
        props[:, 13:16] = err
    return props


def test_quat_identity():
    R = _quat_xyzw_to_mat(np.array([0, 0, 0, 1], dtype=np.float64))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


def test_quat_90_about_z():
    # 90 deg about z: qx=0, qy=0, qz=sin(45)=0.7071, qw=cos(45)=0.7071
    s = np.sqrt(2) / 2
    R = _quat_xyzw_to_mat(np.array([0, 0, s, s], dtype=np.float64))
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    np.testing.assert_allclose(R, expected, atol=1e-10)


def test_infer_phases_far_is_approach():
    goal_pos = np.array([0.5, 0.0, 0.25])
    goal_quat = np.array([0, 0, 0, 1.0])
    rcfg = RewardConfig()
    props = _make_props(5, pos=[1.0, 1.0, 1.0])
    phases = _infer_phases(props, goal_pos, goal_quat, rcfg)
    assert np.all(phases == PHASE_APPROACH)


def test_infer_phases_very_close_is_verify():
    goal_pos = np.array([0.5, 0.0, 0.25])
    goal_quat = np.array([0, 0, 0, 1.0])
    rcfg = RewardConfig()
    props = _make_props(5, pos=goal_pos + 1e-4)
    phases = _infer_phases(props, goal_pos, goal_quat, rcfg)
    assert np.all(phases == PHASE_VERIFY)


def test_reward_high_at_goal():
    goal_pos = np.array([0.5, 0.0, 0.25])
    goal_quat = np.array([0, 0, 0, 1.0])
    rcfg = RewardConfig()
    props = _make_props(1, pos=goal_pos, quat=[0, 0, 0, 1])
    phases = np.array([PHASE_VERIFY])
    r = _compute_structured_reward(props, goal_pos, goal_quat, phases, rcfg)
    assert r[0] > 5.0  # w_pos(1)*1 + w_ori(0.3)*1 + w_success(10)*1 = high


def test_reward_low_far_away():
    goal_pos = np.array([0.5, 0.0, 0.25])
    goal_quat = np.array([0, 0, 0, 1.0])
    rcfg = RewardConfig()
    props = _make_props(1, pos=[5.0, 5.0, 5.0])
    phases = np.array([PHASE_APPROACH])
    r = _compute_structured_reward(props, goal_pos, goal_quat, phases, rcfg)
    assert r[0] < 1.0


def test_phase_bonus_on_transition():
    goal_pos = np.array([0.5, 0.0, 0.25])
    goal_quat = np.array([0, 0, 0, 1.0])
    rcfg = RewardConfig()
    props = _make_props(2, pos=goal_pos)
    phases = np.array([PHASE_APPROACH, PHASE_VERIFY])
    r = _compute_structured_reward(props, goal_pos, goal_quat, phases, rcfg)
    assert r[1] > r[0]  # frame 1 gets phase bonus
