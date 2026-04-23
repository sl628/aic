import importlib.util
from pathlib import Path

import numpy as np

_mod_path = Path(__file__).resolve().parents[2] / "sym_data" / "generate_synthetic.py"
_spec = importlib.util.spec_from_file_location("generate_synthetic", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_norm_q = _mod._norm_q
_slerp = _mod._slerp
_ease = _mod._ease


def test_norm_q_unit():
    q = np.array([1.0, 2.0, 3.0, 4.0])
    out = _norm_q(q)
    np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-7)


def test_norm_q_zero():
    q = np.array([0.0, 0.0, 0.0, 0.0])
    out = _norm_q(q)
    np.testing.assert_array_equal(out, q)


def test_slerp_endpoints():
    q0 = _norm_q(np.array([1.0, 0.0, 0.0, 1.0]))
    q1 = _norm_q(np.array([0.0, 1.0, 0.0, 1.0]))
    np.testing.assert_allclose(_slerp(q0, q1, 0.0), q0, atol=1e-6)
    np.testing.assert_allclose(_slerp(q0, q1, 1.0), q1, atol=1e-6)


def test_slerp_midpoint_unit():
    q0 = _norm_q(np.array([1.0, 0.0, 0.0, 0.0]))
    q1 = _norm_q(np.array([0.0, 1.0, 0.0, 0.0]))
    mid = _slerp(q0, q1, 0.5)
    np.testing.assert_allclose(np.linalg.norm(mid), 1.0, atol=1e-7)


def test_ease_boundaries():
    assert _ease(0.0) == 0.0
    assert _ease(1.0) == 1.0


def test_ease_monotonic():
    ts = np.linspace(0, 1, 50)
    vals = [_ease(float(t)) for t in ts]
    for i in range(1, len(vals)):
        assert vals[i] >= vals[i - 1]
