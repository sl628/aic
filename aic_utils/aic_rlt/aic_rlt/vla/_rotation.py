"""Pure-numpy rotation conversions used by aic's action pipeline.

Extracted from xvla_wrapper.py so that non-xvla code paths (notably pi05, which
runs in openpi's venv where the xvla lerobot policy isn't installed) can import
these without triggering xvla's heavy dependency chain.

All functions are stateless, numpy-only, and rely on transforms3d.quaternions.
"""

import numpy as np


def _quat_wxyz_to_mat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Pure-numpy quaternion → 3×3 rotation matrix.

    Avoids a transforms3d dependency (which isn't installed in openpi's venv).
    Matches transforms3d.quaternions.quat2mat([qw, qx, qy, qz]).
    """
    n = qw * qw + qx * qx + qy * qy + qz * qz
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = s * qw * qx, s * qw * qy, s * qw * qz
    xx, xy, xz = s * qx * qx, s * qx * qy, s * qx * qz
    yy, yz, zz = s * qy * qy, s * qy * qz, s * qz * qz
    R = np.array([
        [1.0 - (yy + zz), xy - wz,          xz + wy],
        [xy + wz,         1.0 - (xx + zz),  yz - wx],
        [xz - wy,         yz + wx,          1.0 - (xx + yy)],
    ], dtype=np.float64)
    return R


def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Pure-numpy 3×3 rotation matrix → quaternion [qw, qx, qy, qz] (Shepperd's method)."""
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return np.array([qw, qx, qy, qz], dtype=np.float64)


def _gram_schmidt_r1r2(r1: np.ndarray, r2: np.ndarray):
    """Orthonormalize two column vectors via Gram-Schmidt."""
    r1 = r1.astype(np.float64)
    r2 = r2.astype(np.float64)
    r1 = r1 / (np.linalg.norm(r1) + 1e-8)
    r2 = r2 - np.dot(r2, r1) * r1
    r2 = r2 / (np.linalg.norm(r2) + 1e-8)
    return r1.astype(np.float32), r2.astype(np.float32)


def quat_to_ee6d(xyz: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Convert position + quaternion to XVLA ee6d format (20-dim, zero-padded).

    Args:
        xyz:  (3,) float32 — TCP position
        quat: (4,) float32 — quaternion [qx, qy, qz, qw]

    Returns:
        (20,) float32 — [x, y, z, r1(3), r2(3), gripper=0, zeros×10]
    """
    qw, qx, qy, qz = float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])
    R = _quat_wxyz_to_mat(qw, qx, qy, qz)
    r1 = R[:, 0].astype(np.float32)
    r2 = R[:, 1].astype(np.float32)
    ee6d = np.zeros(20, dtype=np.float32)
    ee6d[0:3] = xyz.astype(np.float32)
    ee6d[3:6] = r1
    ee6d[6:9] = r2
    ee6d[9] = 0.0
    return ee6d


def ee6d_to_quat_xyz(ee6d: np.ndarray) -> tuple:
    """Convert XVLA ee6d (20-dim) to position + quaternion [qx, qy, qz, qw]."""
    xyz = ee6d[0:3].astype(np.float32)
    r1, r2 = _gram_schmidt_r1r2(ee6d[3:6], ee6d[6:9])
    r3 = np.cross(r1.astype(np.float64), r2.astype(np.float64))
    R = np.stack([r1.astype(np.float64), r2.astype(np.float64), r3], axis=1)
    q_wxyz = _mat_to_quat_wxyz(R)
    quat = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)
    return xyz, quat


def ee6d_to_xyz_rot6d(ee6d: np.ndarray) -> np.ndarray:
    """Extract [xyz, r1, r2] from XVLA ee6d (20D) → 9D action."""
    xyz = ee6d[0:3].astype(np.float32)
    r1, r2 = _gram_schmidt_r1r2(ee6d[3:6], ee6d[6:9])
    return np.concatenate([xyz, r1, r2])


def rot6d_to_quat(rot6d: np.ndarray) -> np.ndarray:
    """6D rotation [r1(3), r2(3)] → quaternion [qx, qy, qz, qw]."""
    r1, r2 = _gram_schmidt_r1r2(rot6d[0:3], rot6d[3:6])
    r3 = np.cross(r1, r2).astype(np.float64)
    R = np.stack([r1.astype(np.float64), r2.astype(np.float64), r3], axis=1)
    q_wxyz = _mat_to_quat_wxyz(R)
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)


def quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Quaternion [qx, qy, qz, qw] → 6D rotation [r1(3), r2(3)]."""
    qw, qx, qy, qz = float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])
    R = _quat_wxyz_to_mat(qw, qx, qy, qz)
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def quat_actions_to_rot6d(actions: np.ndarray) -> np.ndarray:
    """Batch convert actions from 7D [xyz, quat] → 9D [xyz, r1, r2].

    Args:
        actions: (..., 7)
    Returns:
        (..., 9)
    """
    shape = actions.shape[:-1]
    flat = actions.reshape(-1, 7)
    out = np.zeros((flat.shape[0], 9), dtype=np.float32)
    out[:, 0:3] = flat[:, 0:3]
    for i in range(flat.shape[0]):
        out[i, 3:9] = quat_to_rot6d(flat[i, 3:7])
    return out.reshape(*shape, 9)
