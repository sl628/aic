import numpy as np

from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from transforms3d._gohlketransforms import quaternion_slerp


def minimum_jerk(fraction):
    """Quintic time scaling with zero start/end velocity and acceleration."""
    t = float(np.clip(fraction, 0.0, 1.0))
    return (10.0 * t**3) - (15.0 * t**4) + (6.0 * t**5)


def pose_from_transform(transform: Transform) -> Pose:
    return Pose(
        position=Point(
            x=transform.translation.x,
            y=transform.translation.y,
            z=transform.translation.z,
        ),
        orientation=Quaternion(
            w=transform.rotation.w,
            x=transform.rotation.x,
            y=transform.rotation.y,
            z=transform.rotation.z,
        ),
    )


def interpolate_pose(start: Pose, goal: Pose, fraction) -> Pose:
    progress = float(np.clip(fraction, 0.0, 1.0))
    start_xyz = _pose_xyz(start)
    goal_xyz = _pose_xyz(goal)
    xyz = start_xyz + progress * (goal_xyz - start_xyz)

    quat = quaternion_slerp(_pose_quat(start), _pose_quat(goal), progress)
    return Pose(
        position=Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
        orientation=Quaternion(
            w=float(quat[0]),
            x=float(quat[1]),
            y=float(quat[2]),
            z=float(quat[3]),
        ),
    )


def pose_trajectory(start: Pose, goal: Pose, steps):
    if steps < 1:
        raise ValueError("Trajectory must have at least one step")
    for step in range(1, steps + 1):
        yield interpolate_pose(start, goal, step / steps)


def _pose_xyz(pose: Pose):
    return np.array([pose.position.x, pose.position.y, pose.position.z])


def _pose_quat(pose: Pose):
    quat = np.array(
        [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
    )
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("Pose orientation quaternion has zero norm")
    return tuple(quat / norm)
