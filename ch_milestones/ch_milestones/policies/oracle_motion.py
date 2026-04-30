from math import acos, ceil, sqrt
from time import monotonic

from geometry_msgs.msg import Transform
import numpy as np
from transforms3d.quaternions import quat2mat

from ch_milestones.policies.cartesian_trajectory import (
    interpolate_pose,
    pose_from_transform,
    pose_trajectory,
)


class OracleMotionCommander:
    def __init__(self, policy):
        self.policy = policy
        self._plug_step_stage = None
        self._last_target_plug = None
        self._last_debug_log_time = None

    def command_pose(self, pose, target_plug=None, log_debug=None):
        current, plug = self.current_tcp_and_plug()
        if target_plug is None:
            target_plug = self.policy.debug_frames.predicted_child_transform(
                current,
                plug,
                pose,
            )
        if log_debug is None:
            log_debug = self.should_log_debug()
        if log_debug:
            self.policy.get_logger().info(
                f"stage={self.policy.stage} "
                f"current_tcp_p={self.fmt_xyz(current.translation)} "
                f"current_tcp_q={self.fmt_quat(current.rotation)} "
                f"target_tcp_p={self.fmt_xyz(pose.position)} "
                f"target_tcp_q={self.fmt_quat(pose.orientation)}"
            )
            self.log_plug_error(plug, target_plug)
        self.policy.debug_frames.publish_tcp_frames(current, pose, plug, target_plug)
        self.pause_before_motion(pose, current, target_plug)
        self.policy.set_pose_target(
            self.policy.move_robot,
            pose,
            stiffness=self.policy.param("oracle_cartesian_stiffness"),
            damping=self.policy.param("oracle_cartesian_damping"),
        )
        self.policy.sleep_for(self.policy.command_period())

    def pause_before_motion(self, target_pose, current, target_plug=None):
        pause_seconds = self.policy.param("oracle_debug_pause_before_motion_seconds")
        if pause_seconds > 0.0:
            self.policy.get_logger().info(
                f"stage={self.policy.stage} debug pause before motion "
                f"for {pause_seconds:.2f}s"
            )
            self.hold_latched_pose(
                pose_from_transform(current),
                target_pose,
                pause_seconds,
                target_plug,
            )

        if not self.policy.param("oracle_debug_pause_motion"):
            return

        self.policy.get_logger().warning(
            "Oracle debug motion pause is active. "
            "Resume with: ros2 param set /aic_model "
            f"{self.policy.parameter_name('oracle_debug_pause_motion')} false"
        )
        hold_pose = pose_from_transform(current)
        while self.policy.param("oracle_debug_pause_motion"):
            self.hold_latched_pose(
                hold_pose,
                target_pose,
                self.policy.param("oracle_debug_pause_poll_seconds"),
                target_plug,
            )

    def hold_latched_pose(self, hold_pose, target_pose, duration, target_plug=None):
        current, plug = self.current_tcp_and_plug()
        if target_plug is None:
            target_plug = self.policy.debug_frames.predicted_child_transform(
                current,
                plug,
                target_pose,
            )
        self.policy.debug_frames.publish_tcp_frames(
            current,
            target_pose,
            plug,
            target_plug,
        )
        self.policy.set_pose_target(
            self.policy.move_robot,
            hold_pose,
            stiffness=self.policy.param("oracle_cartesian_stiffness"),
            damping=self.policy.param("oracle_cartesian_damping"),
        )
        self.policy.sleep_for(duration)

    def execute_pose_trajectory(self, goal, steps):
        start = pose_from_transform(
            self.policy.guide.transform("base_link", "gripper/tcp")
        )
        self.policy.get_logger().info(
            f"stage={self.policy.stage} planned_cartesian_trajectory steps={steps} "
            f"start_tcp_p={self.fmt_xyz(start.position)} "
            f"goal_tcp_p={self.fmt_xyz(goal.position)}"
        )
        for pose in pose_trajectory(start, goal, steps):
            self.command_pose(pose)

    def execute_live_step_trajectory(
        self,
        goal_fn,
        max_translation_step,
        success_tolerance,
        timeout_sec,
        rotation_success_tolerance=None,
    ):
        if max_translation_step <= 0.0:
            raise ValueError("max_translation_step must be positive")
        if success_tolerance <= 0.0:
            raise ValueError("success_tolerance must be positive")
        if (
            rotation_success_tolerance is not None
            and rotation_success_tolerance <= 0.0
        ):
            raise ValueError("rotation_success_tolerance must be positive")
        if timeout_sec <= 0.0:
            raise ValueError("timeout_sec must be positive")

        rotation_field = ""
        if rotation_success_tolerance is not None:
            rotation_field = (
                f" rotation_success_tolerance={rotation_success_tolerance:.4f}"
            )
        start = monotonic()
        attempts = 0
        self.policy.get_logger().info(
            f"stage={self.policy.stage} live_step_trajectory "
            f"timeout={timeout_sec:.1f}s "
            f"max_translation_step={max_translation_step:.4f} "
            f"success_tolerance={success_tolerance:.4f}"
            f"{rotation_field}"
        )
        while monotonic() - start < timeout_sec:
            attempts += 1
            if self.command_step_toward(
                goal_fn(),
                max_translation_step,
                success_tolerance,
                rotation_success_tolerance,
            ):
                self.policy.get_logger().info(
                    f"stage={self.policy.stage} reached target "
                    f"attempts={attempts}"
                )
                return True
        raise TimeoutError(
            f"{self.policy.stage} did not reach target within "
            f"{timeout_sec:.1f}s wall time after {attempts} attempts"
        )

    def command_step_toward(
        self,
        goal,
        max_translation_step,
        success_tolerance,
        rotation_success_tolerance=None,
    ):
        log_debug = self.should_log_debug()
        current_tcp, current_plug = self.current_tcp_and_plug()
        debug = self.policy.guide.last_gripper_pose_debug
        if debug is None:
            current = pose_from_transform(current_tcp)
            distance = self.translation_distance(current, goal)
            fraction = 1.0
            if distance > max_translation_step:
                fraction = max_translation_step / distance
            self.command_pose(
                interpolate_pose(current, goal, fraction),
                log_debug=log_debug,
            )
            rotation_error = None
            if rotation_success_tolerance is not None:
                rotation_error = self.rotation_distance(
                    current.orientation,
                    goal.orientation,
                )
            return self.reached_success(
                distance,
                success_tolerance,
                rotation_error,
                rotation_success_tolerance,
            )

        goal_plug = debug["goal_plug"]
        distance = self.transform_distance(current_plug, goal_plug)
        rotation_error = self.rotation_distance(
            current_plug.rotation,
            goal_plug.rotation,
        )
        fraction = 1.0
        if distance > max_translation_step:
            fraction = max_translation_step / distance
        rotation_field = ""
        if rotation_success_tolerance is not None:
            rotation_field = (
                f"rotation_error={rotation_error:.4f} "
                f"rotation_success_tolerance={rotation_success_tolerance:.4f} "
            )
        target_plug = self.next_plug_step_target(
            current_plug,
            goal_plug,
            max_translation_step,
        )
        if log_debug:
            self.policy.get_logger().info(
                f"stage={self.policy.stage} plug_space_step "
                f"distance={distance:.4f} fraction={fraction:.4f} "
                f"success_tolerance={success_tolerance:.4f} "
                f"{rotation_field}"
                f"command_target_to_goal={self.transform_distance(target_plug, goal_plug):.4f} "
                f"offset_model={self.offset_model()}"
            )
        target_pose, target_plug = self.plug_space_step_pose(
            current_tcp,
            current_plug,
            goal,
            target_plug,
            fraction,
        )
        self.command_pose(target_pose, target_plug, log_debug=log_debug)
        return self.reached_success(
            distance,
            success_tolerance,
            rotation_error,
            rotation_success_tolerance,
        )

    def reached_success(
        self,
        distance,
        success_tolerance,
        rotation_error,
        rotation_success_tolerance,
    ):
        translation_reached = distance <= success_tolerance
        if rotation_success_tolerance is None:
            return translation_reached
        return translation_reached and rotation_error <= rotation_success_tolerance

    def hold_pose(self, pose, duration):
        for _ in range(ceil(duration / self.policy.command_period())):
            self.command_pose(pose)

    def hold_live_pose(self, pose_fn, duration):
        for _ in range(ceil(duration / self.policy.command_period())):
            self.command_pose(pose_fn())

    def fmt_xyz(self, xyz):
        return f"({xyz.x:.4f},{xyz.y:.4f},{xyz.z:.4f})"

    def fmt_quat(self, quat):
        return f"({quat.w:.4f},{quat.x:.4f},{quat.y:.4f},{quat.z:.4f})"

    def should_log_debug(self):
        rate_hz = self.policy.param("oracle_debug_log_frequency_hz")
        if rate_hz <= 0.0:
            return False
        now = monotonic()
        if (
            self._last_debug_log_time is None
            or now - self._last_debug_log_time >= 1.0 / rate_hz
        ):
            self._last_debug_log_time = now
            return True
        return False

    def current_tcp_and_plug(self):
        return self.policy.guide.synchronized_transforms(
            "base_link",
            "gripper/tcp",
            self.policy.frames.plug_frame,
        )

    def offset_model(self):
        if self.policy.stage in ("approach", "coarse_align"):
            return "base"
        return "rigid"

    def next_plug_step_target(self, current_plug, goal_plug, max_translation_step):
        if (
            self._plug_step_stage != self.policy.stage
            or self._last_target_plug is None
        ):
            start_plug = current_plug
        else:
            start_plug = self._last_target_plug

        start_xyz = self.xyz(start_plug.translation)
        goal_xyz = self.xyz(goal_plug.translation)
        delta = goal_xyz - start_xyz
        distance = float(np.linalg.norm(delta))
        if distance <= max_translation_step:
            target_xyz = goal_xyz
        else:
            target_xyz = start_xyz + (max_translation_step / distance) * delta

        target_plug = self.transform_from_xyz_rotation(
            target_xyz,
            goal_plug.rotation,
        )
        self._plug_step_stage = self.policy.stage
        self._last_target_plug = target_plug
        return target_plug

    def plug_space_step_pose(
        self,
        current_tcp,
        current_plug,
        goal_tcp,
        target_plug,
        fraction,
    ):
        current_pose = pose_from_transform(current_tcp)
        target_pose = interpolate_pose(current_pose, goal_tcp, fraction)

        current_tcp_xyz = self.xyz(current_tcp.translation)
        current_plug_xyz = self.xyz(current_plug.translation)
        target_plug_xyz = self.xyz(target_plug.translation)

        if self.offset_model() == "base":
            plug_to_tcp_base = current_tcp_xyz - current_plug_xyz
            target_tcp_xyz = target_plug_xyz + plug_to_tcp_base
        else:
            tcp_to_plug = self.rotation_matrix(current_tcp.rotation).T @ (
                current_plug_xyz - current_tcp_xyz
            )
            target_tcp_xyz = target_plug_xyz - (
                self.rotation_matrix(target_pose.orientation) @ tcp_to_plug
            )
        target_pose.position.x = float(target_tcp_xyz[0])
        target_pose.position.y = float(target_tcp_xyz[1])
        target_pose.position.z = float(target_tcp_xyz[2])
        return target_pose, target_plug

    def log_plug_error(self, current_plug, target_plug):
        fields = [f"stage={self.policy.stage}"]

        reference = self.policy.debug_frames.reference_transform()
        if reference is not None:
            current_error = self.transform_distance(current_plug, reference)
            target_error = self.transform_distance(target_plug, reference)
            fields += [
                f"current_plug_error={current_error:.4f}",
                f"target_plug_error={target_error:.4f}",
                f"delta={current_error - target_error:.4f}",
            ]

        goal_plug = self.policy.debug_frames.last_goal_plug
        if goal_plug is not None:
            target_goal_distance = self.transform_distance(target_plug, goal_plug)
            fields.append(
                f"target_plug_to_goal_plug_distance={target_goal_distance:.4f}"
            )

        if len(fields) > 1:
            self.policy.get_logger().info(" ".join(fields))

    def translation_distance(self, start, goal):
        dx = goal.position.x - start.position.x
        dy = goal.position.y - start.position.y
        dz = goal.position.z - start.position.z
        return sqrt(dx * dx + dy * dy + dz * dz)

    def transform_distance(self, start, goal):
        dx = goal.translation.x - start.translation.x
        dy = goal.translation.y - start.translation.y
        dz = goal.translation.z - start.translation.z
        return sqrt(dx * dx + dy * dy + dz * dz)

    def rotation_distance(self, start, goal):
        dot = (
            start.w * goal.w
            + start.x * goal.x
            + start.y * goal.y
            + start.z * goal.z
        )
        dot = max(-1.0, min(1.0, abs(dot)))
        return 2.0 * acos(dot)

    def xyz(self, translation):
        return np.array([translation.x, translation.y, translation.z], dtype=float)

    def rotation_matrix(self, quaternion):
        return quat2mat((quaternion.w, quaternion.x, quaternion.y, quaternion.z))

    def transform_from_xyz_rotation(self, xyz, rotation):
        transform = Transform()
        transform.translation.x = float(xyz[0])
        transform.translation.y = float(xyz[1])
        transform.translation.z = float(xyz[2])
        transform.rotation.w = rotation.w
        transform.rotation.x = rotation.x
        transform.rotation.y = rotation.y
        transform.rotation.z = rotation.z
        return transform
