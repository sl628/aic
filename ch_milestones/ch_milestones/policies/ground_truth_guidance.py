import numpy as np

from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from rclpy.duration import Duration
from rclpy.time import Time
from time import monotonic
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp
from transforms3d.quaternions import quat2mat


class GroundTruthGuide:
    def __init__(self, node, task, param=None):
        self.node = node
        self.task = task
        self.param = param or self.node_param
        self.tip_error_i = np.zeros(2)
        self.last_gripper_pose_debug = None
        self._last_debug_log_time = None

    def wait_for(self, target_frame: str, source_frame: str, timeout_sec=10.0):
        start = self.node.get_clock().now()
        timeout = Duration(seconds=timeout_sec)
        while (self.node.get_clock().now() - start) < timeout:
            if self.node._tf_buffer.can_transform(target_frame, source_frame, Time()):
                return
            self.node.get_clock().sleep_for(Duration(seconds=0.1))
        raise TimeoutError(f"Missing transform {source_frame} -> {target_frame}")

    def transform(self, target_frame: str, source_frame: str) -> Transform:
        return self.node._tf_buffer.lookup_transform(
            target_frame, source_frame, Time()
        ).transform

    def synchronized_transforms(self, target_frame: str, *source_frames: str):
        latest = [
            self.node._tf_buffer.lookup_transform(target_frame, source, Time())
            for source in source_frames
        ]
        stamp = min(
            (Time.from_msg(transform.header.stamp) for transform in latest),
            key=lambda value: value.nanoseconds,
        )
        return tuple(
            self.node._tf_buffer.lookup_transform(
                target_frame,
                source,
                stamp,
            ).transform
            for source in source_frames
        )

    def gripper_pose(
        self,
        orientation_ref: Transform,
        position_ref: Transform,
        stage: str,
        slerp_fraction=1.0,
        position_fraction=1.0,
        z_offset=0.1,
        reset_xy_integrator=False,
        xy_integral_gain=0.15,
        xy_integrator_limit=0.05,
    ) -> Pose:
        plug, gripper = self.synchronized_transforms(
            "base_link",
            f"{self.task.cable_name}/{self.task.plug_name}_link",
            "gripper/tcp",
        )

        q_gripper = self._quat(gripper)
        q_start = q_gripper
        q_target = self.target_quat(orientation_ref, plug, gripper)
        q_target = self.same_hemisphere(q_target, q_start)
        q_blend = quaternion_slerp(q_start, q_target, slerp_fraction)
        r_gripper = quat2mat(q_gripper)
        r_blend = quat2mat(q_blend)

        desired_plug = self.desired_plug_position(
            orientation_ref, position_ref, z_offset
        )
        plug_xyz = np.array(
            [plug.translation.x, plug.translation.y, plug.translation.z]
        )
        grip_xyz = np.array(
            [gripper.translation.x, gripper.translation.y, gripper.translation.z]
        )
        tcp_to_plug = r_gripper.T @ (plug_xyz - grip_xyz)
        plug_to_tcp_base = grip_xyz - plug_xyz
        tip_error = desired_plug[:2] - plug_xyz[:2]
        if reset_xy_integrator:
            self.reset_integrator()
        elif xy_integral_gain:
            self.tip_error_i = np.clip(
                self.tip_error_i + tip_error,
                -xy_integrator_limit,
                xy_integrator_limit,
            )

        desired_plug = desired_plug.copy()
        desired_plug[:2] += xy_integral_gain * self.tip_error_i

        if self.use_base_offset(stage):
            target = desired_plug + plug_to_tcp_base
            offset_model = "base"
        else:
            target = desired_plug - r_blend @ tcp_to_plug
            offset_model = "rigid"
        start_xyz = grip_xyz
        xyz = position_fraction * target + (1.0 - position_fraction) * start_xyz
        if self.use_base_offset(stage):
            target_plug = xyz - plug_to_tcp_base
        else:
            target_plug = xyz + r_blend @ tcp_to_plug
        q_target_plug = quaternion_multiply(
            q_blend,
            quaternion_multiply(self.quat_inverse(q_gripper), self._quat(plug)),
        )
        self.last_gripper_pose_debug = {
            "goal_plug": self.transform_from_xyz_quat(target_plug, q_target_plug),
            "desired_plug": self.transform_from_xyz_quat(
                desired_plug,
                self._quat(orientation_ref),
            ),
        }
        if self.should_log_debug():
            self.node.get_logger().info(
                f"stage={stage} z={z_offset:.4f} "
                f"xy_error={tip_error[0]:.4f},{tip_error[1]:.4f} "
                f"offset_model={offset_model}"
            )

        return Pose(
            position=Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
            orientation=Quaternion(
                w=float(q_blend[0]),
                x=float(q_blend[1]),
                y=float(q_blend[2]),
                z=float(q_blend[3]),
            ),
        )

    def alignment_gripper_pose(
        self,
        port_transform: Transform,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        reset_xy_integrator: bool = False,
    ) -> Pose:
        """Find the gripper pose that aligns the plug with the port."""
        q_port = (
            port_transform.rotation.w,
            port_transform.rotation.x,
            port_transform.rotation.y,
            port_transform.rotation.z,
        )
        plug = self.transform(
            "base_link", f"{self.task.cable_name}/{self.task.plug_name}_link"
        )
        q_plug = (
            plug.rotation.w,
            plug.rotation.x,
            plug.rotation.y,
            plug.rotation.z,
        )
        q_plug_inv = (
            -q_plug[0],
            q_plug[1],
            q_plug[2],
            q_plug[3],
        )
        q_diff = quaternion_multiply(q_port, q_plug_inv)
        gripper = self.transform("base_link", "gripper/tcp")
        q_gripper = (
            gripper.rotation.w,
            gripper.rotation.x,
            gripper.rotation.y,
            gripper.rotation.z,
        )
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(
            q_gripper, q_gripper_target, slerp_fraction
        )

        gripper_xyz = (
            gripper.translation.x,
            gripper.translation.y,
            gripper.translation.z,
        )
        port_xy = (
            port_transform.translation.x,
            port_transform.translation.y,
        )
        plug_xyz = (
            plug.translation.x,
            plug.translation.y,
            plug.translation.z,
        )
        plug_tip_gripper_offset = (
            gripper_xyz[0] - plug_xyz[0],
            gripper_xyz[1] - plug_xyz[1],
            gripper_xyz[2] - plug_xyz[2],
        )

        tip_x_error = port_xy[0] - plug_xyz[0]
        tip_y_error = port_xy[1] - plug_xyz[1]

        if reset_xy_integrator:
            self.tip_error_i = np.zeros(2)
        else:
            self.tip_error_i[0] = np.clip(
                self.tip_error_i[0] + tip_x_error,
                -0.05,
                0.05,
            )
            self.tip_error_i[1] = np.clip(
                self.tip_error_i[1] + tip_y_error,
                -0.05,
                0.05,
            )

        self.node.get_logger().info(
            f"pfrac: {position_fraction:.3} "
            f"xy_error: {tip_x_error:0.3} {tip_y_error:0.3}   "
            f"integrators: {self.tip_error_i[0]:.3} , {self.tip_error_i[1]:.3}"
        )

        i_gain = 0.15

        target_x = port_xy[0] + i_gain * self.tip_error_i[0]
        target_y = port_xy[1] + i_gain * self.tip_error_i[1]
        target_z = (
            port_transform.translation.z + z_offset - plug_tip_gripper_offset[2]
        )

        blend_xyz = (
            position_fraction * target_x + (1.0 - position_fraction) * gripper_xyz[0],
            position_fraction * target_y + (1.0 - position_fraction) * gripper_xyz[1],
            position_fraction * target_z + (1.0 - position_fraction) * gripper_xyz[2],
        )

        return Pose(
            position=Point(
                x=blend_xyz[0],
                y=blend_xyz[1],
                z=blend_xyz[2],
            ),
            orientation=Quaternion(
                w=q_gripper_slerp[0],
                x=q_gripper_slerp[1],
                y=q_gripper_slerp[2],
                z=q_gripper_slerp[3],
            ),
        )

    def reset_integrator(self):
        self.tip_error_i = np.zeros(2)

    def should_log_debug(self):
        rate_hz = self.param("oracle_debug_log_frequency_hz")
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

    def node_param(self, name):
        return self.node.get_parameter(name).value

    def use_base_offset(self, stage):
        return stage in ("approach", "coarse_align")

    def desired_plug_position(
        self,
        orientation_ref: Transform,
        position_ref: Transform,
        z_offset,
    ):
        del orientation_ref
        return self._xyz(position_ref) + np.array([0.0, 0.0, z_offset])

    def plug_xy_error(
        self,
        orientation_ref: Transform,
        position_ref: Transform,
        z_offset,
    ):
        plug = self.transform(
            "base_link", f"{self.task.cable_name}/{self.task.plug_name}_link"
        )
        desired_plug = self.desired_plug_position(
            orientation_ref, position_ref, z_offset
        )
        return desired_plug[:2] - self._xyz(plug)[:2]

    def target_quat(
        self,
        orientation_ref: Transform,
        plug: Transform,
        gripper: Transform,
    ):
        q_port = self._quat(orientation_ref)
        q_plug = self._quat(plug)
        q_gripper = self._quat(gripper)
        q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
        return quaternion_multiply(
            quaternion_multiply(q_port, q_plug_inv),
            q_gripper,
        )

    def same_hemisphere(self, quat, reference):
        quat = np.array(quat, dtype=float)
        reference = np.array(reference, dtype=float)
        if np.dot(quat, reference) < 0.0:
            quat = -quat
        return tuple(quat)

    def _quat(self, transform: Transform):
        q = transform.rotation
        return (q.w, q.x, q.y, q.z)

    def _xyz(self, transform: Transform):
        t = transform.translation
        return np.array([t.x, t.y, t.z])

    def quat_inverse(self, quat):
        return (quat[0], -quat[1], -quat[2], -quat[3])

    def transform_from_xyz_quat(self, xyz, quat):
        transform = Transform()
        transform.translation.x = float(xyz[0])
        transform.translation.y = float(xyz[1])
        transform.translation.z = float(xyz[2])
        transform.rotation.w = float(quat[0])
        transform.rotation.x = float(quat[1])
        transform.rotation.y = float(quat[2])
        transform.rotation.z = float(quat[3])
        return transform

    def approach_offset(self, orientation_ref, position_ref, source_frame):
        del orientation_ref
        source = self.transform("base_link", source_frame)
        return float(source.translation.z - position_ref.translation.z)
