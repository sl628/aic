import numpy as np

from geometry_msgs.msg import Pose, Transform, TransformStamped
from tf2_ros import TransformBroadcaster
from transforms3d._gohlketransforms import quaternion_multiply
from transforms3d.quaternions import quat2mat


class OracleDebugFrames:
    def __init__(self, node, param=None):
        self.node = node
        self.param = param or self.node_param
        self.broadcaster = TransformBroadcaster(node)
        self.announced = False
        self.last_reference = None
        self.last_goal_plug = None

    def publish_tcp_frames(
        self,
        current_tcp: Transform,
        target_tcp: Pose,
        current_plug: Transform | None = None,
        target_plug: Transform | None = None,
    ):
        if not self.enabled():
            return

        stamp = self.node.get_clock().now().to_msg()
        prefix = self.prefix()
        transforms = [
            self.from_transform(
                stamp,
                "base_link",
                f"{prefix}/moving_frame",
                current_tcp,
            ),
            self.from_pose(
                stamp,
                "base_link",
                f"{prefix}/target_frame",
                target_tcp,
            ),
        ]
        if current_plug is not None:
            if target_plug is None:
                target_plug = self.predicted_child_transform(
                    current_tcp,
                    current_plug,
                    target_tcp,
                )
            transforms += [
                self.from_transform(
                    stamp,
                    "base_link",
                    f"{prefix}/moving_plug_frame",
                    current_plug,
                ),
                self.from_transform(
                    stamp,
                    "base_link",
                    f"{prefix}/target_plug_frame",
                    target_plug,
                ),
            ]
        reference = self.last_reference_transform(stamp, prefix)
        if reference is not None:
            transforms.append(reference)
        goal_plug = self.last_goal_plug_transform(stamp, prefix)
        if goal_plug is not None:
            transforms.append(goal_plug)
        self.broadcaster.sendTransform(transforms)

        self.announce(prefix)

    def publish_reference_frame(
        self,
        orientation_ref: Transform,
        position_ref: Transform,
        z_offset: float,
    ):
        if not self.enabled():
            return

        self.last_reference = (orientation_ref, position_ref, z_offset)
        stamp = self.node.get_clock().now().to_msg()
        prefix = self.prefix()
        self.broadcaster.sendTransform(
            self.last_reference_transform(stamp, prefix)
        )
        self.announce(prefix)

    def publish_goal_plug_frame(self, goal_plug: Transform):
        if not self.enabled():
            return

        self.last_goal_plug = goal_plug
        stamp = self.node.get_clock().now().to_msg()
        prefix = self.prefix()
        self.broadcaster.sendTransform(
            self.from_transform(
                stamp,
                "base_link",
                f"{prefix}/goal_plug_frame",
                goal_plug,
            )
        )
        self.announce(prefix)

    def announce(self, prefix):
        if self.announced:
            return
        self.node.get_logger().info(
            "Publishing oracle debug TF frames: "
            f"{prefix}/moving_frame and {prefix}/target_frame for TCP, "
            f"{prefix}/moving_plug_frame and {prefix}/target_plug_frame "
            f"for the immediate plug command, {prefix}/goal_plug_frame "
            f"for the stage plug goal, and {prefix}/reference_frame"
        )
        self.announced = True

    def enabled(self):
        return bool(self.param("oracle_publish_debug_frames"))

    def prefix(self):
        prefix = str(self.param("oracle_debug_frame_prefix"))
        return prefix.strip("/") or "oracle_debug"

    def node_param(self, name):
        return self.node.get_parameter(name).value

    def from_transform(self, stamp, parent_frame, child_frame, transform):
        stamped = TransformStamped()
        stamped.header.stamp = stamp
        stamped.header.frame_id = parent_frame
        stamped.child_frame_id = child_frame
        stamped.transform = transform
        return stamped

    def from_pose(self, stamp, parent_frame, child_frame, pose):
        stamped = TransformStamped()
        stamped.header.stamp = stamp
        stamped.header.frame_id = parent_frame
        stamped.child_frame_id = child_frame
        stamped.transform.translation.x = pose.position.x
        stamped.transform.translation.y = pose.position.y
        stamped.transform.translation.z = pose.position.z
        stamped.transform.rotation = pose.orientation
        return stamped

    def predicted_child_transform(
        self,
        current_parent: Transform,
        current_child: Transform,
        target_parent: Pose,
    ):
        parent_xyz = self.transform_xyz(current_parent)
        child_xyz = self.transform_xyz(current_child)
        target_parent_xyz = self.pose_xyz(target_parent)

        q_current_parent = self.quat_from_transform(current_parent)
        q_current_child = self.quat_from_transform(current_child)
        q_target_parent = self.quat_from_pose(target_parent)

        parent_to_child_xyz = quat2mat(q_current_parent).T @ (child_xyz - parent_xyz)
        target_child_xyz = target_parent_xyz + quat2mat(q_target_parent) @ parent_to_child_xyz

        parent_to_child_q = quaternion_multiply(
            self.quat_inverse(q_current_parent),
            q_current_child,
        )
        target_child_q = self.normalize_quat(
            quaternion_multiply(q_target_parent, parent_to_child_q)
        )

        transform = Transform()
        transform.translation.x = float(target_child_xyz[0])
        transform.translation.y = float(target_child_xyz[1])
        transform.translation.z = float(target_child_xyz[2])
        transform.rotation.w = float(target_child_q[0])
        transform.rotation.x = float(target_child_q[1])
        transform.rotation.y = float(target_child_q[2])
        transform.rotation.z = float(target_child_q[3])
        return transform

    def transform_xyz(self, transform):
        t = transform.translation
        return np.array([t.x, t.y, t.z], dtype=float)

    def pose_xyz(self, pose):
        p = pose.position
        return np.array([p.x, p.y, p.z], dtype=float)

    def quat_from_transform(self, transform):
        q = transform.rotation
        return self.normalize_quat((q.w, q.x, q.y, q.z))

    def quat_from_pose(self, pose):
        q = pose.orientation
        return self.normalize_quat((q.w, q.x, q.y, q.z))

    def quat_inverse(self, quat):
        return (quat[0], -quat[1], -quat[2], -quat[3])

    def normalize_quat(self, quat):
        values = np.array(quat, dtype=float)
        norm = np.linalg.norm(values)
        if norm == 0.0:
            raise ValueError("Debug frame quaternion has zero norm")
        return tuple(values / norm)

    def last_reference_transform(self, stamp, prefix):
        reference = self.reference_transform()
        if reference is None:
            return None
        return self.from_transform(
            stamp,
            "base_link",
            f"{prefix}/reference_frame",
            reference,
        )

    def reference_transform(self):
        if self.last_reference is None:
            return None
        orientation_ref, position_ref, z_offset = self.last_reference
        transform = Transform()
        transform.translation.x = position_ref.translation.x
        transform.translation.y = position_ref.translation.y
        transform.translation.z = position_ref.translation.z + z_offset
        transform.rotation = orientation_ref.rotation
        return transform

    def last_goal_plug_transform(self, stamp, prefix):
        if self.last_goal_plug is None:
            return None
        return self.from_transform(
            stamp,
            "base_link",
            f"{prefix}/goal_plug_frame",
            self.last_goal_plug,
        )

    def from_reference(
        self,
        stamp,
        parent_frame,
        child_frame,
        orientation_ref,
        position_ref,
        z_offset,
    ):
        return self.from_transform(
            stamp,
            parent_frame,
            child_frame,
            self.reference_from_parts(
                orientation_ref,
                position_ref,
                z_offset,
            ),
        )

    def reference_from_parts(self, orientation_ref, position_ref, z_offset):
        transform = Transform()
        transform.translation.x = position_ref.translation.x
        transform.translation.y = position_ref.translation.y
        transform.translation.z = position_ref.translation.z + z_offset
        transform.rotation = orientation_ref.rotation
        return transform
