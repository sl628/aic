from math import cos, sin

from geometry_msgs.msg import PoseStamped
from simulation_interfaces.msg import Result
from simulation_interfaces.srv import DeleteEntity, SpawnEntity

from ch_milestones.environment.services import call


class EntitySpawner:
    def __init__(self, node, callback_group=None):
        self.node = node
        self.spawn = node.create_client(
            SpawnEntity, "/gz_server/spawn_entity", callback_group=callback_group
        )
        self.delete = node.create_client(
            DeleteEntity, "/gz_server/delete_entity", callback_group=callback_group
        )

    def spawn_entity(self, name, resource_string, pose, timeout_sec):
        request = SpawnEntity.Request()
        request.name = name
        request.allow_renaming = False
        request.uri = ""
        request.resource_string = resource_string
        request.entity_namespace = ""
        request.initial_pose = pose_stamped(pose)
        response = call(self.node, self.spawn, request, timeout_sec, "spawn entity")
        if response.result.result != Result.RESULT_OK:
            raise RuntimeError(response.result.error_message)
        return response.entity_name

    def delete_entity(self, name, timeout_sec):
        request = DeleteEntity.Request()
        request.entity = name
        response = call(self.node, self.delete, request, timeout_sec, "delete entity")
        if response.result.result != Result.RESULT_OK:
            raise RuntimeError(response.result.error_message)


def pose_stamped(pose):
    x, y, z, roll, pitch, yaw = pose
    msg = PoseStamped()
    msg.header.frame_id = "world"
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    qx, qy, qz, qw = quaternion_from_rpy(roll, pitch, yaw)
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    msg.pose.orientation.w = qw
    return msg


def quaternion_from_rpy(roll, pitch, yaw):
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )
