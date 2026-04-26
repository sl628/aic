import numpy as np
from PIL import Image

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode


STATE_NAMES = [
    "tcp_pose.position.x",
    "tcp_pose.position.y",
    "tcp_pose.position.z",
    "tcp_pose.orientation.x",
    "tcp_pose.orientation.y",
    "tcp_pose.orientation.z",
    "tcp_pose.orientation.w",
    "tcp_velocity.linear.x",
    "tcp_velocity.linear.y",
    "tcp_velocity.linear.z",
    "tcp_velocity.angular.x",
    "tcp_velocity.angular.y",
    "tcp_velocity.angular.z",
    "tcp_error.x",
    "tcp_error.y",
    "tcp_error.z",
    "tcp_error.rx",
    "tcp_error.ry",
    "tcp_error.rz",
    "joint_positions.0",
    "joint_positions.1",
    "joint_positions.2",
    "joint_positions.3",
    "joint_positions.4",
    "joint_positions.5",
    "joint_positions.6",
]

ACTION_NAMES = [
    "target_tcp_pose.position.x",
    "target_tcp_pose.position.y",
    "target_tcp_pose.position.z",
    "target_tcp_pose.orientation.x",
    "target_tcp_pose.orientation.y",
    "target_tcp_pose.orientation.z",
    "target_tcp_pose.orientation.w",
]

IMAGE_FIELDS = {
    "left_camera": "left_image",
    "center_camera": "center_image",
    "right_camera": "right_image",
}


def features(image_shape, use_videos=True):
    dtype = "video" if use_videos else "image"
    image_features = {
        f"observation.images.{name}": {
            "dtype": dtype,
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        }
        for name in IMAGE_FIELDS
    }
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": ACTION_NAMES,
        },
        **image_features,
    }


def frame_from_ros(observation, action: MotionUpdate, image_shape):
    return {
        "observation.state": state_vector(observation),
        "action": action_vector(action),
        **images(observation, image_shape),
    }


def state_vector(observation):
    state = observation.controller_state
    pose = state.tcp_pose
    velocity = state.tcp_velocity
    return np.array(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
            velocity.linear.x,
            velocity.linear.y,
            velocity.linear.z,
            velocity.angular.x,
            velocity.angular.y,
            velocity.angular.z,
            *state.tcp_error,
            *observation.joint_states.position[:7],
        ],
        dtype=np.float32,
    )


def action_vector(action: MotionUpdate):
    if action.trajectory_generation_mode.mode != TrajectoryGenerationMode.MODE_POSITION:
        raise ValueError("Expected Cartesian position MotionUpdate from OraclePolicy")
    pose = action.pose
    return np.array(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=np.float32,
    )


def images(observation, image_shape):
    return {
        f"observation.images.{name}": image_from_msg(
            getattr(observation, field), image_shape
        )
        for name, field in IMAGE_FIELDS.items()
    }


def image_from_msg(msg, image_shape):
    target_h, target_w, _ = image_shape
    image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    return np.asarray(Image.fromarray(image).resize((target_w, target_h)))
