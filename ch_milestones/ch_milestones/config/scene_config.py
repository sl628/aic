from pathlib import Path

from ch_milestones.config.task_options import (
    AUTO_VALUE,
    resolve_cable_type,
    target_board_part,
)


BOARD_PARTS = (
    "lc_mount_rail_0",
    "sfp_mount_rail_0",
    "sc_mount_rail_0",
    "lc_mount_rail_1",
    "sfp_mount_rail_1",
    "sc_mount_rail_1",
    "sc_port_0",
    "sc_port_1",
    "nic_card_mount_0",
    "nic_card_mount_1",
    "nic_card_mount_2",
    "nic_card_mount_3",
    "nic_card_mount_4",
)

BOARD_PART_DEFAULTS = {
    "lc_mount_rail_0": (True, 0.02, 0.0, 0.0, 0.0),
    "sfp_mount_rail_0": (True, 0.03, 0.0, 0.0, 0.0),
    "sc_mount_rail_0": (True, -0.02, 0.0, 0.0, 0.0),
    "lc_mount_rail_1": (True, -0.01, 0.0, 0.0, 0.0),
    "sfp_mount_rail_1": (False, 0.0, 0.0, 0.0, 0.0),
    "sc_mount_rail_1": (False, 0.0, 0.0, 0.0, 0.0),
    "sc_port_0": (True, 0.042, 0.0, 0.0, 0.1),
    "sc_port_1": (False, 0.0, 0.0, 0.0, 0.0),
    "nic_card_mount_0": (True, 0.036, 0.0, 0.0, 0.0),
    "nic_card_mount_1": (False, 0.0, 0.0, 0.0, 0.0),
    "nic_card_mount_2": (False, 0.0, 0.0, 0.0, 0.0),
    "nic_card_mount_3": (False, 0.0, 0.0, 0.0, 0.0),
    "nic_card_mount_4": (False, 0.0, 0.0, 0.0, 0.0),
}

SPAWN_DEFAULTS = {
    "spawn_task_board": True,
    "spawn_cable": True,
}

TASK_BOARD_DEFAULTS = {
    "task_board_name": "task_board",
    "task_board_x": 0.15,
    "task_board_y": -0.2,
    "task_board_z": 1.14,
    "task_board_roll": 0.0,
    "task_board_pitch": 0.0,
    "task_board_yaw": 3.1415,
}

CABLE_POSE_PRESETS = {
    "sfp": {
        "cable_x": 0.172,
        "cable_y": 0.024,
        "cable_z": 1.518,
        "cable_roll": 0.4432,
        "cable_pitch": -0.48,
        "cable_yaw": 1.3303,
    },
    "sc": {
        "cable_x": 0.172,
        "cable_y": 0.024,
        "cable_z": 1.508,
        "cable_roll": 0.4432,
        "cable_pitch": -0.48,
        "cable_yaw": 1.3303,
    },
}

CABLE_DEFAULTS = {
    "auto_cable_pose": True,
    "cable_type": AUTO_VALUE,
    "attach_cable_to_gripper": True,
    **CABLE_POSE_PRESETS["sfp"],
}

CABLE_POSE_FIELDS = (
    "cable_x",
    "cable_y",
    "cable_z",
    "cable_roll",
    "cable_pitch",
    "cable_yaw",
)

HOME_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

HOME_POSITIONS = [-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110]


def declare_scene_parameters(node):
    root = Path.cwd()
    node.declare_parameter("description_root", str(root / "aic_description"))
    node.declare_parameter("assets_root", str(root / "aic_assets"))
    for name, value in SPAWN_DEFAULTS.items():
        node.declare_parameter(name, value)
    for name, value in TASK_BOARD_DEFAULTS.items():
        node.declare_parameter(name, value)
    for name, value in CABLE_DEFAULTS.items():
        node.declare_parameter(name, value)
    node.declare_parameter("home_joint_names", HOME_JOINTS)
    node.declare_parameter("home_joint_positions", HOME_POSITIONS)
    node.declare_parameter("reset_timeout_seconds", 30.0)
    node.declare_parameter("tf_timeout_seconds", 30.0)
    node.declare_parameter("post_spawn_settle_seconds", 1.0)
    for name, values in BOARD_PART_DEFAULTS.items():
        present, translation, roll, pitch, yaw = values
        node.declare_parameter(f"{name}_present", present)
        node.declare_parameter(f"{name}_translation", translation)
        node.declare_parameter(f"{name}_roll", roll)
        node.declare_parameter(f"{name}_pitch", pitch)
        node.declare_parameter(f"{name}_yaw", yaw)


def board_part_mappings(node):
    values = {}
    for name in BOARD_PARTS:
        values[f"{name}_present"] = node.get_parameter(f"{name}_present").value
        values[f"{name}_translation"] = node.get_parameter(f"{name}_translation").value
        values[f"{name}_roll"] = node.get_parameter(f"{name}_roll").value
        values[f"{name}_pitch"] = node.get_parameter(f"{name}_pitch").value
        values[f"{name}_yaw"] = node.get_parameter(f"{name}_yaw").value
    return values


def cable_type_for_task(task, configured_cable_type: str) -> str:
    return resolve_cable_type(task.port_type, configured_cable_type)


def cable_pose_for_task(task, configured_pose, auto_cable_pose: bool):
    if not auto_cable_pose:
        return configured_pose
    preset = CABLE_POSE_PRESETS[task.port_type]
    return tuple(preset[field] for field in CABLE_POSE_FIELDS)


def ensure_target_board_part(task, board_parts: dict[str, bool | float]):
    part = target_board_part(task)
    key = f"{part}_present"
    if key not in board_parts:
        raise ValueError(f"Task target module '{part}' is not a known board part")
    board_parts = dict(board_parts)
    board_parts[key] = True
    return board_parts
