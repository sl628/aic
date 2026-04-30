from ch_milestones.config.scene_config import (
    BOARD_PART_DEFAULTS,
    CABLE_DEFAULTS,
    CABLE_POSE_PRESETS,
    TASK_BOARD_DEFAULTS,
)


SFP_TASK_BOARD_JITTER = {
    "task_board_x": 0.05,
    "task_board_y": 0.05,
    "task_board_z": 0.0,
    "task_board_roll": 0.0,
    "task_board_pitch": 0.0,
    "task_board_yaw": 1.57 / 1.6,
}

SFP_CABLE_JITTER = {
    "cable_x": 0.0,
    "cable_y": 0.0,
    "cable_z": 0.0,
    "cable_roll": 0.0,
    "cable_pitch": 0.0,
    "cable_yaw": 0.0,
}

SC_TASK_BOARD_JITTER = {
    "task_board_x": 0.05,
    "task_board_y": 0.05,
    "task_board_z": 0.0,
    "task_board_roll": 0.0,
    "task_board_pitch": 0.0,
    "task_board_yaw": 1.57 / 1.6,
}

SC_CABLE_JITTER = {
    "cable_x": 0.0,
    "cable_y": 0.0,
    "cable_z": 0.0,
    "cable_roll": 0.0,
    "cable_pitch": 0.0,
    "cable_yaw": 0.0,
}

SC_PORT_PREFIX = "sc_port_"
SC_TASK_BOARD_RANDOMIZATION_PREFIX = "sc_task_board"
SC_CABLE_RANDOMIZATION_PREFIX = "sc_cable"

PART_TRANSLATION_RANGES = {
    "mount_rail": (-0.09625, 0.09625),
    "nic_card_mount": (-0.048, 0.036),
}

SC_PORT_RANGES = {
    "sc_port_translation": (-0.06, 0.055),
    "sc_port_roll": (0.0, 0.0),
    "sc_port_pitch": (0.0, 0.0),
    "sc_port_yaw": (0.0, 0.0),
}


def range_defaults(name, lower, upper):
    return {f"{name}_min": lower, f"{name}_max": upper}


def jitter_ranges(defaults, jitters):
    return {
        name: (defaults[name] - jitter, defaults[name] + jitter)
        for name, jitter in jitters.items()
    }


def prefixed_jitter_ranges(prefix, defaults, jitters):
    return {
        f"{prefix}_{pose_field_name(name)}": bounds
        for name, bounds in jitter_ranges(defaults, jitters).items()
    }


def pose_field_name(name):
    return name.removeprefix("task_board_").removeprefix("cable_")


def part_translation_range(name, value):
    for key, limits in PART_TRANSLATION_RANGES.items():
        if key in name:
            return limits
    return value, value


def board_part_ranges():
    ranges = {}
    for name, values in BOARD_PART_DEFAULTS.items():
        if name.startswith(SC_PORT_PREFIX):
            continue
        _, translation, roll, pitch, yaw = values
        ranges.update(
            range_defaults(
                f"{name}_translation",
                *part_translation_range(name, translation),
            )
        )
        ranges.update(range_defaults(f"{name}_roll", roll, roll))
        ranges.update(range_defaults(f"{name}_pitch", pitch, pitch))
        ranges.update(range_defaults(f"{name}_yaw", yaw, yaw))
    return ranges


def sc_port_ranges():
    defaults = {}
    for name, bounds in SC_PORT_RANGES.items():
        defaults.update(range_defaults(name, *bounds))
    return defaults


def pose_ranges(ranges):
    defaults = {}
    for name, bounds in ranges.items():
        defaults.update(range_defaults(name, *bounds))
    return defaults


RANDOMIZATION_DEFAULTS = {
    "randomize_scene": True,
    "random_seed": -1,
    "randomization_distribution": "normal",
    "randomization_normal_stddevs": 3.0,
    "randomization_normal_max_attempts": 50,
    **pose_ranges(jitter_ranges(TASK_BOARD_DEFAULTS, SFP_TASK_BOARD_JITTER)),
    **board_part_ranges(),
    **sc_port_ranges(),
    **pose_ranges(jitter_ranges(CABLE_DEFAULTS, SFP_CABLE_JITTER)),
    **pose_ranges(
        prefixed_jitter_ranges(
            SC_TASK_BOARD_RANDOMIZATION_PREFIX,
            TASK_BOARD_DEFAULTS,
            SC_TASK_BOARD_JITTER,
        )
    ),
    **pose_ranges(
        prefixed_jitter_ranges(
            SC_CABLE_RANDOMIZATION_PREFIX,
            CABLE_POSE_PRESETS["sc"],
            SC_CABLE_JITTER,
        )
    ),
}


def declare_randomization_parameters(node):
    for name, value in RANDOMIZATION_DEFAULTS.items():
        node.declare_parameter(name, value)
