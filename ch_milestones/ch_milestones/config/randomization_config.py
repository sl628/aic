from ch_milestones.config.scene_config import (
    BOARD_PART_DEFAULTS,
    CABLE_DEFAULTS,
    TASK_BOARD_DEFAULTS,
)


TASK_BOARD_JITTER = {
    "task_board_x": 0.05,
    "task_board_y": 0.05,
    "task_board_z": 0.0,
    "task_board_roll": 0.0,
    "task_board_pitch": 0.0,
    "task_board_yaw": 1.57/1.6,
}

CABLE_JITTER = {
    "cable_x": 0.0,
    "cable_y": 0.0,
    "cable_z": 0.0,
    "cable_roll": 0.0,
    "cable_pitch": 0.0,
    "cable_yaw": 0.0,
}

PART_TRANSLATION_RANGES = {
    "mount_rail": (-0.09625, 0.09625),
    "sc_port": (-0.06, 0.055),
    "nic_card_mount": (-0.048, 0.036),
}


def range_defaults(name, lower, upper):
    return {f"{name}_min": lower, f"{name}_max": upper}


def jitter_ranges(defaults, jitters):
    return {
        name: (defaults[name] - jitter, defaults[name] + jitter)
        for name, jitter in jitters.items()
    }


def part_translation_range(name, value):
    for key, limits in PART_TRANSLATION_RANGES.items():
        if key in name:
            return limits
    return value, value


def board_part_ranges():
    ranges = {}
    for name, values in BOARD_PART_DEFAULTS.items():
        _, translation, roll, pitch, yaw = values
        ranges.update(
            range_defaults(
                f"{name}_translation", *part_translation_range(name, translation)
            )
        )
        ranges.update(range_defaults(f"{name}_roll", roll, roll))
        ranges.update(range_defaults(f"{name}_pitch", pitch, pitch))
        ranges.update(range_defaults(f"{name}_yaw", yaw, yaw))
    return ranges


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
    **pose_ranges(jitter_ranges(TASK_BOARD_DEFAULTS, TASK_BOARD_JITTER)),
    **board_part_ranges(),
    **pose_ranges(jitter_ranges(CABLE_DEFAULTS, CABLE_JITTER)),
}


def declare_randomization_parameters(node):
    for name, value in RANDOMIZATION_DEFAULTS.items():
        node.declare_parameter(name, value)
