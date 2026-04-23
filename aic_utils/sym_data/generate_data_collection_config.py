#!/usr/bin/env python3
"""Generate a randomized aic_engine config for CheatCodeDataCollector.

Produces N trials per scenario (3 scenarios × N = total trials), with
task_board pose, rail translations, and cable z-offset sampled from uniform
distributions within validated bounds.

Usage::

    pixi run python aic_utils/sym_data/generate_data_collection_config.py \\
        --episodes_per_scenario 100 \\
        --output /tmp/data_collection_config.yaml \\
        --seed 42

Then launch the eval container with the generated config::

    distrobox enter -r aic_eval -- /entrypoint.sh \\
        ground_truth:=true \\
        start_aic_engine:=true \\
        "aic_engine_config_file:=/tmp/data_collection_config.yaml"
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Fixed sections (copied verbatim from sample_config.yaml)
# ---------------------------------------------------------------------------

SCORING = {
    "record_bag": False,
    "topics": [
        {"topic": {"name": "/joint_states", "type": "sensor_msgs/msg/JointState"}},
        {"topic": {"name": "/tf", "type": "tf2_msgs/msg/TFMessage"}},
        {
            "topic": {
                "name": "/tf_static",
                "type": "tf2_msgs/msg/TFMessage",
                "latched": True,
            }
        },
        {"topic": {"name": "/scoring/tf", "type": "tf2_msgs/msg/TFMessage"}},
        {
            "topic": {
                "name": "/aic/gazebo/contacts/off_limit",
                "type": "ros_gz_interfaces/msg/Contacts",
            }
        },
        {
            "topic": {
                "name": "/fts_broadcaster/wrench",
                "type": "geometry_msgs/msg/WrenchStamped",
            }
        },
        {
            "topic": {
                "name": "/aic_controller/joint_commands",
                "type": "aic_control_interfaces/msg/JointMotionUpdate",
            }
        },
        {
            "topic": {
                "name": "/aic_controller/pose_commands",
                "type": "aic_control_interfaces/msg/MotionUpdate",
            }
        },
        {"topic": {"name": "/scoring/insertion_event", "type": "std_msgs/msg/String"}},
        {
            "topic": {
                "name": "/aic_controller/controller_state",
                "type": "aic_control_interfaces/msg/ControllerState",
            }
        },
    ],
}

TASK_BOARD_LIMITS = {
    "nic_rail": {"min_translation": -0.0215, "max_translation": 0.0234},
    "sc_rail": {"min_translation": -0.06, "max_translation": 0.055},
    "mount_rail": {"min_translation": -0.09425, "max_translation": 0.09425},
}

ROBOT = {
    "home_joint_positions": {
        "shoulder_pan_joint": -0.1597,
        "shoulder_lift_joint": -1.3542,
        "elbow_joint": -1.6648,
        "wrist_1_joint": -1.6933,
        "wrist_2_joint": 1.5710,
        "wrist_3_joint": 1.4110,
    }
}

# ---------------------------------------------------------------------------
# Randomization ranges
# ---------------------------------------------------------------------------

YAW_CENTER = math.pi  # 3.1415...
YAW_RANGE = 0.30  # ± rad

# Task board x/y per scenario
NIC_X_RANGE = (0.13, 0.17)
NIC_Y_RANGE = (-0.22, -0.18)
SC_X_RANGE = (0.15, 0.19)
SC_Y_RANGE = (-0.02, 0.02)

# Rail translation ranges (stay inside task_board_limits)
NIC_RAIL_RANGE = (-0.021, 0.023)
SC_RAIL_RANGE = (-0.06, 0.055)
MOUNT_RAIL_RANGE = (-0.09, 0.09)

# Cable gripper z-offset range
CABLE_Z_RANGE = (0.040, 0.046)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _r(lo: float, hi: float) -> float:
    """Uniform sample rounded to 5 decimal places."""
    return round(random.uniform(lo, hi), 5)


def _board_pose(x: float, y: float, yaw: float) -> dict:
    return {"x": x, "y": y, "z": 1.14, "roll": 0.0, "pitch": 0.0, "yaw": round(yaw, 5)}


def _entity(translation: float) -> dict:
    return {"translation": round(translation, 5), "roll": 0.0, "pitch": 0.0, "yaw": 0.0}


def _absent() -> dict:
    return {"entity_present": False}


def _present(name: str, translation: float, yaw: float = 0.0) -> dict:
    return {
        "entity_present": True,
        "entity_name": name,
        "entity_pose": {
            "translation": round(translation, 5),
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": round(yaw, 5),
        },
    }


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def _scenario_a(rng_yaw: float, rng_x: float, rng_y: float, cable_z: float) -> dict:
    """SFP → nic_card_mount_0 (nic_rail_0)."""
    return {
        "scene": {
            "task_board": {
                "pose": _board_pose(rng_x, rng_y, rng_yaw),
                "nic_rail_0": _present("nic_card_0", _r(*NIC_RAIL_RANGE)),
                "nic_rail_1": _absent(),
                "nic_rail_2": _absent(),
                "nic_rail_3": _absent(),
                "nic_rail_4": _absent(),
                "sc_rail_0": _present("sc_mount_0", _r(*SC_RAIL_RANGE), yaw=0.1),
                "sc_rail_1": _absent(),
                "lc_mount_rail_0": _present("lc_mount_0", _r(*MOUNT_RAIL_RANGE)),
                "sfp_mount_rail_0": _present("sfp_mount_0", _r(*MOUNT_RAIL_RANGE)),
                "sc_mount_rail_0": _present("sc_mount_0", _r(*MOUNT_RAIL_RANGE)),
                "lc_mount_rail_1": _present("lc_mount_1", _r(*MOUNT_RAIL_RANGE)),
                "sfp_mount_rail_1": _absent(),
                "sc_mount_rail_1": _absent(),
            },
            "cables": {
                "cable_0": {
                    "pose": {
                        "gripper_offset": {
                            "x": 0.0,
                            "y": 0.015385,
                            "z": round(cable_z, 5),
                        },
                        "roll": 0.4432,
                        "pitch": -0.4838,
                        "yaw": 1.3303,
                    },
                    "attach_cable_to_gripper": True,
                    "cable_type": "sfp_sc_cable",
                }
            },
        },
        "tasks": {
            "task_1": {
                "cable_type": "sfp_sc",
                "cable_name": "cable_0",
                "plug_type": "sfp",
                "plug_name": "sfp_tip",
                "port_type": "sfp",
                "port_name": "sfp_port_0",
                "target_module_name": "nic_card_mount_0",
                "time_limit": 180,
            }
        },
    }


def _scenario_b(rng_yaw: float, rng_x: float, rng_y: float, cable_z: float) -> dict:
    """SFP → nic_card_mount_1 (nic_rail_1)."""
    return {
        "scene": {
            "task_board": {
                "pose": _board_pose(rng_x, rng_y, rng_yaw),
                "nic_rail_0": _absent(),
                "nic_rail_1": _present("nic_card_1", _r(*NIC_RAIL_RANGE)),
                "nic_rail_2": _absent(),
                "nic_rail_3": _absent(),
                "nic_rail_4": _absent(),
                "sc_rail_0": _present("sc_mount_0", _r(*SC_RAIL_RANGE), yaw=0.1),
                "sc_rail_1": _absent(),
                "lc_mount_rail_0": _present("lc_mount_0", _r(*MOUNT_RAIL_RANGE)),
                "sfp_mount_rail_0": _present("sfp_mount_0", _r(*MOUNT_RAIL_RANGE)),
                "sc_mount_rail_0": _present("sc_mount_0", _r(*MOUNT_RAIL_RANGE)),
                "lc_mount_rail_1": _present("lc_mount_1", _r(*MOUNT_RAIL_RANGE)),
                "sfp_mount_rail_1": _absent(),
                "sc_mount_rail_1": _absent(),
            },
            "cables": {
                "cable_0": {
                    "pose": {
                        "gripper_offset": {
                            "x": 0.0,
                            "y": 0.015385,
                            "z": round(cable_z, 5),
                        },
                        "roll": 0.4432,
                        "pitch": -0.4838,
                        "yaw": 1.3303,
                    },
                    "attach_cable_to_gripper": True,
                    "cable_type": "sfp_sc_cable",
                }
            },
        },
        "tasks": {
            "task_1": {
                "cable_type": "sfp_sc",
                "cable_name": "cable_0",
                "plug_type": "sfp",
                "plug_name": "sfp_tip",
                "port_type": "sfp",
                "port_name": "sfp_port_0",
                "target_module_name": "nic_card_mount_1",
                "time_limit": 180,
            }
        },
    }


def _scenario_c(rng_yaw: float, rng_x: float, rng_y: float, cable_z: float) -> dict:
    """SC → sc_port_1 (sc_rail_1)."""
    return {
        "scene": {
            "task_board": {
                "pose": _board_pose(rng_x, rng_y, rng_yaw),
                "nic_rail_0": _absent(),
                "nic_rail_1": _absent(),
                "nic_rail_2": _absent(),
                "nic_rail_3": _absent(),
                "nic_rail_4": _absent(),
                "sc_rail_0": _absent(),
                "sc_rail_1": _present("sc_mount_1", _r(*SC_RAIL_RANGE)),
                "lc_mount_rail_0": _absent(),
                "sfp_mount_rail_0": _present("sfp_mount_0", _r(*MOUNT_RAIL_RANGE)),
                "sc_mount_rail_0": _present("sc_mount_2", _r(*MOUNT_RAIL_RANGE)),
                "lc_mount_rail_1": _present("lc_mount_1", _r(*MOUNT_RAIL_RANGE)),
                "sfp_mount_rail_1": _absent(),
                "sc_mount_rail_1": _absent(),
            },
            "cables": {
                "cable_1": {
                    "pose": {
                        "gripper_offset": {
                            "x": 0.0,
                            "y": 0.015385,
                            "z": round(cable_z, 5),
                        },
                        "roll": 0.4432,
                        "pitch": -0.4838,
                        "yaw": 1.3303,
                    },
                    "attach_cable_to_gripper": True,
                    "cable_type": "sfp_sc_cable_reversed",
                }
            },
        },
        "tasks": {
            "task_1": {
                "cable_type": "sfp_sc",
                "cable_name": "cable_1",
                "plug_type": "sc",
                "plug_name": "sc_tip",
                "port_type": "sc",
                "port_name": "sc_port_base",
                "target_module_name": "sc_port_1",
                "time_limit": 180,
            }
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate(episodes_per_scenario: int, seed: int) -> dict:
    random.seed(seed)

    trials: dict = {}
    trial_idx = 1

    scenarios = [
        ("A: SFP→nic_card_mount_0", _scenario_a, NIC_X_RANGE, NIC_Y_RANGE),
        ("B: SFP→nic_card_mount_1", _scenario_b, NIC_X_RANGE, NIC_Y_RANGE),
        ("C: SC→sc_port_1", _scenario_c, SC_X_RANGE, SC_Y_RANGE),
    ]

    for _label, builder, x_range, y_range in scenarios:
        for _ in range(episodes_per_scenario):
            yaw = _r(YAW_CENTER - YAW_RANGE, YAW_CENTER + YAW_RANGE)
            x = _r(*x_range)
            y = _r(*y_range)
            cable_z = _r(*CABLE_Z_RANGE)
            trials[f"trial_{trial_idx}"] = builder(yaw, x, y, cable_z)
            trial_idx += 1

    return {
        "scoring": SCORING,
        "task_board_limits": TASK_BOARD_LIMITS,
        "trials": trials,
        "robot": ROBOT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes_per_scenario",
        type=int,
        default=100,
        help="Number of trials per scenario (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/data_collection_config.yaml",
        help="Output YAML path (default: /tmp/data_collection_config.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    config = generate(args.episodes_per_scenario, args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# Auto-generated by generate_data_collection_config.py\n")
        f.write(
            f"# episodes_per_scenario={args.episodes_per_scenario}, seed={args.seed}\n"
        )
        f.write(f"# total_trials={3 * args.episodes_per_scenario}\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    total = 3 * args.episodes_per_scenario
    print(
        f"Generated {total} trials ({args.episodes_per_scenario} × 3 scenarios) → {out}"
    )


if __name__ == "__main__":
    main()
