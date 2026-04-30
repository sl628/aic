from types import SimpleNamespace

from ch_milestones.config.randomization_config import (
    RANDOMIZATION_DEFAULTS,
    SC_CABLE_RANDOMIZATION_PREFIX,
    SC_TASK_BOARD_RANDOMIZATION_PREFIX,
)
from ch_milestones.config.scene_config import CABLE_POSE_PRESETS
from ch_milestones.environment.randomization import SceneRandomizer


class Parameter:
    def __init__(self, value):
        self.value = value


class Node:
    def __init__(self, values):
        self.values = values

    def get_parameter(self, name):
        return Parameter(self.values[name])


def test_sc_ports_have_shared_randomization_parameters():
    assert "sc_port_translation_min" in RANDOMIZATION_DEFAULTS
    assert "sc_port_translation_max" in RANDOMIZATION_DEFAULTS
    assert "sc_port_yaw_min" in RANDOMIZATION_DEFAULTS
    assert "sc_port_yaw_max" in RANDOMIZATION_DEFAULTS

    assert "sc_port_0_translation_min" not in RANDOMIZATION_DEFAULTS
    assert "sc_port_1_translation_min" not in RANDOMIZATION_DEFAULTS


def test_sc_targets_have_separate_board_and_cable_jitter_parameters():
    assert "sc_task_board_x_min" in RANDOMIZATION_DEFAULTS
    assert "sc_task_board_yaw_max" in RANDOMIZATION_DEFAULTS
    assert "sc_cable_z_min" in RANDOMIZATION_DEFAULTS
    assert "sc_cable_yaw_max" in RANDOMIZATION_DEFAULTS

    expected_sc_cable_z = CABLE_POSE_PRESETS["sc"]["cable_z"]
    assert RANDOMIZATION_DEFAULTS["sc_cable_z_min"] == expected_sc_cable_z
    assert RANDOMIZATION_DEFAULTS["sc_cable_z_max"] == expected_sc_cable_z


def test_randomizer_uses_sc_pose_randomization_keys_for_sc_targets():
    randomizer = SceneRandomizer(Node({"random_seed": -1}))

    assert randomizer.pose_randomization_key(
        "task_board", SimpleNamespace(port_type="sc")
    ) == SC_TASK_BOARD_RANDOMIZATION_PREFIX
    assert randomizer.pose_randomization_key(
        "cable", SimpleNamespace(port_type="sc")
    ) == SC_CABLE_RANDOMIZATION_PREFIX
    assert (
        randomizer.pose_randomization_key(
            "task_board", SimpleNamespace(port_type="sfp")
        )
        == "task_board"
    )


def test_randomizer_uses_shared_sc_port_randomization_keys():
    randomizer = SceneRandomizer(Node({"random_seed": -1}))

    assert randomizer.randomization_key("sc_port_1", "translation") == (
        "sc_port_translation"
    )
    assert randomizer.randomization_key("sc_port_1", "yaw") == "sc_port_yaw"
    assert randomizer.randomization_key("nic_card_mount_1", "translation") == (
        "nic_card_mount_1_translation"
    )
