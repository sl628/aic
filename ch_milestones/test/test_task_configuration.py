import pytest

from ch_milestones.config.scene_config import (
    BOARD_PART_DEFAULTS,
    CABLE_DEFAULTS,
    CABLE_POSE_FIELDS,
    cable_pose_for_task,
    cable_type_for_task,
    ensure_target_board_part,
)
from ch_milestones.config.task_config import (
    TASK_FIELDS,
    task_from_parameters,
    task_sequence_size,
)
from ch_milestones.config.task_options import (
    TARGET_CONFIGURATIONS,
    plug_configuration,
    validate_task_values,
)


class Parameter:
    def __init__(self, value):
        self.value = value


class Node:
    def __init__(self, values):
        self.values = values

    def get_parameter(self, name):
        return Parameter(self.values[name])


def task_node(**overrides):
    values = {name: default for name, default in TASK_FIELDS}
    values["time_limit"] = 180
    values.update(overrides)
    return Node(values)


@pytest.mark.parametrize("target", TARGET_CONFIGURATIONS)
def test_all_supported_targets_resolve_to_their_plug_configuration(target):
    target_index = TARGET_CONFIGURATIONS.index(target)
    task = task_from_parameters(
        task_node(
            port_type="auto",
            plug_type="auto",
            plug_name="auto",
            target_module_name="all",
            port_name="all",
        ),
        target_index=target_index,
    )

    plug = plug_configuration(target.port_type)
    assert task.port_type == target.port_type
    assert task.plug_type == plug.plug_type
    assert task.plug_name == plug.plug_name


def test_default_task_selection_covers_every_supported_target():
    assert task_sequence_size(task_node()) == len(TARGET_CONFIGURATIONS)


def test_specific_target_selection_has_one_target():
    assert (
        task_sequence_size(
            task_node(target_module_name="nic_card_mount_4", port_name="sfp_port_1")
        )
        == 1
    )


def test_sc_target_resolves_to_reversed_cable_and_sc_pose():
    task = task_from_parameters(
        task_node(target_module_name="sc_port_1", port_name="sc_port_base")
    )

    assert task.port_type == "sc"
    assert (
        cable_type_for_task(task, CABLE_DEFAULTS["cable_type"])
        == "sfp_sc_cable_reversed"
    )
    assert cable_pose_for_task(
        task,
        tuple(CABLE_DEFAULTS[field] for field in CABLE_POSE_FIELDS),
        auto_cable_pose=True,
    ) == (0.172, 0.024, 1.508, 0.4432, -0.48, 1.3303)


def test_target_board_part_is_enabled_for_selected_mount():
    board_parts = {}
    for name, values in BOARD_PART_DEFAULTS.items():
        present, translation, roll, pitch, yaw = values
        board_parts[f"{name}_present"] = present
        board_parts[f"{name}_translation"] = translation
        board_parts[f"{name}_roll"] = roll
        board_parts[f"{name}_pitch"] = pitch
        board_parts[f"{name}_yaw"] = yaw

    task = task_from_parameters(
        task_node(target_module_name="nic_card_mount_4", port_name="sfp_port_1")
    )
    assert board_parts["nic_card_mount_4_present"] is False

    updated = ensure_target_board_part(task, board_parts)

    assert updated["nic_card_mount_4_present"] is True
    assert board_parts["nic_card_mount_4_present"] is False


def test_incompatible_plug_and_port_are_rejected():
    with pytest.raises(ValueError, match="plug_type"):
        validate_task_values(
            {
                "plug_type": "sc",
                "plug_name": "sc_tip",
                "port_type": "sfp",
                "port_name": "sfp_port_0",
                "target_module_name": "nic_card_mount_0",
            }
        )
