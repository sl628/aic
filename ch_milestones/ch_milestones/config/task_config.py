from aic_task_interfaces.msg import Task

from ch_milestones.config.task_options import (
    ALL_VALUE,
    AUTO_VALUE,
    resolve_plug_field,
    resolve_target,
    target_sequence_size as target_options_size,
    validate_task_values,
)


TASK_FIELDS = [
    ("task_id", "task_1"),
    ("task_cable_type", "sfp_sc"),
    ("cable_name", "cable_0"),
    ("plug_type", AUTO_VALUE),
    ("plug_name", AUTO_VALUE),
    ("port_type", AUTO_VALUE),
    ("port_name", ALL_VALUE),
    ("target_module_name", ALL_VALUE),
]


def declare_task_parameters(node):
    for name, default in TASK_FIELDS:
        node.declare_parameter(name, default)
    node.declare_parameter("time_limit", 180)


def task_from_parameters(node, target_index: int = 0) -> Task:
    values = task_values_from_parameters(node, target_index)
    validate_task_values(values)
    return Task(
        id=values["task_id"],
        cable_type=values["task_cable_type"],
        cable_name=values["cable_name"],
        plug_type=values["plug_type"],
        plug_name=values["plug_name"],
        port_type=values["port_type"],
        port_name=values["port_name"],
        target_module_name=values["target_module_name"],
        time_limit=values["time_limit"],
    )


def task_values_from_parameters(node, target_index: int = 0):
    values = {name: node.get_parameter(name).value for name, _ in TASK_FIELDS}
    values["time_limit"] = node.get_parameter("time_limit").value
    target = resolve_target(
        values["port_type"],
        values["target_module_name"],
        values["port_name"],
        target_index,
    )
    values["port_type"] = target.port_type
    values["target_module_name"] = target.target_module_name
    values["port_name"] = target.port_name
    values["plug_type"] = resolve_plug_field(
        values["port_type"], "plug_type", values["plug_type"]
    )
    values["plug_name"] = resolve_plug_field(
        values["port_type"], "plug_name", values["plug_name"]
    )
    return values


def task_sequence_size(node) -> int:
    values = {name: node.get_parameter(name).value for name, _ in TASK_FIELDS}
    return target_options_size(
        values["port_type"], values["target_module_name"], values["port_name"]
    )


def validate_task(task: Task) -> None:
    validate_task_values(
        {
            "plug_type": task.plug_type,
            "plug_name": task.plug_name,
            "port_type": task.port_type,
            "port_name": task.port_name,
            "target_module_name": task.target_module_name,
        }
    )


def task_prompt(task: Task) -> str:
    return (
        f"insert {task.plug_type} plug {task.plug_name} "
        f"into {task.port_type} port {task.target_module_name}/{task.port_name}"
    )


def port_link_frame(task: Task, task_board_name="task_board") -> str:
    return f"{task_board_name}/{task.target_module_name}/{task.port_name}_link"


def port_entrance_frame(task: Task, task_board_name="task_board") -> str:
    return f"{port_link_frame(task, task_board_name)}_entrance"
