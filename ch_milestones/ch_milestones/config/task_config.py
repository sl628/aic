from aic_task_interfaces.msg import Task


TASK_FIELDS = [
    ("task_id", "task_1"),
    ("task_cable_type", "sfp_sc"),
    ("cable_name", "cable_0"),
    ("plug_type", "sfp"),
    ("plug_name", "sfp_tip"),
    ("port_type", "sfp"),
    ("port_name", "sfp_port_0"),
    ("target_module_name", "nic_card_mount_0"),
]


def declare_task_parameters(node):
    for name, default in TASK_FIELDS:
        node.declare_parameter(name, default)
    node.declare_parameter("time_limit", 180)


def task_from_parameters(node) -> Task:
    return Task(
        id=node.get_parameter("task_id").value,
        cable_type=node.get_parameter("task_cable_type").value,
        cable_name=node.get_parameter("cable_name").value,
        plug_type=node.get_parameter("plug_type").value,
        plug_name=node.get_parameter("plug_name").value,
        port_type=node.get_parameter("port_type").value,
        port_name=node.get_parameter("port_name").value,
        target_module_name=node.get_parameter("target_module_name").value,
        time_limit=node.get_parameter("time_limit").value,
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
