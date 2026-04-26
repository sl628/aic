import time
from pathlib import Path

from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from ch_milestones.config.task_config import (
    port_entrance_frame,
    port_link_frame,
    task_from_parameters,
)
from ch_milestones.environment.description import expand_xacro
from ch_milestones.environment.entities import EntitySpawner
from ch_milestones.environment.randomization import SceneRandomizer
from ch_milestones.environment.robot import RobotResetter


class EnvironmentResetter:
    def __init__(self, node, callback_group=None):
        self.node = node
        self.entities = EntitySpawner(node, callback_group)
        self.robot = RobotResetter(node, callback_group)
        self.tf_buffer = Buffer(cache_time=None, node=node)
        self.tf_listener = TransformListener(self.tf_buffer, node, spin_thread=False)
        self.randomizer = SceneRandomizer(node)
        self.spawned = []

    def reset(self):
        timeout = self.node.get_parameter("reset_timeout_seconds").value
        scene = self.randomizer.sample()
        self.log_scene(scene)
        self.delete_spawned(timeout)
        self.robot.home(timeout)
        if self.node.get_parameter("spawn_task_board").value:
            self.spawn_board(timeout, scene)
        self.robot.tare(timeout)
        if self.node.get_parameter("spawn_cable").value:
            self.spawn_cable(timeout, scene)
        time.sleep(self.node.get_parameter("post_spawn_settle_seconds").value)
        self.wait_for_task_frames()

    def clear(self):
        timeout = self.node.get_parameter("reset_timeout_seconds").value
        self.delete_spawned(timeout)
        self.robot.home(timeout)

    def delete_spawned(self, timeout):
        for name in reversed(self.spawned):
            self.entities.delete_entity(name, timeout)
        self.spawned.clear()

    def spawn_board(self, timeout, scene):
        name = self.node.get_parameter("task_board_name").value
        xml = expand_xacro(
            self.description_file("task_board.urdf.xacro"),
            scene.board_parts,
            self.package_paths(),
        )
        self.spawned.append(
            self.entities.spawn_entity(name, xml, scene.board_pose, timeout)
        )

    def spawn_cable(self, timeout, scene):
        name = self.node.get_parameter("cable_name").value
        xml = expand_xacro(
            self.description_file("cable.sdf.xacro"),
            {
                "attach_cable_to_gripper": self.node.get_parameter(
                    "attach_cable_to_gripper"
                ).value,
                "cable_type": self.node.get_parameter("cable_type").value,
            },
            self.package_paths(),
        )
        self.spawned.append(
            self.entities.spawn_entity(name, xml, self.cable_pose(scene), timeout)
        )

    def cable_pose(self, scene):
        return scene.cable_pose

    def log_scene(self, scene):
        def fmt(values):
            return ", ".join(f"{value:.4f}" for value in values)

        self.node.get_logger().info(
            "Scene sample: "
            f"board=({fmt(scene.board_pose)}), "
            f"cable=({fmt(scene.cable_pose)})"
        )

    def wait_for_task_frames(self):
        task = task_from_parameters(self.node)
        task_board = self.node.get_parameter("task_board_name").value
        cable = self.node.get_parameter("cable_name").value
        plug = self.node.get_parameter("plug_name").value
        self.wait_for("base_link", port_link_frame(task, task_board))
        self.wait_for("base_link", port_entrance_frame(task, task_board))
        self.wait_for("base_link", f"{cable}/{plug}_link")

    def wait_for(self, target_frame, source_frame):
        timeout = self.node.get_parameter("tf_timeout_seconds").value
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.tf_buffer.can_transform(target_frame, source_frame, Time()):
                return
            time.sleep(0.1)
        raise TimeoutError(f"Missing transform {source_frame} -> {target_frame}")

    def description_file(self, name):
        return Path(self.node.get_parameter("description_root").value) / "urdf" / name

    def package_paths(self):
        return {"aic_assets": Path(self.node.get_parameter("assets_root").value)}
