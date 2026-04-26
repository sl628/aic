import sys

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from ch_milestones.config.randomization_config import (
    declare_randomization_parameters,
)
from ch_milestones.config.scene_config import declare_scene_parameters
from ch_milestones.config.task_config import declare_task_parameters
from ch_milestones.environment.resetter import EnvironmentResetter


class EnvironmentResetterNode(Node):
    def __init__(self):
        super().__init__("ch_milestone_environment_resetter")
        declare_task_parameters(self)
        declare_scene_parameters(self)
        declare_randomization_parameters(self)
        self.callback_group = ReentrantCallbackGroup()
        self.resetter = EnvironmentResetter(self, self.callback_group)
        self.create_service(
            Trigger,
            "/ch_milestones/reset_episode",
            self.reset,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "/ch_milestones/clear_environment",
            self.clear,
            callback_group=self.callback_group,
        )

    def reset(self, request, response):
        self.get_logger().info("Resetting milestone environment")
        self.resetter.reset()
        response.success = True
        response.message = "Milestone environment reset"
        return response

    def clear(self, request, response):
        self.get_logger().info("Clearing milestone environment")
        self.resetter.clear()
        response.success = True
        response.message = "Milestone environment cleared"
        return response


def main(args=None):
    try:
        with rclpy.init(args=args):
            node = EnvironmentResetterNode()
            executor = MultiThreadedExecutor()
            executor.add_node(node)
            try:
                executor.spin()
            finally:
                executor.shutdown()
                node.destroy_node()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == "__main__":
    main(sys.argv)
