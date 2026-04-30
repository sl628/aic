"""Empty policy that does nothing — useful for observing the initial setup."""

import time
from aic_task_interfaces.msg import Task
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)


class EmptyPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.get_logger().info("EmptyPolicy loaded — robot will not move")

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ) -> bool:
        self.get_logger().info("EmptyPolicy.insert_cable() — doing nothing for 60s")
        send_feedback("EmptyPolicy: holding position")
        time.sleep(60)
        self.get_logger().info("EmptyPolicy.insert_cable() — done")
        return True
