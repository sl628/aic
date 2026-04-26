from aic_engine_interfaces.srv import ResetJoints
from controller_manager_msgs.srv import SwitchController
from std_srvs.srv import Trigger

from ch_milestones.environment.services import call


class RobotResetter:
    def __init__(self, node, callback_group=None):
        self.node = node
        self.reset_joints = node.create_client(
            ResetJoints, "/scoring/reset_joints", callback_group=callback_group
        )
        self.switch_controller = node.create_client(
            SwitchController,
            "/controller_manager/switch_controller",
            callback_group=callback_group,
        )
        self.tare_ft = node.create_client(
            Trigger,
            "/aic_controller/tare_force_torque_sensor",
            callback_group=callback_group,
        )

    def home(self, timeout_sec):
        self.switch([], ["aic_controller"], timeout_sec)
        request = ResetJoints.Request()
        request.joint_names = self.node.get_parameter("home_joint_names").value
        request.initial_positions = self.node.get_parameter(
            "home_joint_positions"
        ).value
        response = call(
            self.node, self.reset_joints, request, timeout_sec, "reset joints"
        )
        if not response.success:
            raise RuntimeError(response.message)
        self.switch(["aic_controller"], [], timeout_sec)

    def tare(self, timeout_sec):
        response = call(
            self.node, self.tare_ft, Trigger.Request(), timeout_sec, "tare FT"
        )
        if not response.success:
            raise RuntimeError(response.message)

    def switch(self, activate, deactivate, timeout_sec):
        request = SwitchController.Request()
        request.activate_controllers = activate
        request.deactivate_controllers = deactivate
        request.strictness = SwitchController.Request.BEST_EFFORT
        response = call(
            self.node, self.switch_controller, request, timeout_sec, "switch controller"
        )
        if not response.ok:
            raise RuntimeError("Failed to switch controllers")
