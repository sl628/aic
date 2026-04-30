from math import ceil

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task

from ch_milestones.config.policy_config import (
    SFP_ORACLE_DEFAULTS,
    STAGES,
    declare_oracle_parameters,
    oracle_parameter_name,
)
from ch_milestones.config.task_config import validate_task
from ch_milestones.policies.oracle_debug_frames import OracleDebugFrames
from ch_milestones.policies.ground_truth_guidance import GroundTruthGuide
from ch_milestones.policies.oracle_frames import OracleFrames
from ch_milestones.policies.oracle_motion import OracleMotionCommander
from ch_milestones.policies.oracle_stages import OracleStageSet
from ch_milestones.policies.oracle_validation import validate_oracle_params


class OraclePolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        declare_oracle_parameters(parent_node)
        self._task = None

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"OraclePolicy.insert_cable() task: {task}")
        validate_task(task)
        self.setup(task, move_robot, send_feedback)
        self.validate_params()

        try:
            self._stages.approach.run()
            self._stages.coarse_align.run()
            self._stages.fine_align.run()
            self._stages.insert.run()
        except TimeoutError as ex:
            self.get_logger().error(str(ex))
            self._send_feedback(f"error:{ex}")
            return False

        self._send_feedback("settling")
        for _ in range(self.final_settle_steps()):
            self.sleep_for(self.command_period())
        return True

    def setup(self, task, move_robot, send_feedback):
        self._task = task
        self._guide = GroundTruthGuide(
            self._parent_node, task, param=self.param
        )
        self._frames = OracleFrames.load(self._parent_node, self._guide, task)
        self._move_robot = move_robot
        self._send_feedback = send_feedback
        self._stage = None
        self._debug_frames = OracleDebugFrames(
            self._parent_node, param=self.param
        )
        self._motion = OracleMotionCommander(self)
        self._stages = OracleStageSet(self)

    @property
    def guide(self):
        return self._guide

    @property
    def frames(self):
        return self._frames

    @property
    def motion(self):
        return self._motion

    @property
    def debug_frames(self):
        return self._debug_frames

    @property
    def move_robot(self):
        return self._move_robot

    @property
    def stage(self):
        return self._stage

    def validate_params(self):
        validate_oracle_params(self.param)

    def set_stage(self, stage):
        if stage not in STAGES:
            raise ValueError(f"Unknown oracle stage: {stage}")
        self._stage = stage
        self._send_feedback(f"stage:{stage}")

    def gripper_pose(self, **kwargs):
        if self._stage is None:
            raise RuntimeError("Oracle stage is not set")
        z_offset = kwargs.get("z_offset")
        if self._stage != "insert" and z_offset is not None and z_offset < 0.0:
            raise ValueError(
                f"{self._stage} z_offset must stay above the port"
            )
        position_ref = kwargs.pop("position_ref", self.frames.port_ref)
        orientation_ref = self.frames.orientation_ref
        self.debug_frames.publish_reference_frame(
            orientation_ref,
            position_ref,
            z_offset if z_offset is not None else 0.1,
        )
        pose = self.guide.gripper_pose(
            orientation_ref,
            position_ref,
            stage=self._stage,
            **kwargs,
        )
        debug = self.guide.last_gripper_pose_debug
        if debug is not None:
            self.debug_frames.publish_goal_plug_frame(debug["goal_plug"])
        return pose

    def current_z_offset(self, position_ref):
        return self.guide.approach_offset(
            self.frames.orientation_ref,
            position_ref,
            self.frames.plug_frame,
        )

    def speed_scale(self):
        speed = self.param("oracle_speed_scale")
        if speed <= 0:
            raise ValueError("oracle_speed_scale must be positive")
        return speed

    def command_period(self):
        period = self.param("oracle_command_period")
        if period <= 0:
            raise ValueError("oracle_command_period must be positive")
        return period

    def final_settle_steps(self):
        return max(
            0,
            ceil(self.param("oracle_final_settle_steps") / self.speed_scale()),
        )

    def param(self, name):
        parameter = self._parent_node.get_parameter(self.parameter_name(name))
        value = parameter.value
        if self.use_legacy_sfp_parameter(name, value):
            return self._parent_node.get_parameter(name).value
        return value

    def parameter_name(self, name):
        return oracle_parameter_name(name, self._task)

    def use_legacy_sfp_parameter(self, name, value):
        if self._task is None or self._task.port_type != "sfp":
            return False
        if name not in SFP_ORACLE_DEFAULTS:
            return False
        legacy_value = self._parent_node.get_parameter(name).value
        return value == SFP_ORACLE_DEFAULTS[name] and (
            legacy_value != SFP_ORACLE_DEFAULTS[name]
        )
