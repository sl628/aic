from ch_milestones.policies.oracle_stage_base import OracleStage


class ApproachStage(OracleStage):
    stage = "approach"

    def run(self):
        self.begin()
        z_offset = self.param("oracle_approach_z_offset")
        step_meters = self.param("oracle_approach_step_meters") * self.policy.speed_scale()

        self.policy.guide.reset_integrator()
        self.motion.execute_live_step_trajectory(
            lambda: self.live_approach_pose(z_offset),
            step_meters,
            self.param("oracle_approach_success_tolerance_meters"),
            self.timeout_seconds(),
        )

    def live_approach_pose(self, z_offset):
        return self.policy.gripper_pose(
            slerp_fraction=0.0,
            position_fraction=1.0,
            position_ref=self.frames.offset_ref(z_offset),
            z_offset=0.0,
            reset_xy_integrator=True,
            xy_integral_gain=0.0,
        )
