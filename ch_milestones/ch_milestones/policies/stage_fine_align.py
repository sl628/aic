from ch_milestones.policies.oracle_stage_base import OracleStage
from ch_milestones.policies.cartesian_trajectory import minimum_jerk


class FineAlignStage(OracleStage):
    stage = "fine_align"

    def run(self):
        self.begin()
        z_offset = self.param("oracle_alignment_fine_align_z_offset")
        steps = int(self.param("oracle_alignment_fine_align_steps"))
        command_period = self.param("oracle_alignment_command_period")

        denominator = max(1, steps - 1)
        for t in range(steps):
            interp_fraction = minimum_jerk(t / float(denominator))
            pose = self.policy.guide.alignment_gripper_pose(
                self.frames.port_ref,
                slerp_fraction=interp_fraction,
                position_fraction=interp_fraction,
                z_offset=z_offset,
                reset_xy_integrator=True,
            )
            self.policy.set_pose_target(
                move_robot=self.policy.move_robot,
                pose=pose,
            )
            self.policy.sleep_for(command_period)
