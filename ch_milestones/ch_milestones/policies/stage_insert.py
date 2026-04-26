from ch_milestones.policies.oracle_stage_base import OracleStage


class InsertStage(OracleStage):
    stage = "insert"

    def run(self):
        self.begin()
        z_offset = self.param("oracle_alignment_fine_align_z_offset")
        step_m = self.param("oracle_alignment_insert_step_meters")
        command_period = self.param("oracle_alignment_command_period")

        while True:
            if z_offset < -0.015:
                break

            z_offset -= step_m
            self.policy.get_logger().info(f"z_offset: {z_offset:0.5}")
            pose = self.policy.guide.alignment_gripper_pose(
                self.frames.port_ref,
                slerp_fraction=1.0,
                position_fraction=1.0,
                z_offset=z_offset,
            )
            self.policy.set_pose_target(
                move_robot=self.policy.move_robot,
                pose=pose,
            )
            self.policy.sleep_for(command_period)
