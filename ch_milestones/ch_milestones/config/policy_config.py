STAGES = ("approach", "coarse_align", "fine_align", "insert")

ORACLE_DEFAULTS = {
    "oracle_speed_scale": 0.5,
    "oracle_command_period": 0.05,
    "oracle_stage_timeout_seconds": 60.0,
    "oracle_cartesian_stiffness": [100.0, 100.0, 100.0, 50.0, 50.0, 50.0],
    "oracle_cartesian_damping": [40.0, 40.0, 40.0, 15.0, 15.0, 15.0],
    "oracle_approach_step_meters": 0.001,
    "oracle_approach_success_tolerance_meters": 0.010,
    "oracle_approach_z_offset": 0.15,
    "oracle_coarse_align_step_meters": 0.001,
    "oracle_coarse_align_success_tolerance_meters": 0.010,
    "oracle_coarse_align_success_rotation_tolerance_radians": 0.08,
    "oracle_hover_z_offset": 0.05,
    "oracle_fine_align_step_meters": 0.001,
    "oracle_fine_align_hold_steps": 30,
    "oracle_alignment_fine_align_z_offset": 0.175,
    "oracle_alignment_fine_align_steps": 350,
    "oracle_alignment_insert_step_meters": 0.00025,
    "oracle_alignment_command_period": 0.05,
    "oracle_alignment_integral_gain": 0.12,
    "oracle_alignment_integrator_limit": 0.05,
    "oracle_insert_step_meters": 0.0005,
    "oracle_insert_end_z_offset": -0.015,
    "oracle_final_settle_steps": 50,
    "oracle_publish_debug_frames": True,
    "oracle_debug_frame_prefix": "oracle_debug",
    "oracle_debug_log_frequency_hz": 3.0,
    "oracle_debug_pause_motion": False,
    "oracle_debug_pause_poll_seconds": 0.05,
    "oracle_debug_pause_before_motion_seconds": 0.0,
}


def declare_oracle_parameters(node):
    for name, value in ORACLE_DEFAULTS.items():
        if not node.has_parameter(name):
            node.declare_parameter(name, value)
