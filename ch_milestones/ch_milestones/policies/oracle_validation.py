def validate_oracle_params(param):
    approach_z = param("oracle_approach_z_offset")
    hover_z = param("oracle_hover_z_offset")
    insert_end_z = param("oracle_insert_end_z_offset")

    for name in (
        "oracle_speed_scale",
        "oracle_command_period",
        "oracle_stage_timeout_seconds",
        "oracle_approach_step_meters",
        "oracle_approach_success_tolerance_meters",
        "oracle_coarse_align_step_meters",
        "oracle_coarse_align_success_tolerance_meters",
        "oracle_coarse_align_success_rotation_tolerance_radians",
        "oracle_fine_align_step_meters",
        "oracle_alignment_fine_align_z_offset",
        "oracle_alignment_fine_align_steps",
        "oracle_alignment_insert_step_meters",
        "oracle_alignment_command_period",
        "oracle_insert_step_meters",
        "oracle_alignment_integrator_limit",
        "oracle_debug_log_frequency_hz",
        "oracle_debug_pause_poll_seconds",
    ):
        if param(name) <= 0:
            raise ValueError(f"{name} must be positive")

    for name in (
        "oracle_fine_align_hold_steps",
        "oracle_final_settle_steps",
        "oracle_debug_pause_before_motion_seconds",
    ):
        if param(name) < 0:
            raise ValueError(f"{name} must be non-negative")

    if param("oracle_alignment_integral_gain") < 0:
        raise ValueError("oracle_alignment_integral_gain must be non-negative")

    for name in ("oracle_cartesian_stiffness", "oracle_cartesian_damping"):
        values = param(name)
        if len(values) != 6:
            raise ValueError(f"{name} must have 6 values")
        if any(value <= 0 for value in values):
            raise ValueError(f"{name} values must be positive")

    if not approach_z > hover_z > 0.0 > insert_end_z:
        raise ValueError(
            "Expected oracle_approach_z_offset > oracle_hover_z_offset > 0 "
            "> oracle_insert_end_z_offset"
        )
