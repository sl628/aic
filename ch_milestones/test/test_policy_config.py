from types import SimpleNamespace

from ch_milestones.config.policy_config import (
    ORACLE_DEFAULTS,
    SC_ORACLE_DEFAULTS,
    SC_ORACLE_PARAMETER_PREFIX,
    SFP_ORACLE_DEFAULTS,
    SFP_ORACLE_PARAMETER_PREFIX,
    ORACLE_PARAMETER_BASE_PREFIX,
    oracle_parameter_name,
    sc_oracle_parameter_name,
    sfp_oracle_parameter_name,
)


def test_sc_oracle_parameters_are_declared_with_sc_prefix():
    assert ORACLE_PARAMETER_BASE_PREFIX == "oracle"
    assert SFP_ORACLE_PARAMETER_PREFIX == "sfp_oracle"
    assert SC_ORACLE_PARAMETER_PREFIX == "sc_oracle"

    for name, value in SFP_ORACLE_DEFAULTS.items():
        sfp_name = sfp_oracle_parameter_name(name)
        sc_name = sc_oracle_parameter_name(name)

        assert sfp_name == name.replace("oracle_", "sfp_oracle_", 1)
        assert name in ORACLE_DEFAULTS
        assert sfp_name in ORACLE_DEFAULTS
        assert sc_name in ORACLE_DEFAULTS
        assert ORACLE_DEFAULTS[name] == value
        assert ORACLE_DEFAULTS[sfp_name] == value
        assert ORACLE_DEFAULTS[sc_name] == SC_ORACLE_DEFAULTS[name]


def test_oracle_parameter_name_selects_task_specific_parameter():
    assert oracle_parameter_name(
        "oracle_speed_scale",
        SimpleNamespace(port_type="sfp"),
    ) == "sfp_oracle_speed_scale"
    assert oracle_parameter_name(
        "oracle_speed_scale",
        SimpleNamespace(port_type="sc"),
    ) == "sc_oracle_speed_scale"
    assert (
        oracle_parameter_name("oracle_speed_scale")
        == "sfp_oracle_speed_scale"
    )


def test_sc_fine_align_hover_is_lower_than_sfp_default():
    name = "oracle_alignment_fine_align_z_offset"

    assert SFP_ORACLE_DEFAULTS[name] == 0.175
    assert SC_ORACLE_DEFAULTS[name] == 0.05
    assert ORACLE_DEFAULTS[sfp_oracle_parameter_name(name)] == 0.175
    assert ORACLE_DEFAULTS[sc_oracle_parameter_name(name)] == 0.05


def test_sc_impedance_is_firmer_than_sfp_default():
    stiffness = "oracle_cartesian_stiffness"
    damping = "oracle_cartesian_damping"

    assert SFP_ORACLE_DEFAULTS[stiffness] == [
        100.0,
        100.0,
        100.0,
        50.0,
        50.0,
        50.0,
    ]
    assert SC_ORACLE_DEFAULTS[stiffness] == [
        300.0,
        300.0,
        300.0,
        80.0,
        80.0,
        80.0,
    ]
    assert SC_ORACLE_DEFAULTS[damping] == [
        70.0,
        70.0,
        70.0,
        25.0,
        25.0,
        25.0,
    ]
    assert ORACLE_DEFAULTS[sc_oracle_parameter_name(stiffness)] == (
        SC_ORACLE_DEFAULTS[stiffness]
    )
    assert ORACLE_DEFAULTS[sc_oracle_parameter_name(damping)] == (
        SC_ORACLE_DEFAULTS[damping]
    )
