from dataclasses import dataclass


AUTO_VALUE = "auto"
ALL_VALUE = "all"


@dataclass(frozen=True)
class PlugConfiguration:
    port_type: str
    plug_type: str
    plug_name: str
    cable_type: str


@dataclass(frozen=True)
class TargetConfiguration:
    port_type: str
    target_module_name: str
    port_name: str


PLUG_CONFIGURATIONS = {
    "sfp": PlugConfiguration(
        port_type="sfp",
        plug_type="sfp",
        plug_name="sfp_tip",
        cable_type="sfp_sc_cable",
    ),
    "sc": PlugConfiguration(
        port_type="sc",
        plug_type="sc",
        plug_name="sc_tip",
        cable_type="sfp_sc_cable_reversed",
    ),
}

SFP_TARGET_MODULES = tuple(f"nic_card_mount_{index}" for index in range(5))
SFP_PORTS = ("sfp_port_0", "sfp_port_1")
SC_TARGET_MODULES = ("sc_port_0", "sc_port_1")
SC_PORTS = ("sc_port_base",)

TARGET_CONFIGURATIONS = tuple(
    TargetConfiguration("sfp", module, port)
    for module in SFP_TARGET_MODULES
    for port in SFP_PORTS
) + tuple(
    TargetConfiguration("sc", module, port)
    for module in SC_TARGET_MODULES
    for port in SC_PORTS
)


def normalize_auto(value):
    if isinstance(value, str) and value.strip().lower() == AUTO_VALUE:
        return AUTO_VALUE
    return value


def normalize_all(value):
    if isinstance(value, str) and value.strip().lower() == ALL_VALUE:
        return ALL_VALUE
    return value


def plug_configuration(port_type: str) -> PlugConfiguration:
    try:
        return PLUG_CONFIGURATIONS[port_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported port_type '{port_type}'. "
            f"Supported values: {supported_port_types()}"
        ) from exc


def infer_port_type(target_module_name: str, port_name: str) -> str:
    matches = {
        target.port_type
        for target in TARGET_CONFIGURATIONS
        if target.target_module_name == target_module_name
        and target.port_name == port_name
    }
    if len(matches) == 1:
        return matches.pop()
    if not matches:
        raise ValueError(
            "Cannot infer port_type for target "
            f"'{target_module_name}/{port_name}'. Supported targets: "
            f"{supported_targets()}"
        )
    raise ValueError(
        f"Target '{target_module_name}/{port_name}' matches multiple port types: "
        f"{', '.join(sorted(matches))}"
    )


def resolve_plug_field(port_type: str, name: str, value: str) -> str:
    value = normalize_auto(value)
    if value != AUTO_VALUE:
        return value
    config = plug_configuration(port_type)
    return getattr(config, name)


def resolve_cable_type(port_type: str, value: str) -> str:
    value = normalize_auto(value)
    expected = plug_configuration(port_type).cable_type
    if value == AUTO_VALUE:
        return expected
    if value != expected:
        raise ValueError(
            f"cable_type '{value}' is incompatible with {port_type} insertion; "
            f"expected '{expected}'"
        )
    return value


def matching_targets(
    port_type: str, target_module_name: str, port_name: str
) -> tuple[TargetConfiguration, ...]:
    port_type = normalize_auto(port_type)
    target_module_name = normalize_all(target_module_name)
    port_name = normalize_all(port_name)

    if port_type != AUTO_VALUE:
        plug_configuration(port_type)

    matches = []
    for target in TARGET_CONFIGURATIONS:
        if port_type != AUTO_VALUE and target.port_type != port_type:
            continue
        if (
            target_module_name != ALL_VALUE
            and target.target_module_name != target_module_name
        ):
            continue
        if port_name != ALL_VALUE and target.port_name != port_name:
            continue
        matches.append(target)

    if not matches:
        raise ValueError(
            "Unsupported target port selection "
            f"port_type='{port_type}', target_module_name='{target_module_name}', "
            f"port_name='{port_name}'. Supported targets: {supported_targets()}"
        )
    return tuple(matches)


def resolve_target(
    port_type: str, target_module_name: str, port_name: str, target_index: int = 0
) -> TargetConfiguration:
    targets = matching_targets(port_type, target_module_name, port_name)
    return targets[target_index % len(targets)]


def target_sequence_size(
    port_type: str, target_module_name: str, port_name: str
) -> int:
    return len(matching_targets(port_type, target_module_name, port_name))


def validate_task_values(values: dict[str, str]) -> None:
    port_type = values["port_type"]
    plug = plug_configuration(port_type)

    if values["plug_type"] != plug.plug_type:
        raise ValueError(
            f"plug_type '{values['plug_type']}' is incompatible with "
            f"port_type '{port_type}'; expected '{plug.plug_type}'"
        )
    if values["plug_name"] != plug.plug_name:
        raise ValueError(
            f"plug_name '{values['plug_name']}' is incompatible with "
            f"port_type '{port_type}'; expected '{plug.plug_name}'"
        )

    target = TargetConfiguration(
        port_type=port_type,
        target_module_name=values["target_module_name"],
        port_name=values["port_name"],
    )
    if target not in TARGET_CONFIGURATIONS:
        raise ValueError(
            "Unsupported target port configuration "
            f"'{target.target_module_name}/{target.port_name}' for "
            f"port_type '{target.port_type}'. Supported targets: "
            f"{supported_targets(port_type)}"
        )


def target_board_part(values_or_task) -> str:
    if isinstance(values_or_task, dict):
        return values_or_task["target_module_name"]
    return values_or_task.target_module_name


def supported_port_types() -> str:
    return ", ".join(sorted(PLUG_CONFIGURATIONS))


def supported_targets(port_type: str | None = None) -> str:
    targets = [
        f"{target.target_module_name}/{target.port_name}"
        for target in TARGET_CONFIGURATIONS
        if port_type is None or target.port_type == port_type
    ]
    return ", ".join(targets)
