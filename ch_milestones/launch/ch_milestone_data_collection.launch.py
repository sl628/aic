
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from ch_milestones.config.policy_config import ORACLE_DEFAULTS
from ch_milestones.config.randomization_config import RANDOMIZATION_DEFAULTS
from ch_milestones.config.scene_config import (
    BOARD_PART_DEFAULTS,
    CABLE_DEFAULTS,
    SPAWN_DEFAULTS,
    TASK_BOARD_DEFAULTS,
)


def argument(name, default):
    value = str(default).lower() if isinstance(default, bool) else str(default)
    return DeclareLaunchArgument(name, default_value=value)


def typed(name, value_type):
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def parameter(name, default):
    if isinstance(default, (bool, int, float)):
        return typed(name, type(default))
    return LaunchConfiguration(name)


def task_parameters():
    return {
        "task_id": LaunchConfiguration("task_id"),
        "task_cable_type": LaunchConfiguration("task_cable_type"),
        "cable_name": LaunchConfiguration("cable_name"),
        "plug_type": LaunchConfiguration("plug_type"),
        "plug_name": LaunchConfiguration("plug_name"),
        "port_type": LaunchConfiguration("port_type"),
        "port_name": LaunchConfiguration("port_name"),
        "target_module_name": LaunchConfiguration("target_module_name"),
        "time_limit": typed("time_limit", int),
    }


def scene_parameters():
    params = {
        "description_root": LaunchConfiguration("description_root"),
        "assets_root": LaunchConfiguration("assets_root"),
        "reset_timeout_seconds": typed("reset_timeout_seconds", float),
        "tf_timeout_seconds": typed("tf_timeout_seconds", float),
        "post_spawn_settle_seconds": typed("post_spawn_settle_seconds", float),
    }
    for name, default in SPAWN_DEFAULTS.items():
        params[name] = parameter(name, default)
    for name, default in TASK_BOARD_DEFAULTS.items():
        params[name] = parameter(name, default)
    for name, default in CABLE_DEFAULTS.items():
        params[name] = parameter(name, default)
    for name in BOARD_PART_DEFAULTS:
        params[f"{name}_present"] = typed(f"{name}_present", bool)
        params[f"{name}_translation"] = typed(f"{name}_translation", float)
        params[f"{name}_roll"] = typed(f"{name}_roll", float)
        params[f"{name}_pitch"] = typed(f"{name}_pitch", float)
        params[f"{name}_yaw"] = typed(f"{name}_yaw", float)
    return params


def randomization_parameters():
    return {
        name: parameter(name, value) for name, value in RANDOMIZATION_DEFAULTS.items()
    }


def oracle_parameters():
    return {name: parameter(name, value) for name, value in ORACLE_DEFAULTS.items()}


def scene_arguments():
    return [
        argument("description_root", "./aic_description"),
        argument("assets_root", "./aic_assets"),
        *(argument(name, value) for name, value in SPAWN_DEFAULTS.items()),
        *(argument(name, value) for name, value in TASK_BOARD_DEFAULTS.items()),
        *(argument(name, value) for name, value in CABLE_DEFAULTS.items()),
        argument("tf_timeout_seconds", 30.0),
        argument("post_spawn_settle_seconds", 1.0),
    ]


def board_part_arguments():
    args = []
    for name, values in BOARD_PART_DEFAULTS.items():
        present, translation, roll, pitch, yaw = values
        args += [
            argument(f"{name}_present", present),
            argument(f"{name}_translation", translation),
            argument(f"{name}_roll", roll),
            argument(f"{name}_pitch", pitch),
            argument(f"{name}_yaw", yaw),
        ]
    return args


def randomization_arguments():
    return [argument(name, value) for name, value in RANDOMIZATION_DEFAULTS.items()]


def oracle_arguments():
    return [argument(name, value) for name, value in ORACLE_DEFAULTS.items()]


def generate_launch_description():
    producer = Node(
        package="ch_milestones",
        executable="ch_milestone_data_producer",
        name="ch_milestone_data_producer",
        output="screen",
        parameters=[
            {
                "use_sim_time": typed("use_sim_time", bool),
                "dataset_root": LaunchConfiguration("dataset_root"),
                "repo_id": LaunchConfiguration("repo_id"),
                "fps": typed("fps", int),
                "task_prompt": LaunchConfiguration("task_prompt"),
                "image_width": typed("image_width", int),
                "image_height": typed("image_height", int),
                "image_scale": typed("image_scale", float),
                "use_videos": typed("use_videos", bool),
                "vcodec": LaunchConfiguration("vcodec"),
                "recording_clock": LaunchConfiguration("recording_clock"),
                "observation_timeout_seconds": typed(
                    "observation_timeout_seconds", float
                ),
                "settle_seconds": typed("settle_seconds", float),
                "action_timeout_padding_seconds": typed(
                    "action_timeout_padding_seconds", float
                ),
                "episode_count": typed("episode_count", int),
                "retry_failed_episodes": typed("retry_failed_episodes", bool),
                "max_episode_attempts": typed("max_episode_attempts", int),
                "reset_before_episode": typed("reset_before_episode", bool),
                "clear_after_episodes": typed("clear_after_episodes", bool),
                "reset_service_timeout_seconds": typed(
                    "reset_service_timeout_seconds", float
                ),
                "insertion_event_topic": LaunchConfiguration("insertion_event_topic"),
                "trim_on_insertion_event": typed("trim_on_insertion_event", bool),
                "post_insertion_frames": typed("post_insertion_frames", int),
                "require_insertion_event": typed("require_insertion_event", bool),
                **task_parameters(),
            }
        ],
    )
    resetter = Node(
        package="ch_milestones",
        executable="ch_milestone_environment_resetter",
        name="ch_milestone_environment_resetter",
        output="screen",
        parameters=[
            {
                "use_sim_time": typed("use_sim_time", bool),
                **task_parameters(),
                **scene_parameters(),
                **randomization_parameters(),
            }
        ],
    )
    model = Node(
        package="aic_model",
        executable="aic_model",
        name="aic_model",
        output="screen",
        parameters=[
            {
                "use_sim_time": typed("use_sim_time", bool),
                "policy": LaunchConfiguration("policy"),
                **oracle_parameters(),
            }
        ],
    )
    return LaunchDescription(
        [
            argument("use_sim_time", "true"),
            argument("policy", "ch_milestones.policies.OraclePolicy"),
            argument("dataset_root", "~/aic_results/ch_milestones_lerobot"),
            argument("repo_id", "local/ch_milestones"),
            argument("fps", 20),
            argument("task_prompt", ""),
            argument("image_width", 1152),
            argument("image_height", 1024),
            argument("image_scale", 0.25),
            argument("use_videos", "true"),
            argument("vcodec", "h264"),
            argument("recording_clock", "wall"),
            argument("observation_timeout_seconds", 30.0),
            argument("settle_seconds", 1.0),
            argument("action_timeout_padding_seconds", 15.0),
            argument("episode_count", 1),
            argument("retry_failed_episodes", "true"),
            argument("max_episode_attempts", 0),
            argument("reset_before_episode", "true"),
            argument("clear_after_episodes", "true"),
            argument("reset_service_timeout_seconds", 300.0),
            argument("insertion_event_topic", "/scoring/insertion_event"),
            argument("trim_on_insertion_event", "true"),
            argument("post_insertion_frames", 10),
            argument("require_insertion_event", "true"),
            argument("reset_timeout_seconds", 30.0),
            argument("task_id", "task_1"),
            argument("task_cable_type", "sfp_sc"),
            argument("cable_name", "cable_0"),
            argument("plug_type", "auto"),
            argument("plug_name", "auto"),
            argument("port_type", "auto"),
            argument("port_name", "all"),
            argument("target_module_name", "all"),
            argument("time_limit", 180),
            *scene_arguments(),
            *board_part_arguments(),
            *randomization_arguments(),
            *oracle_arguments(),
            resetter,
            model,
            producer,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=producer,
                    on_exit=[
                        EmitEvent(
                            event=Shutdown(
                                reason="ch_milestone_data_producer exited"
                            )
                        )
                    ],
                )
            ),
        ]
    )
