# ch_milestones

Automated data collection for the AIC cable insertion task. This package runs an **OraclePolicy** that uses ground-truth TF frames to guide a robot arm through a four-stage insertion sequence, while recording every frame into per-stage [LeRobot](https://github.com/huggingface/lerobot) datasets.

## Quick Start

**1. Start the simulation** (in a separate terminal):

```bash
/entrypoint.sh start_aic_engine:=false ground_truth:=true
```

**2. Launch data collection**:

```bash
pixi run ros2 launch ch_milestones ch_milestone_data_collection.launch.py \
  episode_count:=500 \
  sfp_oracle_debug_pause_motion:=false \
  sc_oracle_debug_pause_motion:=false
```

This will run 500 episodes of cable insertion, resetting and randomizing the environment between each episode, and saving LeRobot datasets to `~/aic_results/ch_milestones_lerobot/`.

## Architecture

The launch file starts three ROS 2 nodes:

| Node | Role |
|------|------|
| `ch_milestone_data_producer` | Orchestrates episodes: resets the environment, manages `aic_model` lifecycle, sends `InsertCable` action goals, and records frames into LeRobot datasets |
| `ch_milestone_environment_resetter` | Exposes `/ch_milestones/reset_episode` and `/ch_milestones/clear_environment` services to spawn/randomize the task board and cable |
| `aic_model` | Lifecycle-managed policy executor. Loads `OraclePolicy`, receives `InsertCable` action goals, and publishes cartesian motion commands |

### Episode Lifecycle

```
┌─ Data Producer ─────────────────────────────────────┐
│  for each episode:                                  │
│    1. Call /ch_milestones/reset_episode              │
│    2. Configure + Activate aic_model lifecycle       │
│    3. Start recording (subscribe to obs + actions)   │
│    4. Send InsertCable action goal                   │
│    5. OraclePolicy runs four stages                  │
│    6. Wait for result, settle, finish recording      │
│    7. Deactivate + Cleanup aic_model lifecycle       │
│    8. Save per-stage LeRobot episodes                │
└─────────────────────────────────────────────────────┘
```

## OraclePolicy Stages

The oracle uses ground-truth TF to compute gripper targets that align the plug with the port. The insertion is decomposed into four stages:

| Stage | Description |
|-------|-------------|
| **Approach** | Translate to a hover position above the port (`oracle_approach_z_offset`). No orientation change (`slerp_fraction=0.0`). Uses base-frame offset model. |
| **Coarse Align** | At the hover height, slerp the gripper orientation to align the plug with the port while maintaining XY alignment. Uses base-frame offset model with rotation tolerance. |
| **Fine Align** | Minimum-jerk trajectory blending both position and orientation toward the port at `oracle_alignment_fine_align_z_offset`. Uses rigid offset model with integral gain for XY correction. |
| **Insert** | Descend incrementally in Z (step = `oracle_alignment_insert_step_meters`) until `z_offset < -0.015`, pushing the plug into the port. Uses rigid offset model with integral gain. |

### Offset Models

- **Base** (approach, coarse_align): `tcp_target = plug_goal + (current_tcp − current_plug)`. Assumes the plug-to-TCP vector is constant in the base frame.
- **Rigid** (fine_align, insert): `tcp_target = plug_goal − R_tcp @ tcp_to_plug_body`. Uses the body-frame TCP-to-plug offset rotated by the target orientation, which is more accurate when the gripper has been reoriented.

## Recording

Each stage records into its own LeRobot dataset under `<dataset_root>/<stage>/`:

```
~/aic_results/ch_milestones_lerobot/
├── approach/
├── coarse_align/
├── fine_align/
└── insert/
```

Each frame contains:
- **`observation.state`** (25D): TCP pose (7), TCP velocity (6), TCP error (6), joint positions (7)
- **`action`** (7D): target TCP pose (position + quaternion)
- **`observation.images.{left,center,right}_camera`**: RGB images (downscaled by `image_scale`)

Datasets support video encoding (`use_videos:=true`, default) or raw images, and can be resumed across runs.

### Insertion Event Trimming

When `trim_on_insertion_event:=true` (default), the insert stage is trimmed at the `/scoring/insertion_event` topic, keeping only `post_insertion_frames` frames after the event. This removes the tail where the robot is just holding position.

## Environment Randomization

When `randomize_scene:=true` (default), each reset randomizes:

| Parameter | Range |
|-----------|-------|
| Task board XY, SFP targets | `task_board_x_min/max`, `task_board_y_min/max`, default ± 0.05 m |
| Task board yaw, SFP targets | `task_board_yaw_min/max`, default ± ~1.0 rad |
| Task board pose, SC targets | `sc_task_board_*_min/max`, default same as SFP |
| Cable pose, SFP targets | `cable_*_min/max`, default no jitter |
| Cable pose, SC targets | `sc_cable_*_min/max`, default no jitter around the SC cable pose |
| Board part translations | part-specific ranges (e.g. `nic_card_mount`: -0.048 to 0.036) |
| SC port translations | shared `sc_port_translation_min/max` range, default -0.06 to 0.055 |

Randomization uses a configurable distribution (`normal` or `uniform`) controlled by `randomization_distribution`. Set `random_seed` to a positive value for reproducibility. SC targets use separate task-board and cable randomization parameters (`sc_task_board_*_min/max`, `sc_cable_*_min/max`) plus shared SC port parameters (`sc_port_translation_min/max`, `sc_port_roll_min/max`, `sc_port_pitch_min/max`, `sc_port_yaw_min/max`). The base scene values remain `task_board_*`, `cable_*`, `sc_port_0_*`, and `sc_port_1_*`.

## Configuration Reference

### Launch Arguments

#### Data Collection
| Argument | Default | Description |
|----------|---------|-------------|
| `episode_count` | `1` | Number of successful episodes to collect |
| `retry_failed_episodes` | `true` | Retry on failure instead of aborting |
| `max_episode_attempts` | `0` | Max retries per episode (0 = unlimited) |
| `reset_before_episode` | `true` | Reset environment before each episode |
| `clear_after_episodes` | `true` | Clear environment after all episodes |
| `dataset_root` | `~/aic_results/ch_milestones_lerobot` | Output directory |
| `repo_id` | `local/ch_milestones` | LeRobot dataset repo ID |
| `fps` | `20` | Recording frame rate |
| `image_scale` | `0.25` | Image downscale factor |
| `use_videos` | `true` | Encode images as video |
| `vcodec` | `h264` | Video codec |
| `recording_clock` | `wall` | Clock source (`wall` or `sim`) |
| `trim_on_insertion_event` | `true` | Trim insert stage at insertion event |
| `post_insertion_frames` | `10` | Frames to keep after insertion event |
| `require_insertion_event` | `true` | Fail if no insertion event during insert |

#### Task Target
| Argument | Default | Description |
|----------|---------|-------------|
| `port_type` | `auto` | Target port type. `auto` allows both SFP and SC targets |
| `target_module_name` | `all` | Component that owns the target port. `all` cycles every compatible component |
| `port_name` | `all` | Target port on the component. `all` cycles every compatible port |
| `plug_type` | `auto` | Plug type. `auto` selects `sfp` for SFP ports and `sc` for SC ports |
| `plug_name` | `auto` | Plug link prefix. `auto` selects `sfp_tip` or `sc_tip` |
| `cable_type` | `auto` | Cable model. `auto` selects `sfp_sc_cable` or `sfp_sc_cable_reversed` |
| `auto_cable_pose` | `true` | Use the default gripper-aligned cable pose for the inferred plug |

Supported fixed-target combinations:

| `target_module_name` | `port_name` | Inferred `port_type` | Inferred `plug_type` / `plug_name` | Inferred `cable_type` |
|----------------------|-------------|----------------------|-------------------------------------|-----------------------|
| `nic_card_mount_0` | `sfp_port_0` | `sfp` | `sfp` / `sfp_tip` | `sfp_sc_cable` |
| `nic_card_mount_0` | `sfp_port_1` | `sfp` | `sfp` / `sfp_tip` | `sfp_sc_cable` |
| `nic_card_mount_1` | `sfp_port_0` | `sfp` | `sfp` / `sfp_tip` | `sfp_sc_cable` |
| `nic_card_mount_1` | `sfp_port_1` | `sfp` | `sfp` / `sfp_tip` | `sfp_sc_cable` |
| `nic_card_mount_2` | `sfp_port_0` | `sfp` | `sfp` / `sfp_tip` | `sfp_sc_cable` |
| `nic_card_mount_2` | `sfp_port_1` | `sfp` | `sfp` / `sfp_tip` | `sfp_sc_cable` |
| `nic_card_mount_3` | `sfp_port_0` | `sfp` | `sfp` / `sfp_tip` | `sfp_sc_cable` |
| `nic_card_mount_3` | `sfp_port_1` | `sfp` | `sfp` / `sfp_tip` | `sfp_sc_cable` |
| `nic_card_mount_4` | `sfp_port_0` | `sfp` | `sfp` / `sfp_tip` | `sfp_sc_cable` |
| `nic_card_mount_4` | `sfp_port_1` | `sfp` | `sfp` / `sfp_tip` | `sfp_sc_cable` |
| `sc_port_0` | `sc_port_base` | `sc` | `sc` / `sc_tip` | `sfp_sc_cable_reversed` |
| `sc_port_1` | `sc_port_base` | `sc` | `sc` / `sc_tip` | `sfp_sc_cable_reversed` |

With the defaults, episodes cycle through the full supported target list. The resetter automatically marks the selected target component as present before spawning the board. For example:

```bash
# Full target sweep across all SFP and SC targets.
pixi run ros2 launch ch_milestones ch_milestone_data_collection.launch.py \
  episode_count:=500 \
  sfp_oracle_debug_pause_motion:=false \
  sc_oracle_debug_pause_motion:=false

# All SFP targets only.
pixi run ros2 launch ch_milestones ch_milestone_data_collection.launch.py \
  port_type:=sfp

# SFP port on the fifth NIC mount.
pixi run ros2 launch ch_milestones ch_milestone_data_collection.launch.py \
  target_module_name:=nic_card_mount_4 \
  port_name:=sfp_port_1

# SC port on the second SC rail.
pixi run ros2 launch ch_milestones ch_milestone_data_collection.launch.py \
  target_module_name:=sc_port_1 \
  port_name:=sc_port_base
```

If a launch still appears to use old defaults after editing this package, reinstall the local Pixi package before launching:

```bash
pixi reinstall ros-kilted-ch-milestones
```

#### Oracle Policy
| Argument | Default | Description |
|----------|---------|-------------|
| `sfp_oracle_speed_scale` | `0.5` | Speed multiplier (affects step counts) |
| `sfp_oracle_command_period` | `0.05` | Seconds between motion commands |
| `sfp_oracle_stage_timeout_seconds` | `60.0` | Per-stage wall-time timeout |
| `sfp_oracle_cartesian_stiffness` | `[100,100,100,50,50,50]` | Impedance stiffness |
| `sfp_oracle_cartesian_damping` | `[40,40,40,15,15,15]` | Impedance damping |
| `sfp_oracle_approach_z_offset` | `0.15` | Hover height above port (m) |
| `sfp_oracle_approach_step_meters` | `0.001` | Max plug step per command cycle (m) |
| `sfp_oracle_approach_success_tolerance_meters` | `0.010` | Approach success distance (m) |
| `sfp_oracle_coarse_align_step_meters` | `0.001` | Coarse align step size (m) |
| `sfp_oracle_coarse_align_success_tolerance_meters` | `0.010` | Coarse align position tolerance (m) |
| `sfp_oracle_coarse_align_success_rotation_tolerance_radians` | `0.08` | Coarse align rotation tolerance (rad) |
| `sfp_oracle_alignment_fine_align_z_offset` | `0.175` | Fine-align height above port (m) |
| `sfp_oracle_alignment_integral_gain` | `0.12` | XY integrator gain (fine align / insert) |
| `sfp_oracle_alignment_integrator_limit` | `0.05` | XY integrator anti-windup limit (m) |
| `sfp_oracle_insert_end_z_offset` | `-0.015` | Final Z offset for insertion (m) |
| `sfp_oracle_final_settle_steps` | `50` | Settle steps after insert |

SC targets use a separate oracle parameter set with an `sc_` prefix. For example, `sc_oracle_speed_scale`, `sc_oracle_approach_z_offset`, and `sc_oracle_alignment_fine_align_z_offset` apply only to SC episodes. Most defaults match the SFP values unless overridden.

The SC fine-align height defaults lower than SFP: `sc_oracle_alignment_fine_align_z_offset:=0.05`.

The SC Cartesian impedance also defaults firmer to resist cable drag: `sc_oracle_cartesian_stiffness:=[300,300,300,80,80,80]` and `sc_oracle_cartesian_damping:=[70,70,70,25,25,25]`.

During a mixed target sweep, SFP episodes use `sfp_oracle_*` and SC episodes use `sc_oracle_*`. Legacy `oracle_*` launch arguments are still accepted as SFP aliases when the matching `sfp_oracle_*` value is left at its default.

#### Debug
| Argument | Default | Description |
|----------|---------|-------------|
| `sfp_oracle_publish_debug_frames` | `true` | Publish debug TF frames |
| `sfp_oracle_debug_frame_prefix` | `oracle_debug` | TF frame prefix |
| `sfp_oracle_debug_log_frequency_hz` | `3.0` | Diagnostic log rate |
| `sfp_oracle_debug_pause_motion` | `false` | Pause before each motion command |
| `sfp_oracle_debug_pause_before_motion_seconds` | `0.0` | Hold time before each command |

Debug parameters also have SC-specific forms, such as `sc_oracle_debug_pause_motion` and `sc_oracle_debug_frame_prefix`.

### Debug TF Frames

When debug frames are enabled, the following frames are broadcast under the configured prefix (default `oracle_debug`):

| Frame | Description |
|-------|-------------|
| `oracle_debug/moving_frame` | Current TCP position |
| `oracle_debug/target_frame` | Commanded TCP target |
| `oracle_debug/moving_plug_frame` | Current plug position |
| `oracle_debug/target_plug_frame` | Predicted plug position after command |
| `oracle_debug/goal_plug_frame` | Stage goal for the plug |
| `oracle_debug/reference_frame` | Reference position (port + z_offset) |

## Package Structure

```
ch_milestones/
├── launch/
│   └── ch_milestone_data_collection.launch.py
└── ch_milestones/
    ├── config/
    │   ├── policy_config.py          # Oracle parameter defaults & STAGES
    │   ├── randomization_config.py   # Jitter ranges & distribution config
    │   ├── scene_config.py           # Task board, cable, board part defaults
    │   └── task_config.py            # Task message construction
    ├── environment/
    │   ├── description.py            # URDF/xacro loading
    │   ├── entities.py               # Task board & cable entity spawning
    │   ├── randomization.py          # Pose randomization logic
    │   ├── reset_client.py           # Service client for episode resets
    │   ├── resetter.py               # Environment spawn/reset/clear orchestration
    │   ├── robot.py                  # Robot homing via joint commands
    │   └── services.py               # Simulation service helpers
    ├── nodes/
    │   ├── data_producer.py          # Episode loop & LeRobot recording
    │   └── environment_resetter.py   # Reset/clear service node
    ├── policies/
    │   ├── OraclePolicy.py           # Main policy entry point
    │   ├── ground_truth_guidance.py   # TF-based gripper pose computation
    │   ├── oracle_motion.py          # Motion commanding & plug-space stepping
    │   ├── oracle_stage_base.py      # Stage base class
    │   ├── oracle_stages.py          # Stage set (approach → insert)
    │   ├── oracle_frames.py          # Frame references (port, plug, orientation)
    │   ├── oracle_debug_frames.py    # Debug TF broadcasting
    │   ├── oracle_validation.py      # Parameter validation
    │   ├── cartesian_trajectory.py   # Pose interpolation & minimum jerk
    │   ├── stage_approach.py         # Approach stage implementation
    │   ├── stage_coarse_align.py     # Coarse alignment stage
    │   ├── stage_fine_align.py       # Fine alignment stage
    │   └── stage_insert.py           # Insertion stage
    └── recording/
        ├── stage_episode_recorder.py # Per-stage LeRobot recording
        ├── lerobot_format.py         # State/action/image feature definitions
        └── episode_trim.py           # Insertion event trimming
```
