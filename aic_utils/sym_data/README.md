# sym_data — Synthetic & Sim Data Collection

Two paths for generating training data for the AIC cable insertion task.

## Path A — Pure Synthetic (no ROS, no sim)

Generates procedural trajectories with randomised port positions and simulated noise.
Use this to bootstrap training before real or sim data is available.

```bash
pixi run python aic_utils/sym_data/generate_synthetic.py \
    --output_dir ~/aic_data \
    --num_episodes 200
```

Output: `~/aic_data/synthetic/cable_insertion/` (LeRobot v3.0)

**Action space**: 7D absolute TCP position target `[x, y, z, qx, qy, qz, qw]`
Compatible with: RunACT, RunRLT (XVLA/Pi0.5 backends)

---

## Path B — Sim Data via CheatCode (requires ROS + eval container)

Records ground-truth CheatCode trajectories in simulation, then converts to LeRobot format.

### How CheatCodeDataCollector works

`CheatCodeDataCollector` subclasses `CheatCode` and uses a **move_robot wrapper**
to record data without modifying the policy itself.

The AIC framework calls `insert_cable(task, get_observation, move_robot, send_feedback)`
passing `move_robot` as a callback — a function the policy calls whenever it wants
to send a pose command to the robot. `CheatCodeDataCollector` intercepts this by
substituting its own wrapper:

```
CheatCodeDataCollector.insert_cable()
│
├── define recording_move_robot(pose):
│       ├── obs = get_observation()    # snapshot state at this exact moment
│       ├── steps.append(state, pose)  # record (state, action) pair + images
│       └── real_move_robot(pose)      # forward to the actual robot controller
│
└── super().insert_cable(..., move_robot=recording_move_robot)
        │   CheatCode runs its approach + insertion loop (~530 steps)
        │   Each step: reads TF → computes pose → calls recording_move_robot()
        └── returns success
```

The wrapper is the only place where both pieces exist simultaneously:
- **The action** — the pose CheatCode just computed from TF
- **The state** — what the robot looked like at that exact moment

`CheatCode` never calls `get_observation()` itself (it reads TF directly), so the
wrapper must fetch it. `CheatCode` has no knowledge of recording — it just calls
whatever `move_robot` function it was handed.

### Step 1: Generate a randomized config

The default eval environment runs only **3 trials** before stopping. Generate
a randomized config with as many trials as you need (3 scenarios × N):

```bash
pixi run python aic_utils/sym_data/generate_data_collection_config.py --episodes_per_scenario 100 --output /tmp/data_collection_config.yaml --seed 42
```

### Step 2: Collect raw episodes

**Terminal 1** — start the eval container with ground truth TF frames:
```bash
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=true start_aic_engine:=true "aic_engine_config_file:=/tmp/data_collection_config.yaml"
```

**Terminal 2** — run the data collector:
```bash
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.CheatCodeDataCollector -p output_dir:=/home/yifeng/aic_data_raw
```

Each completed `insert_cable()` call saves one episode to `~/aic_data_raw/`.

### Step 2: Convert to LeRobot format

```bash
pixi run python aic_utils/sym_data/convert_cheatcode_to_lerobot.py \
    --input_dir /home/yifeng/aic_data_raw \
    --output_dir /home/yifeng/aic_data_sym
```

By default only successful episodes are kept (`--only_successful`, on by default).
Use `--no-only_successful` to include failed episodes.

Output: `~/aic_data_sym/cheatcode/cable_insertion/` (LeRobot v3.0)

**Action space**: 7D absolute TCP position target `[x, y, z, qx, qy, qz, qw]`
Compatible with: RunACT, RunRLT (XVLA/Pi0.5 backends)

---

## Path C — Autonomous Teleop via lerobot record (requires ROS + eval container)

Uses `AICCheatCodeTeleop` (in `lerobot_robot_aic/aic_teleop.py`) as a lerobot
`Teleoperator` with `lerobot record`. Data is written directly to LeRobot v3.0
format — no converter needed.

```bash
pixi run lerobot record \
    --robot.type=aic_controller \
    --teleop.type=aic_cheatcode \
    --dataset.repo_id=cheatcode_teleop/cable_insertion \
    --dataset.root=~/aic_data_teleop
```

**Action space**: 6D TCP velocity `[vx, vy, vz, wx, wy, wz]` (MODE_VELOCITY)
Note: This differs from Path A/B. Use Path A or B if training RunACT/RunRLT,
which expect 7D position targets.

---

## Choosing a path

| | Path A | Path B | Path C |
|---|---|---|---|
| Requires sim | No | Yes | Yes |
| Action space | Position (7D) | Position (7D) | Velocity (6D) |
| Compatible with RunRLT | Yes | Yes | No (different action space) |
| Dataset format | LeRobot v3.0 | LeRobot v3.0 | LeRobot v3.0 |
| Extra steps | None | Collect + convert | None |
