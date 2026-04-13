# Data Collection with CheatCodeDataCollector

Records ground-truth CheatCode trajectories in simulation for offline training.
Requires `ground_truth:=true` so the simulation publishes TF frames for the
plug and port.

Each completed `insert_cable()` call saves one episode to disk. After
collection, run the converter to produce a LeRobot v3.0 dataset.

**Action space recorded:** 7D absolute TCP position target `[x, y, z, qx, qy, qz, qw]`
Compatible with: RunACT, RunRLT (XVLA / Pi0.5 backends)

---

## How it works

`CheatCodeDataCollector` subclasses `CheatCode` and uses a **move_robot wrapper**
to record data without modifying the policy itself.

The AIC framework calls `insert_cable(task, get_observation, move_robot, send_feedback)`
passing `move_robot` as a callback — a plain function the policy should call whenever
it wants to send a pose command to the robot.

`CheatCodeDataCollector` intercepts this by substituting its own wrapper:

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

---

## Step 1 — Generate a randomized config

The default eval environment runs only **3 trials** before stopping. Generate
a randomized config with as many trials as you need (3 scenarios × N):

```bash
pixi run python aic_utils/sym_data/generate_data_collection_config.py --episodes_per_scenario 100 --output /tmp/data_collection_config.yaml --seed 42
```

See `aic_utils/sym_data/generate_data_collection_config.py --help` for all options.

---

## Step 2 — Start the eval container with ground truth

```bash
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=true start_aic_engine:=true "aic_engine_config_file:=/tmp/data_collection_config.yaml"
```

Wait until the terminal shows `Retrying connection to aic_engine...` before
starting the collector.

---

## Step 3 — Run CheatCodeDataCollector

In a second terminal:

```bash
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.CheatCodeDataCollector \
    -p output_dir:=/home/yifeng/aic_data_raw
```

The collector runs continuously. Each episode is saved automatically when
`insert_cable()` returns. Stop with Ctrl-C when you have enough episodes.

### Output layout

```
/home/yifeng/aic_data_raw/
  dataset_index.jsonl          # one JSON line per episode
  episodes/
    <episode_id>/
      meta.json                # episode metadata (num_steps, success, task)
      data.parquet             # per-step state + action (flat columns)
      images/
        left_camera/           # 000000.jpg, 000001.jpg, ...
        center_camera/
        right_camera/
```

### Environment variable shortcut

Set `AIC_DATA_DIR` to avoid passing `output_dir` every time:

```bash
export AIC_DATA_DIR=/home/yifeng/aic_data_raw
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.CheatCodeDataCollector
```

---

## Step 4 — Convert to LeRobot v3.0

```bash
pixi run python aic_utils/sym_data/convert_cheatcode_to_lerobot.py \
    --input_dir /home/yifeng/aic_data_raw \
    --output_dir /home/yifeng/aic_data_sym
```

By default only successful episodes are converted. To include failed episodes:

```bash
pixi run python aic_utils/sym_data/convert_cheatcode_to_lerobot.py \
    --input_dir /home/yifeng/aic_data_raw \
    --output_dir /home/yifeng/aic_data_sym \
    --no-only_successful
```

Output dataset: `~/aic_data_sym/cheatcode/cable_insertion/`

---

## Step 5 — Train

```bash
# ACT
pixi run train-act

# RLT (see aic_utils/aic_rlt/README.md for full workflow)
pixi run python aic_utils/aic_rlt/scripts/train.py \
    --dataset_root /home/yifeng/aic_data_sym
```

---

## Checking collected data

Inspect episode count and success rate from the index:

```bash
python3 -c "
import json
from pathlib import Path
lines = Path('/home/yifeng/aic_data_raw/dataset_index.jsonl').read_text().splitlines()
eps = [json.loads(l) for l in lines if l]
n_ok = sum(1 for e in eps if e['success'])
print(f'{len(eps)} episodes, {n_ok} successful ({100*n_ok//len(eps)}%)')
"
```

Spot-check a converted episode:

```bash
pixi run python -c "import pandas as pd; df = pd.read_parquet('/home/yifeng/aic_data_sym/data/chunk-000/file-000.parquet'); print(df.columns.tolist()); print(len(df), 'rows')"
```
