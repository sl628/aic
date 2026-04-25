# aic_xvla

End-to-end fine-tune of [X-VLA](https://github.com/2toinf/X-VLA) on aic data + inference wrapper for eval. Tracks issue [sl628/aic#14](https://github.com/sl628/aic/issues/14).

> Uses X-VLA's **official** training and inference path. **Do not** use `aic_utils/aic_rlt/aic_rlt/vla/xvla_wrapper.py` — that wrapper is for the old RLT (frozen feature-extractor) approach, which is a different task.

## Layout

| File | Purpose |
|---|---|
| `aic_xvla/handler.py` | `AICHandler` — X-VLA `DomainHandler` for aic flat-parquet schema. Maps 26D state + 7D TCP actions → X-VLA `ee6d` (20D) format. |
| `aic_xvla/build_meta.py` | Generates `meta.json` (X-VLA "general style") from a glob of per-episode parquets. |
| `aic_xvla/train.py` | Thin wrapper around X-VLA's `train.py` — registers `AICHandler` then forwards CLI args. |
| `aic_xvla/eval.py` | `AICXVLAPolicy` — loads checkpoint, predicts action chunk, converts X-VLA 20D output back to aic 7D `(pos, quat_xyzw)`. |

## Environment

X-VLA pins `pytorch=2.1` + `numpy=1.26`, which conflicts with the aic pixi env (`torch 2.7.1+cu128`). Use **X-VLA's own conda env**, not pixi:

```bash
cd ~/workspace/X-VLA
conda env create -f environment.yml -n xvla
conda activate xvla
pip install -r requirements.txt
pip install -e ~/workspace/aic/aic_utils/aic_xvla   # install our adapter into xvla env
```

`~/workspace/X-VLA` is read-only by convention — install `aic_xvla` as an editable package into the env, then put the X-VLA repo on `PYTHONPATH` at run time.

## Data prep

The colleague is uploading the corrected dataset. Reference one-episode dataset for prototyping: <https://huggingface.co/datasets/siyulw2025/aic_data_one_episode>.

Expected on-disk layout (per-episode parquet, sibling `images/` tree):

```
<root>/
  episodes/<episode_id>/data.parquet         # state_0..25, action_0..6, image_path_*
  episodes/<episode_id>/images/<cam>/*.jpg
```

Generate the X-VLA meta:

```bash
python -m aic_xvla.build_meta \
    --parquet-glob '/path/to/aic/episodes/*/data.parquet' \
    --image-root   /path/to/aic \
    --instruction  "insert the SFP cable into the port" \
    --fps 20 \
    --out /path/to/aic_meta.json
```

## Train

X-VLA ships two official fine-tune entry points; the wrapper exposes both via `--mode`:

| `--mode` | Calls | Trainable params | When to use |
|---|---|---|---|
| `full` (default) | `train.py` | all (~890M) | full FT, multi-GPU or ≥40 GB VRAM |
| `peft` | `peft_train.py` | LoRA only (~12M) | single-GPU / consumer cards, overfit demos |

Run from inside the X-VLA env, with both repos on `PYTHONPATH`:

```bash
export XVLA_REPO=~/workspace/X-VLA
export PYTHONPATH=$XVLA_REPO:$PYTHONPATH
```

**LoRA fine-tune (fits on a 16 GB card with bf16):**

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --num_processes 1 --mixed_precision bf16 -m aic_xvla.train \
    --mode peft \
    --models 2toINF/X-VLA-Pt \
    --train_metas_path /path/to/aic_meta.json \
    --output_dir runnings/aic_xvla \
    --batch_size 1 \
    --learning_rate 5e-4 \
    --iters 3000 \
    --warmup_steps 50 \
    --freeze_steps 0 \
    --save_interval 1000
```

**Full fine-tune (needs more VRAM):**

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 -m aic_xvla.train \
    --mode full \
    --models 2toINF/X-VLA-Pt \
    --train_metas_path /path/to/aic_meta.json \
    --output_dir runnings/aic_xvla \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --iters 100000 \
    --warmup_steps 2000 \
    --freeze_steps 1000 \
    --save_interval 10000 \
    --use_cosine_decay
```

Checkpoints land in `runnings/aic_xvla/ckpt-<step>/`.

## W&B

X-VLA logs to **tensorboard** (via `Accelerator(log_with="tensorboard")`); it does not call `wandb` directly. The wrapper bridges the gap by calling `wandb.init(sync_tensorboard=True)` before X-VLA writes any tfevent — wandb then captures every event live.

**Live mirror** — pass `--wandb-project` (and optionally `--wandb-entity`, `--wandb-run-name`) to `aic_xvla.train`:

```bash
pip install wandb            # one-time, in the X-VLA env
wandb login                  # one-time

accelerate launch --num_processes 1 --mixed_precision bf16 -m aic_xvla.train \
    --mode peft \
    --wandb-project aic-xvla --wandb-entity <your-entity> --wandb-run-name overfit-1ep \
    --models 2toINF/X-VLA-Pt \
    ...                                # rest of training args as above
```

**Post-hoc upload** — for a run launched without `--wandb-project`:

```bash
wandb sync --sync-tensorboard runnings/aic_xvla/XVLA-Training \
    --project aic-xvla --entity <your-entity>
```

## Eval — offline replay (sanity check, no simulator)

Predict actions for sampled frames of an episode and compare to ground-truth:

```bash
PYTHONPATH=$XVLA_REPO python -m aic_xvla.replay \
    --checkpoint /home/yifeng/aic_xvla_overfit/ckpt-3000 \
    --meta /home/yifeng/aic_data_one_ep/aic_meta.json \
    --sample-every 50 --horizon 10
```

Reports per-axis pos MAE (m) and quat-angle MAE (deg). On the verified 1-episode overfit: 14 mm / 0.87°.

## Eval — closed-loop in the `aic_eval` simulator

Three terminals. The X-VLA inference runs in its own conda env (torch 2.11), the ROS policy runs in the aic pixi env (torch 2.7), and they talk over HTTP.

**Terminal 1 — start the inference server (X-VLA env):**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate XVLA
pip install fastapi uvicorn pydantic   # one-time
export XVLA_REPO=~/workspace/X-VLA
PYTHONPATH=$XVLA_REPO CUDA_VISIBLE_DEVICES=0 \
    python -m aic_xvla.serve \
        --checkpoint /home/yifeng/aic_xvla_overfit/ckpt-3000 \
        --host 0.0.0.0 --port 8010
# verify with:  curl http://127.0.0.1:8010/healthz
```

**Terminal 2 — start the eval container + engine (host shell):**

```bash
export DBX_CONTAINER_MANAGER=docker
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval
distrobox enter -r aic_eval
# inside the container:
/entrypoint.sh ground_truth:=false start_aic_engine:=true
# wait for Gazebo + RViz + "No node with name 'aic_model' found. Retrying..."
```

**Terminal 3 — start the aic_model node with our policy (aic pixi env):**

```bash
cd ~/workspace/aic
pip install requests   # one-time, into the pixi env
pip install -e .claude/worktree-xvla-finetune/aic_utils/aic_xvla   # adapter package
export AIC_XVLA_SERVER_URL=http://127.0.0.1:8010
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_xvla.ros.RunXVLA.RunXVLA
```

The policy will:
- POST each `Observation` to `/act` (state[26] + 3 base64 JPEGs + instruction)
- get back `[T, 7]` absolute TCP poses (pos + quat_xyzw)
- replan every step (`AIC_XVLA_REPLAN=1`); execute via `set_pose_target` until task timeout (`AIC_XVLA_TASK_TIMEOUT_S=60`).

Tunable env vars: `AIC_XVLA_SERVER_URL`, `AIC_XVLA_REPLAN`, `AIC_XVLA_TIMEOUT_S`, `AIC_XVLA_TASK_TIMEOUT_S`, `AIC_XVLA_CONTROL_PERIOD_S`.

> ⚠️ With a 1-episode overfit checkpoint, closed-loop success is unlikely unless the simulator spawns the cable+port at the exact demo poses. Treat this as wiring validation; meaningful success rates need the full multi-episode dataset.

## Status (issue #14 checklist)

- [x] X-VLA recipe studied; integration shape pinned
- [x] Adapter package scaffolded
- [x] Overfit run on the 1-episode HF dataset (loss 24.2 → 0.017 in <500 steps)
- [x] W&B live mirror via `--wandb-project`
- [x] Offline replay eval (14 mm / 0.87° on training episode)
- [x] HTTP server + ROS policy for closed-loop wiring
- [ ] Wait on colleague's full dataset upload
- [ ] Closed-loop rollouts in the aic engine vs pi05 / ACT baselines
