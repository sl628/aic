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

X-VLA logs to **tensorboard** (via `Accelerator(log_with="tensorboard")`); it does not call `wandb` directly. Two options:

**Live mirror (recommended)** — start a W&B run that tails the tensorboard event file:

```bash
wandb login                                             # one-time
wandb init -p aic-xvla -e <your-entity>                 # creates wandb/run dir
wandb sync --sync-tensorboard runnings/aic_xvla &       # background
# then launch training as above
```

**Post-hoc upload** — after a run finishes:

```bash
wandb sync --sync-tensorboard runnings/aic_xvla --project aic-xvla --entity <your-entity>
```

If you prefer native W&B logging, patch X-VLA `train.py` `Accelerator(log_with=["tensorboard","wandb"])` and `accelerator.init_trackers("XVLA-Training", config=vars(args), init_kwargs={"wandb": {"project": "aic-xvla"}})` — kept out of this wrapper to avoid forking X-VLA.

## Eval

Smoke test:

```bash
python -m aic_xvla.eval \
    --checkpoint runnings/aic_xvla/ckpt-100000 \
    --left  /tmp/left.jpg \
    --center /tmp/center.jpg \
    --right /tmp/right.jpg
```

For closed-loop rollouts in the aic engine, use `AICXVLAPolicy` from `aic_xvla.eval` and wire it behind the `aic_model.policy.Policy` interface. The policy returns 7D TCP `(x,y,z,qx,qy,qz,qw)` per timestep — same shape as the dataset action.

## Status (issue #14 checklist)

- [x] X-VLA recipe studied; integration shape pinned
- [x] Adapter package scaffolded
- [ ] Wait on colleague's full dataset upload
- [ ] Run end-to-end fine-tune
- [ ] Eval rollouts vs pi05 / ACT baselines
