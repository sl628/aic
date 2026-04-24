# Communication
- Be extremely concise in all responses and commit messages; sacrifice grammar for brevity.

# Git
- Branch names: prefix with `yf_` (e.g. `yf_fix-auth`).
- Commits: one commit per feature/major change; never bundle unrelated changes.
- Push to origin (sl628/aic), not upstream (intrinsic-dev/aic).
- PRs target `main` branch.

# Project Structure
- ROS 2 Kilted workspace managed by Pixi (conda-based).
- Key packages: aic_model (participant policy), aic_engine (trial orchestrator), aic_controller, aic_adapter, aic_rlt (RL training).
- Python code: `<pkg>/<pkg>/*.py`.
- C++ code: Google style (`.clang-format` at repo root).

# Environment
- Use `pixi install` to set up (NOT pip install).
- Use `pixi run <cmd>` to run commands in the environment.
- Never modify pixi.lock manually.
- Do not add dependencies without updating pixi.toml.

# Style & Linting
- Python: `black .` before committing.
- Python imports: `isort`.
- C/C++: `clang-format` (v19, Google style).
- Type checking: `pyright` (basic mode).

# Testing
- CI: build.yml, style.yml, pixi.yml on every push/PR.
- Run `black --check .` and `isort --check-only .` locally before pushing.

# Constraints
- Python 3.10+ only.
- numpy < 2.3.0.
- torch 2.7.1+cu128 (CUDA 12.8).
- Do not modify files in `aic_assets/` or `aic_gazebo/` unless specifically tasked.

# Key APIs — READ EXISTING CODE BEFORE WRITING NEW CODE

## LeRobot (v0.5.1)
- **ACT policy**: `lerobot.policies.act.modeling_act.ACTPolicy` / `configuration_act.ACTConfig`
- **ACTConfig key fields**: `vision_backbone` (not `backbone`), `chunk_size`, `n_action_steps`, `dim_model`, `n_heads`, `n_encoder_layers`, `n_decoder_layers`, `latent_dim`, `input_features`, `output_features`, `normalization_mapping`
- **Training**: `lerobot.scripts.lerobot_train.train(cfg)` with `lerobot.configs.train.TrainPipelineConfig`
- **Dataset**: `lerobot.datasets.lerobot_dataset.LeRobotDataset`
- **Working ACT inference example**: `aic_example_policies/aic_example_policies/ros/RunACT.py`
- **Pre-trained checkpoint config**: `outputs/train/act_synthetic/checkpoints/080000/pretrained_model/config.json`

## Data
- Synthetic data: `/home/yifeng/aic_data_sym` (LeRobot v3.0 format)
  - 300 episodes, 159K frames, 20 FPS
  - 3 cameras (256x288 RGB), 26D state, 7D actions (pos + quat)
- Data generation: `aic_utils/sym_data/generate_synthetic.py`
- Data conversion: `aic_utils/sym_data/convert_cheatcode_to_lerobot.py`

## RLT (current approach)
- Trainer: `aic_utils/aic_rlt/aic_rlt/trainer.py`
- Models: `aic_utils/aic_rlt/aic_rlt/models/` (rl_token.py, actor_critic.py)
- VLA backends: `aic_utils/aic_rlt/aic_rlt/vla/` (xvla_wrapper.py, pi05_backend.py)
- Embedding extraction: `aic_utils/aic_rlt/scripts/prepare_embeddings.py`
- Training entry: `aic_utils/aic_rlt/scripts/train.py`

## Robot Interface
- LeRobot ↔ ROS 2 bridge: `aic_utils/lerobot_robot_aic/`
- Policy base class: `aic_model/aic_model/policy.py`
- ROS 2 lifecycle node: `aic_model/aic_model/aic_model.py`
