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
