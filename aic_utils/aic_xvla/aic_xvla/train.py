"""Thin wrapper around X-VLA's official training scripts that:

  1. Registers the aic domain handler with X-VLA's registry.
  2. Selects between X-VLA's `train.py` (full FT) and `peft_train.py` (LoRA FT).
  3. Optionally enables live W&B mirroring of X-VLA's tensorboard metrics via
     `wandb.init(sync_tensorboard=True)` (X-VLA writes tensorboard only).
  4. Forwards all remaining CLI args to the chosen entry point.

Usage (from inside X-VLA's environment, with both repos on PYTHONPATH):
    PYTHONPATH=$XVLA_REPO:$AIC_XVLA_PKG \\
    accelerate launch -m aic_xvla.train \\
        --mode peft \\
        --wandb-project aic-xvla --wandb-entity <your-entity> \\
        --models 2toINF/X-VLA-Pt \\
        --train_metas_path /path/to/aic_meta.json \\
        --output_dir runnings/aic_xvla \\
        --batch_size 1 --learning_rate 5e-4 \\
        --iters 2000 --save_interval 1000

Both `train.py` (full) and `peft_train.py` (LoRA) are official X-VLA
fine-tune recipes. Pick `peft` when GPU memory is tight; `full` when you
have the budget to update all weights.
"""

from __future__ import annotations

import argparse
import sys


def _init_wandb(
    output_dir: str,
    project: str,
    entity: str | None,
    run_name: str | None,
    config: dict,
) -> None:
    """Live W&B mirror: capture every tfevent X-VLA's Accelerator writes."""
    import wandb  # imported lazily so the wrapper still works without wandb

    wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        dir=output_dir,
        config=config,
        sync_tensorboard=True,
    )


def main() -> None:
    from aic_xvla.handler import register as register_aic_handler

    register_aic_handler()

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--mode",
        choices=["full", "peft"],
        default="full",
        help="X-VLA training entry point: 'full' = train.py, 'peft' = peft_train.py (LoRA)",
    )
    pre_parser.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project; if set, mirror tensorboard metrics live to W&B",
    )
    pre_parser.add_argument(
        "--wandb-entity", default=None, help="W&B entity (user/team)"
    )
    pre_parser.add_argument("--wandb-run-name", default=None, help="W&B run name")
    pre_args, remaining = pre_parser.parse_known_args()

    if pre_args.mode == "peft":
        from peft_train import get_args_parser
        from peft_train import main as xvla_main
    else:
        from train import get_args_parser
        from train import main as xvla_main

    parser = argparse.ArgumentParser("aic-xvla training", parents=[get_args_parser()])
    args = parser.parse_args(remaining)

    if pre_args.wandb_project:
        cfg = {**vars(args), "mode": pre_args.mode}
        _init_wandb(
            args.output_dir,
            pre_args.wandb_project,
            pre_args.wandb_entity,
            pre_args.wandb_run_name,
            cfg,
        )

    xvla_main(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
