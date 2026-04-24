#!/usr/bin/env python3
#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""Launch ACT training using YAML configuration file.

This script provides a simpler interface for training ACT policies using
a YAML configuration file, which is easier to manage than command-line args.

Usage:
    python train_with_config.py --config config_act_aic.yaml
    python train_with_config.py --config config_act_aic.yaml --wandb
    python train_with_config.py --config config_act_aic.yaml --data_dir /path/to/data
"""

import argparse
import logging
from pathlib import Path

import draccus
import yaml
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_train import train
from lerobot.utils.utils import init_logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config_from_yaml(config_path: str) -> TrainPipelineConfig:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Convert to TrainPipelineConfig using draccus
    return draccus.decode(TrainPipelineConfig, config_dict)


def main():
    parser = argparse.ArgumentParser(description="Train ACT with YAML config")
    parser.add_argument(
        "--config",
        type=str,
        default="config_act_aic.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--data_dir", type=str, help="Override data directory from config"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Override output directory from config"
    )
    parser.add_argument(
        "--batch_size", type=int, help="Override batch size from config"
    )
    parser.add_argument("--steps", type=int, help="Override training steps from config")
    parser.add_argument(
        "--wandb", action="store_true", help="Enable wandb logging (overrides config)"
    )
    parser.add_argument("--wandb_project", type=str, help="Override wandb project name")
    parser.add_argument("--wandb_run_name", type=str, help="Override wandb run name")
    parser.add_argument("--wandb_entity", type=str, help="Override wandb entity/team")

    args = parser.parse_args()

    # Check config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to this script's directory
        script_dir = Path(__file__).parent
        config_path = script_dir / args.config
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")

    logger.info(f"Loading configuration from {config_path}")

    # Load base configuration from YAML
    train_config = load_config_from_yaml(str(config_path))

    # Apply command-line overrides
    if args.data_dir:
        train_config.dataset.root = args.data_dir
        logger.info(f"Overriding data_dir: {args.data_dir}")

    if args.output_dir:
        train_config.output_dir = Path(args.output_dir)
        logger.info(f"Overriding output_dir: {args.output_dir}")

    if args.batch_size:
        train_config.batch_size = args.batch_size
        logger.info(f"Overriding batch_size: {args.batch_size}")

    if args.steps:
        train_config.steps = args.steps
        logger.info(f"Overriding steps: {args.steps}")

    if args.wandb:
        train_config.wandb.enable = True
        logger.info("Enabling wandb logging")

    if args.wandb_project:
        train_config.wandb.project = args.wandb_project
        logger.info(f"Overriding wandb_project: {args.wandb_project}")

    if args.wandb_entity:
        train_config.wandb.entity = args.wandb_entity
        logger.info(f"Overriding wandb_entity: {args.wandb_entity}")

    if args.wandb_run_name:
        train_config.wandb.run_name = args.wandb_run_name
        logger.info(f"Overriding wandb_run_name: {args.wandb_run_name}")

    # Validate data directory exists
    data_path = Path(train_config.dataset.root)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    required_subdirs = ["data", "meta"]
    for subdir in required_subdirs:
        subdir_path = data_path / subdir
        if not subdir_path.exists():
            raise FileNotFoundError(
                f"Required subdirectory does not exist: {subdir_path}"
            )

    logger.info("=== Training Configuration ===")
    logger.info(f"Data directory: {train_config.dataset.root}")
    logger.info(f"Output directory: {train_config.output_dir}")
    logger.info(f"Policy type: {train_config.policy.type}")
    logger.info(f"Chunk size: {train_config.policy.chunk_size}")
    logger.info(f"Use VAE: {train_config.policy.use_vae}")
    logger.info(f"Batch size: {train_config.batch_size}")
    logger.info(f"Training steps: {train_config.steps}")
    logger.info(f"Learning rate: {train_config.policy.optimizer_lr}")
    logger.info(f"Wandb enabled: {train_config.wandb.enable}")
    logger.info("===============================")

    # Initialize logging
    init_logging()

    try:
        # Start training
        logger.info("Starting ACT training...")
        train(train_config)
        logger.info("Training completed successfully!")

        # Log final model location
        if train_config.output_dir:
            logger.info(f"Trained model saved to: {train_config.output_dir}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
