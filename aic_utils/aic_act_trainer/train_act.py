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

"""ACT Training Script for AIC Cable Insertion Task

Trains an Action Chunking Transformer (ACT) policy on synthetic data for the
AIC cable insertion task. This provides an alternative to the RLT approach.

Usage:
    python train_act.py \
        --data_dir /home/yifeng/aic_data_sym \
        --output_dir outputs/act_training \
        --chunk_size 10 \
        --batch_size 8 \
        --steps 100000 \
        --lr 1e-5 \
        --use_vae
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import draccus
import torch
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PolicyFeature
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, NormalizationMode

# LeRobot imports
from lerobot.datasets.factory import make_dataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.scripts.lerobot_train import train
from lerobot.utils.utils import init_logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_act_config(
    chunk_size: int = 10,
    use_vae: bool = True,
    lr: float = 1e-5,
    use_amp: bool = False,
) -> ACTConfig:
    """Create an ACT configuration for AIC cable insertion task."""

    # Create input/output feature mappings
    input_features = {
        "observation.images.left_camera": PolicyFeature(
            FeatureType.VISUAL, [3, 256, 288]
        ),
        "observation.images.center_camera": PolicyFeature(
            FeatureType.VISUAL, [3, 256, 288]
        ),
        "observation.images.right_camera": PolicyFeature(
            FeatureType.VISUAL, [3, 256, 288]
        ),
        "observation.state": PolicyFeature(FeatureType.STATE, [26]),
    }

    output_features = {
        "action": PolicyFeature(FeatureType.ACTION, [7]),  # [x,y,z,qx,qy,qz,qw]
    }

    # Create ACT config with our settings
    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        # Action chunking
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        # Architecture optimized for manipulation
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        # Transformer config - smaller than default for faster training
        dim_model=256,  # Reduced from 512
        n_heads=8,
        dim_feedforward=1024,  # Reduced from 3200
        n_encoder_layers=4,
        n_decoder_layers=1,  # Match original ACT bug/feature
        # VAE config
        use_vae=use_vae,
        latent_dim=32,
        n_vae_encoder_layers=4,
        # Training config
        dropout=0.1,
        kl_weight=10.0,
        # Mixed precision
        use_amp=use_amp,
        # Learning rates
        optimizer_lr=lr,
        optimizer_lr_backbone=lr,
        optimizer_weight_decay=1e-4,
        # Hub config - train locally, don't push
        push_to_hub=False,
        repo_id="synthetic/act_aic_cable_insertion",
    )

    return config


def create_dataset_config(data_dir: str) -> DatasetConfig:
    """Create dataset configuration for AIC synthetic data."""
    return DatasetConfig(
        repo_id="synthetic/cable_insertion",  # Will be ignored, use root instead
        root=data_dir,  # Points to aic_data_sym
    )


def create_training_config(
    data_dir: str,
    output_dir: str,
    chunk_size: int = 10,
    batch_size: int = 8,
    steps: int = 100000,
    lr: float = 1e-5,
    use_vae: bool = True,
    eval_freq: int = 10000,
    log_freq: int = 200,
    save_freq: int = 10000,
    seed: int = 1000,
    wandb_enable: bool = False,
    wandb_project: str = "aic-act",
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    checkpoint: Optional[str] = None,
    use_amp: bool = False,
) -> TrainPipelineConfig:
    """Create complete training pipeline configuration."""

    # Create policy config
    policy_config = create_act_config(
        chunk_size=chunk_size, use_vae=use_vae, lr=lr, use_amp=use_amp
    )

    # Load from checkpoint if provided
    if checkpoint:
        policy_config.pretrained_path = Path(checkpoint)

    # Create dataset config
    dataset_config = create_dataset_config(data_dir)

    # Create wandb config
    from lerobot.configs.default import WandBConfig

    wandb_config = WandBConfig(
        enable=wandb_enable,
        entity=wandb_entity,
        project=wandb_project if wandb_enable else None,
        run_id=wandb_run_name,
    )

    # Create training config
    train_config = TrainPipelineConfig(
        dataset=dataset_config,
        policy=policy_config,
        output_dir=Path(output_dir),
        seed=seed,
        batch_size=batch_size,
        steps=steps,
        eval_freq=eval_freq,
        log_freq=log_freq,
        save_freq=save_freq,
        save_checkpoint=True,
        use_policy_training_preset=True,
        wandb=wandb_config,
    )

    return train_config


def main():
    parser = argparse.ArgumentParser(
        description="Train ACT policy on AIC synthetic data"
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/yifeng/aic_data_sym",
        help="Path to LeRobot dataset (contains data/, meta/, images/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/act_training",
        help="Directory to save training outputs",
    )

    # Model arguments
    parser.add_argument("--chunk_size", type=int, default=10, help="Action chunk size")
    parser.add_argument(
        "--use_vae", action="store_true", default=True, help="Enable VAE"
    )
    parser.add_argument(
        "--no_vae", action="store_false", dest="use_vae", help="Disable VAE"
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument(
        "--steps", type=int, default=100000, help="Total training steps"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed")
    parser.add_argument(
        "--use_amp", action="store_true", help="Enable mixed precision training"
    )

    # Logging arguments
    parser.add_argument(
        "--eval_freq", type=int, default=10000, help="Evaluation frequency"
    )
    parser.add_argument("--log_freq", type=int, default=200, help="Logging frequency")
    parser.add_argument("--save_freq", type=int, default=10000, help="Save frequency")

    # Wandb arguments
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--wandb_project", type=str, default="aic-act", help="Wandb project name"
    )
    parser.add_argument("--wandb_run_name", type=str, help="Wandb run name")
    parser.add_argument("--wandb_entity", type=str, help="WandB entity/team name")

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained_model dir for continued training",
    )

    args = parser.parse_args()

    # Validate data directory exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_path} does not exist")

    required_subdirs = ["data", "meta"]
    for subdir in required_subdirs:
        subdir_path = data_path / subdir
        if not subdir_path.exists():
            raise FileNotFoundError(
                f"Required subdirectory {subdir_path} does not exist"
            )

    logger.info(f"Training ACT policy on data from {data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Use VAE: {args.use_vae}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Training steps: {args.steps}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Use AMP: {args.use_amp}")

    # Create training configuration
    train_config = create_training_config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        use_vae=args.use_vae,
        eval_freq=args.eval_freq,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        seed=args.seed,
        wandb_enable=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        checkpoint=args.checkpoint,
        use_amp=args.use_amp,
    )

    # Initialize logging
    init_logging()

    try:
        # Start training using LeRobot's training infrastructure
        logger.info("Starting ACT training...")
        train(train_config)
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
