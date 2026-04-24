#!/usr/bin/env python3

"""Simple ACT training script for AIC data without HuggingFace Hub dependencies."""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def create_act_policy():
    """Create ACT policy with AIC configuration."""
    from lerobot.configs.policies import PolicyFeature
    from lerobot.configs.types import FeatureType
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    # Define input and output features for AIC task
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

    # Create ACT configuration optimized for manipulation
    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        # Action chunking
        chunk_size=10,
        n_action_steps=10,
        n_obs_steps=1,  # Single observation input
        # Vision backbone
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        replace_final_stride_with_dilation=False,
        # Transformer architecture (smaller for faster training)
        pre_norm=False,
        dim_model=256,  # Reduced from 512
        n_heads=8,
        dim_feedforward=1024,  # Reduced from 3200
        feedforward_activation="relu",
        n_encoder_layers=4,
        n_decoder_layers=1,
        # VAE configuration
        use_vae=True,
        latent_dim=32,
        n_vae_encoder_layers=4,
        # Training settings
        dropout=0.1,
        kl_weight=10.0,
    )

    return ACTPolicy(config, dataset_stats=None)


def load_dataset_stats(data_dir: str) -> Dict[str, Any]:
    """Load dataset statistics for normalization."""
    stats_path = Path(data_dir) / "meta" / "stats.json"
    with open(stats_path, "r") as f:
        stats = json.load(f)
    return stats


def create_dummy_batch(
    batch_size: int = 4, device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """Create a properly shaped batch for testing ACT training."""

    batch = {
        # Images as individual keys matching input_features
        "observation.images.left_camera": torch.randn(batch_size, 3, 256, 288).to(
            device
        ),
        "observation.images.center_camera": torch.randn(batch_size, 3, 256, 288).to(
            device
        ),
        "observation.images.right_camera": torch.randn(batch_size, 3, 256, 288).to(
            device
        ),
        # State: (B, state_dim) - no time dimension for single observation
        "observation.state": torch.randn(batch_size, 26).to(device),
        # Actions: (B, chunk_size, action_dim) - target actions for loss calculation
        "action": torch.randn(batch_size, 10, 7).to(device),
        "action_is_pad": torch.zeros(batch_size, 10, dtype=torch.bool).to(device),
    }

    return batch


def test_act_training_loop():
    """Test ACT training with dummy data."""
    print("🚀 Starting ACT training test...")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create policy
    print("Creating ACT policy...")
    policy = create_act_policy()
    policy.to(device)
    print(
        f"✅ Policy created with {sum(p.numel() for p in policy.parameters()):,} parameters"
    )

    # Setup optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)

    # Test training loop
    policy.train()
    num_steps = 10

    print(f"Running {num_steps} training steps...")
    for step in range(num_steps):
        # Create batch
        batch = create_dummy_batch(batch_size=4, device=device)

        # Forward pass
        optimizer.zero_grad()
        loss, loss_dict = policy(batch)
        if loss is None:
            print(
                "❌ No loss in output - this indicates a problem with the training setup"
            )
            return False

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"Step {step+1:2d}: loss={loss.item():.4f}")

    print("✅ Training loop completed successfully!")

    # Test inference mode
    policy.eval()
    print("Testing inference mode...")

    with torch.no_grad():
        # For inference, we don't need target actions
        inference_batch = {
            "observation.images.left_camera": torch.randn(1, 3, 256, 288).to(device),
            "observation.images.center_camera": torch.randn(1, 3, 256, 288).to(device),
            "observation.images.right_camera": torch.randn(1, 3, 256, 288).to(device),
            "observation.state": torch.randn(1, 26).to(device),
        }

        # Use select_action for inference
        action = policy.select_action(inference_batch)
        print(f"✅ Inference successful! Action shape: {action.shape}")

    return True


if __name__ == "__main__":
    try:
        success = test_act_training_loop()
        if success:
            print("\n🎉 ACT training test completed successfully!")
            exit(0)
        else:
            print("\n❌ ACT training test failed!")
            exit(1)
    except Exception as e:
        print(f"\n💥 Training test crashed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
