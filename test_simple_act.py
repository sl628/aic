#!/usr/bin/env python3

"""Simple ACT training test without HuggingFace Hub requirements."""

import sys

import torch


def test_simple_act():
    """Test ACT policy creation and basic forward pass."""
    print("Testing simple ACT policy creation...")

    try:
        from lerobot.configs.policies import PolicyFeature
        from lerobot.configs.types import FeatureType
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        # Create simple ACT config for AIC task
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

        config = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            chunk_size=10,
            n_action_steps=10,
            vision_backbone="resnet18",
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
            dim_model=256,
            n_heads=8,
            dim_feedforward=1024,
            n_encoder_layers=4,
            n_decoder_layers=1,
            use_vae=True,
            latent_dim=32,
            n_vae_encoder_layers=4,
            dropout=0.1,
            kl_weight=10.0,
        )

        print(f"✅ Config created successfully!")
        print(f"   Input features: {list(config.input_features.keys())}")
        print(f"   Output features: {list(config.output_features.keys())}")
        print(f"   Chunk size: {config.chunk_size}")
        print(f"   Use VAE: {config.use_vae}")

        # Test policy creation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = ACTPolicy(config, dataset_stats=None)
        policy.to(device)

        print(f"✅ Policy created successfully on {device}!")
        print(f"   Parameters: {sum(p.numel() for p in policy.parameters()):,}")

        # Test forward pass with dummy data
        batch_size = 2
        dummy_input = {
            "observation.images.left_camera": torch.randn(batch_size, 3, 256, 288).to(
                device
            ),
            "observation.images.center_camera": torch.randn(batch_size, 3, 256, 288).to(
                device
            ),
            "observation.images.right_camera": torch.randn(batch_size, 3, 256, 288).to(
                device
            ),
            "observation.state": torch.randn(batch_size, 26).to(device),
            "action": torch.randn(batch_size, 10, 7).to(
                device
            ),  # Target actions for training
            "action_is_pad": torch.zeros(batch_size, 10, dtype=torch.bool).to(device),
        }

        policy.train()  # Set to training mode
        loss, loss_dict = policy(dummy_input)

        print(f"✅ Forward pass successful!")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Loss components: {loss_dict}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_simple_act()
    sys.exit(0 if success else 1)
