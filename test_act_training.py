#!/usr/bin/env python3

"""Test ACT training pipeline with minimal steps to validate configuration."""

import shutil
from pathlib import Path


def test_act_training():
    """Run a minimal ACT training test."""
    print("Testing ACT training pipeline...")

    # Clean up any previous test outputs
    test_output_dir = Path("outputs/test_act_training")
    if test_output_dir.exists():
        shutil.rmtree(test_output_dir)

    try:
        from lerobot.scripts.lerobot_train import train
        from lerobot.utils.utils import init_logging

        from aic_utils.aic_act_trainer.train_act import create_training_config

        # Initialize logging
        init_logging()

        # Create minimal training config
        train_config = create_training_config(
            data_dir="/home/yifeng/aic_data_sym",
            output_dir="outputs/test_act_training",
            chunk_size=10,
            batch_size=2,  # Very small batch for testing
            steps=10,  # Very few steps for testing
            lr=1e-5,
            use_vae=True,
            eval_freq=20,  # Disable eval (more steps than total)
            log_freq=5,
            save_freq=20,  # Disable save (more steps than total)
            seed=1000,
            wandb_enable=False,
        )

        print(f"Config created successfully!")
        print(f"Data dir: {train_config.dataset.root}")
        print(f"Output dir: {train_config.output_dir}")
        print(f"Batch size: {train_config.batch_size}")
        print(f"Steps: {train_config.steps}")
        print(f"Policy type: {train_config.policy.type}")
        print(f"Chunk size: {train_config.policy.chunk_size}")

        # Try to start training
        print("\nStarting minimal training...")
        train(train_config)

        print("✅ Training test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Clean up test outputs
        if test_output_dir.exists():
            shutil.rmtree(test_output_dir)
            print("Cleaned up test outputs")


if __name__ == "__main__":
    success = test_act_training()
    if not success:
        sys.exit(1)
