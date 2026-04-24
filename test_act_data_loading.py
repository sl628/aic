#!/usr/bin/env python3

"""Test script to validate ACT training setup before full training."""

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def test_data_loading():
    """Test if we can load the AIC synthetic dataset."""
    print("Testing data loading...")

    try:
        # Load dataset directly
        print("Loading dataset...")
        dataset = LeRobotDataset(
            repo_id="synthetic/cable_insertion",
            root="/home/yifeng/aic_data_sym",
        )

        print(f"Dataset loaded successfully!")
        print(f"Total frames: {len(dataset)}")

        # Get first sample
        print("\nLoading first sample...")
        sample = dataset[0]

        print("Sample keys:", list(sample.keys()))

        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value)} = {value}")

        return True

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n✅ Data loading test passed!")
    else:
        print("\n❌ Data loading test failed!")
        sys.exit(1)
