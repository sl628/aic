"""Image-based 4-phase classifier using ResNet18 late-fusion over 3 camera views.

Architecture:
  - Shared ResNet18 backbone (ImageNet-pretrained, fc removed → 512-d features)
  - Each of 3 views (left, center, right) processed independently
  - 512×3 = 1536-d concatenated → Linear(1536, 256) → ReLU → Linear(256, 4)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

PHASES = ["approach", "coarse_align", "fine_align", "insert"]
NUM_PHASES = len(PHASES)

# ImageNet-normalisation matching ResNet18 pretrained weights.
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

VAL_TRANSFORMS = T.Compose([
    T.Resize((224, 224), antialias=True),
    T.Normalize(IMAGE_MEAN, IMAGE_STD),
])
TRAIN_TRANSFORMS = T.Compose([
    T.Resize((224, 224), antialias=True),
    T.RandomRotation(5),
    T.ColorJitter(brightness=0.05, contrast=0.05),
    T.Normalize(IMAGE_MEAN, IMAGE_STD),
])


class PhaseClassifierNet(nn.Module):
    """ResNet18 late-fusion classifier: 3 views → features → concat → MLP → 4 classes."""

    def __init__(self, num_views: int = 3, hidden: int = 256):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * num_views, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, NUM_PHASES),
        )
        self.num_views = num_views

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, V, 3, H, W) → logits: (B, 4)."""
        B, V, C, H, W = images.shape
        feats = self.backbone(images.view(B * V, C, H, W))  # (B*V, 512)
        feats = feats.view(B, V * feats.shape[-1])           # (B, 1536)
        return self.fusion(feats)                             # (B, 4)

    @torch.no_grad()
    def predict_phase(self, images: torch.Tensor) -> int:
        """images: (V, 3, H, W) or (1, V, 3, H, W) → phase index."""
        if images.dim() == 4:
            images = images.unsqueeze(0)
        logits = self(images)
        return logits.argmax(dim=1).item()

    @torch.no_grad()
    def predict_phase_name(self, images: torch.Tensor) -> str:
        idx = self.predict_phase(images)
        return PHASES[idx]
