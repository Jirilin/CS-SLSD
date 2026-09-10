from __future__ import annotations

import torch
from torch import nn


class SDSLVisionNet(nn.Module):
    
    def __init__(self, in_channels: int, num_classes: int = 10, feature_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projector = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(128, feature_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
