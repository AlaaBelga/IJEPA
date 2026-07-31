import torch
import torch.nn as nn
from torchvision import models


class ConvEncoder(nn.Module):
    """Simple CNN encoder for both context images and target patches."""

    def __init__(self, in_channels: int = 3, embedding_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.proj(x)


class PretrainedEncoder(nn.Module):
    """ResNet18 encoder pretrained on ImageNet, adapted for 32x32 images."""

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))
