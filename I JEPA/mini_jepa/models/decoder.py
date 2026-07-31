import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchDecoder(nn.Module):
    """Decode a latent embedding back to a patch for visualization."""

    def __init__(self, embedding_dim: int = 256, out_channels: int = 3, patch_size: int = 8):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.GELU(),
            nn.Linear(512, out_channels * patch_size * patch_size),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        out = self.net(embedding)
        return out.view(-1, self.out_channels, self.patch_size, self.patch_size)


class ResidualBlock(nn.Module):
    """Residual convolutional block with skip connection."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        
        # Skip connection: adjust channels if needed
        self.skip = nn.Identity() if in_channels == out_channels else \
                    nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.gelu(out + residual)


class StrongPatchDecoder(nn.Module):
    """Strong convolutional decoder with ResBlocks for higher-quality patch reconstruction.
    
    Uses transposed convolutions to upsample from latent embedding to full patch.
    Includes residual blocks for better feature propagation and detail preservation.
    """
    
    def __init__(self, embedding_dim: int = 512, out_channels: int = 3, patch_size: int = 8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.out_channels = out_channels
        self.patch_size = patch_size
        
        # Project embedding to initial feature map (4x4)
        self.initial_proj = nn.Linear(embedding_dim, 256 * 4 * 4)
        self.initial_bn = nn.BatchNorm1d(256 * 4 * 4)
        
        # Residual blocks to refine features
        self.res_block1 = ResidualBlock(256, 256, kernel_size=3, padding=1)
        
        # Upsample 4x4 -> 8x8 using transposed convolution
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.res_block2 = ResidualBlock(128, 128, kernel_size=3, padding=1)
        
        # Output layer: 128 channels -> 3 channels
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Decode embedding to patch.
        
        Args:
            embedding: (batch, embedding_dim)
        
        Returns:
            patch: (batch, out_channels, patch_size, patch_size)
        """
        batch_size = embedding.shape[0]
        
        # Project to initial feature map
        x = self.initial_proj(embedding)
        x = F.gelu(self.initial_bn(x))
        x = x.view(batch_size, 256, 4, 4)
        
        # Refine with residual block
        x = self.res_block1(x)
        
        # Upsample and refine
        x = self.upsample1(x)
        x = self.res_block2(x)
        
        # Final output
        x = self.final_conv(x)
        
        # Clamp to [-1, 1] range for consistency
        x = torch.clamp(x, -1.0, 1.0)
        
        return x


class TransposePatchDecoder(nn.Module):
    """Transposed-convolution decoder used in the ImageNet-trained checkpoint."""

    def __init__(self, embedding_dim: int = 512, out_channels: int = 3, patch_size: int = 8):
        super().__init__()
        if patch_size != 8:
            raise ValueError("TransposePatchDecoder supports patch_size=8 only")
        self.proj = nn.Linear(embedding_dim, 256 * 2 * 2)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        x = self.proj(embedding).view(-1, 256, 2, 2)
        return self.up(x)
