import torch
import torch.nn as nn


class JEPAEmbeddingPredictor(nn.Module):
    """Predict hidden patch embedding from context embedding and mask geometry."""

    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 512, mask_feat_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + mask_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, context_embedding: torch.Tensor, mask_features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([context_embedding, mask_features], dim=1)
        return self.net(x)


class LayerNormPredictor(nn.Module):
    """Predictor with LayerNorm blocks to match ImageNet-trained checkpoints."""

    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 512, mask_feat_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + mask_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, context_embedding: torch.Tensor, mask_features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([context_embedding, mask_features], dim=1)
        return self.net(x)
