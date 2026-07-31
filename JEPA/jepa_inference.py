from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF


PATCH_SIZE = 8
IMAGE_SIZE = 32
NUM_MASKS = 1
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = out + identity
        return torch.relu(out)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Identity(),
            nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64)),
            nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128)),
            nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256)),
            nn.Sequential(BasicBlock(256, 512, stride=2), BasicBlock(512, 512)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = self.proj(x)
        return x


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(514, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(512, 1024)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.view(x.shape[0], 256, 2, 2)
        return self.up(x)


class MaskToken(nn.Module):
    def __init__(self):
        super().__init__()
        self.pixel = nn.Parameter(torch.zeros(3, 1, 1))


class JEPAInpaintingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.predictor = Predictor()
        self.decoder = Decoder()
        self.mask_token = MaskToken()

    def forward(self, masked_image: torch.Tensor, mask_xy: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(masked_image)
        x = torch.cat([feats, mask_xy], dim=1)
        pred = self.predictor(x)
        return self.decoder(pred)


@dataclass
class InferenceResult:
    original: Image.Image
    masked: Image.Image
    reconstructed: Image.Image
    mask_box: Tuple[int, int, int, int]


class JEPAService:
    def __init__(self, checkpoint_path: str | Path):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device("cpu")
        self.model = JEPAInpaintingModel().to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.encoder.load_state_dict(checkpoint["encoder"], strict=True)
        self.model.predictor.load_state_dict(checkpoint["predictor"], strict=True)
        self.model.decoder.load_state_dict(checkpoint["decoder"], strict=True)
        self.model.mask_token.load_state_dict(checkpoint["mask_token"], strict=True)
        self.model.eval()
        
        # Debug: Print mask token value
        mask_token_value = self.model.mask_token.pixel.detach().cpu()
        print(f"DEBUG: Loaded mask_token.pixel = {mask_token_value.flatten().tolist()}")
        print(f"DEBUG: Checkpoint keys = {list(checkpoint.keys())}")

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = MEAN.to(tensor.device)
        std = STD.to(tensor.device)
        return (tensor - mean) / std

    def _denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = MEAN.to(tensor.device)
        std = STD.to(tensor.device)
        return (tensor * std + mean).clamp(0, 1)

    def _sample_mask(self, seed: int | None = None) -> Tuple[int, int]:
        rng = np.random.default_rng(seed)
        grid = IMAGE_SIZE // PATCH_SIZE
        max_idx = max(grid - 1, 0)
        x = int(rng.integers(0, max_idx + 1))
        y = int(rng.integers(0, max_idx + 1))
        return x * PATCH_SIZE, y * PATCH_SIZE

    def _mask_image(self, image: Image.Image, mask_x: int, mask_y: int) -> Image.Image:
        masked = self._to_tensor(image).squeeze(0)
        token = self.model.mask_token.pixel.detach().to(masked.device)
        masked[:, mask_y : mask_y + PATCH_SIZE, mask_x : mask_x + PATCH_SIZE] = token.expand(
            3, PATCH_SIZE, PATCH_SIZE
        )
        return TF.to_pil_image(self._denormalize(masked.unsqueeze(0)).squeeze(0))

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        tensor = TF.to_tensor(image).unsqueeze(0)
        tensor = self._normalize(tensor)
        return tensor.to(self.device)

    def _predict_patch(self, masked_tensor: torch.Tensor, mask_x: int, mask_y: int) -> torch.Tensor:
        mask_xy = torch.tensor([[mask_x / IMAGE_SIZE, mask_y / IMAGE_SIZE]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            patch = self.model(masked_tensor, mask_xy)
        return patch.squeeze(0)

    def reconstruct(self, image: Image.Image, seed: int | None = None) -> InferenceResult:
        prepared = self._prepare_image(image)
        mask_x, mask_y = self._sample_mask(seed)
        masked_tensor = self._to_tensor(prepared)
        token = self.model.mask_token.pixel.detach().to(masked_tensor.device)
        masked_tensor[:, :, mask_y : mask_y + PATCH_SIZE, mask_x : mask_x + PATCH_SIZE] = token.expand(
            1, 3, PATCH_SIZE, PATCH_SIZE
        )
        masked = TF.to_pil_image(self._denormalize(masked_tensor).squeeze(0))
        patch_tensor = self._predict_patch(masked_tensor, mask_x, mask_y)
        recon_tensor = masked_tensor.clone()
        recon_tensor[0, :, mask_y : mask_y + PATCH_SIZE, mask_x : mask_x + PATCH_SIZE] = patch_tensor
        reconstructed = TF.to_pil_image(self._denormalize(recon_tensor).squeeze(0))
        return InferenceResult(
            original=prepared,
            masked=masked,
            reconstructed=reconstructed,
            mask_box=(mask_x, mask_y, mask_x + PATCH_SIZE, mask_y + PATCH_SIZE),
        )
