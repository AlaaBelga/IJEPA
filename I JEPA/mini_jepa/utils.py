import random
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms


@dataclass
class DatasetConfig:
    name: str = "cifar10"
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 2


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(cfg: DatasetConfig):
    if cfg.name.lower() == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_set = datasets.CIFAR10(cfg.data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(cfg.data_dir, train=False, download=True, transform=transform)
        channels, img_size = 3, 32
    elif cfg.name.lower() == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        train_set = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(cfg.data_dir, train=False, download=True, transform=transform)
        channels, img_size = 1, 28
    else:
        raise ValueError(f"Unsupported dataset: {cfg.name}")

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    return train_loader, test_loader, channels, img_size


def random_mask_batch(images: torch.Tensor, patch_size: int):
    """Mask one random rectangular patch per image and return target patch + mask features."""
    b, c, h, w = images.shape
    if patch_size > h or patch_size > w:
        raise ValueError("patch_size cannot exceed image dimensions")

    masked = images.clone()
    patches = torch.zeros((b, c, patch_size, patch_size), device=images.device)
    mask_features = torch.zeros((b, 4), device=images.device)

    for i in range(b):
        top = torch.randint(0, h - patch_size + 1, (1,), device=images.device).item()
        left = torch.randint(0, w - patch_size + 1, (1,), device=images.device).item()

        patch = images[i, :, top : top + patch_size, left : left + patch_size]
        patches[i] = patch
        masked[i, :, top : top + patch_size, left : left + patch_size] = 0.0

        # Normalized mask geometry: (x, y, w, h).
        mask_features[i] = torch.tensor(
            [left / max(1, w - 1), top / max(1, h - 1), patch_size / w, patch_size / h],
            device=images.device,
        )

    return masked, patches, mask_features


def find_mask_region(masked_image: torch.Tensor, patch_size: int) -> Tuple[int, int]:
    """Find the darkest patch-sized region in a masked image."""
    if masked_image.dim() == 4:
        masked_image = masked_image[0]

    if patch_size > masked_image.shape[-1] or patch_size > masked_image.shape[-2]:
        raise ValueError("patch_size cannot exceed image dimensions")

    grayscale = masked_image.mean(dim=0, keepdim=True).unsqueeze(0)
    kernel = torch.ones((1, 1, patch_size, patch_size), device=masked_image.device)
    scores = F.conv2d(grayscale, kernel).squeeze(0).squeeze(0)

    min_index = torch.argmin(scores)
    width = scores.shape[-1]
    top = int(min_index // width)
    left = int(min_index % width)
    return top, left


def find_black_square_region(masked_image: torch.Tensor, patch_size: int, threshold: float = -0.85) -> Tuple[int, int]:
    """Find the top-left corner of the near-black masked square.

    This is more stable than a sliding darkness score when the masked image
    contains other dark objects.
    """
    if masked_image.dim() == 4:
        masked_image = masked_image[0]

    if patch_size > masked_image.shape[-1] or patch_size > masked_image.shape[-2]:
        raise ValueError("patch_size cannot exceed image dimensions")

    grayscale = masked_image.mean(dim=0)
    dark_pixels = grayscale <= threshold
    if not dark_pixels.any():
        return find_mask_region(masked_image, patch_size)

    ys, xs = torch.where(dark_pixels)
    top = int(ys.min().item())
    left = int(xs.min().item())

    top = min(max(0, top), masked_image.shape[-2] - patch_size)
    left = min(max(0, left), masked_image.shape[-1] - patch_size)
    return top, left


def masked_image_to_features(masked_image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert a masked image into the normalized mask geometry expected by the predictor."""
    if masked_image.dim() == 4:
        masked_image = masked_image[0]

    top, left = find_black_square_region(masked_image, patch_size)
    _, height, width = masked_image.shape
    return torch.tensor(
        [[left / max(1, width - 1), top / max(1, height - 1), patch_size / width, patch_size / height]],
        device=masked_image.device,
    )


def update_ema(target_model: torch.nn.Module, source_model: torch.nn.Module, momentum: float = 0.99) -> None:
    with torch.no_grad():
        for p_t, p_s in zip(target_model.parameters(), source_model.parameters()):
            p_t.data = momentum * p_t.data + (1.0 - momentum) * p_s.data


def denormalize(
    images: torch.Tensor,
    dataset: str,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
        std_t = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
        return (images * std_t + mean_t).clamp(0.0, 1.0)

    if dataset.lower() == "cifar10":
        mean_t = torch.tensor([0.5, 0.5, 0.5], device=images.device).view(1, -1, 1, 1)
        std_t = torch.tensor([0.5, 0.5, 0.5], device=images.device).view(1, -1, 1, 1)
    elif dataset.lower() == "mnist":
        mean_t = torch.tensor([0.5], device=images.device).view(1, -1, 1, 1)
        std_t = torch.tensor([0.5], device=images.device).view(1, -1, 1, 1)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return (images * std_t + mean_t).clamp(0.0, 1.0)


def reconstruct_images(masked_images: torch.Tensor, predicted_patches: torch.Tensor, mask_features: torch.Tensor):
    recon = masked_images.clone()
    b, _, h, w = masked_images.shape
    patch_size = predicted_patches.shape[-1]

    for i in range(b):
        left = int(mask_features[i, 0].item() * max(1, w - 1))
        top = int(mask_features[i, 1].item() * max(1, h - 1))
        left = min(max(0, left), w - patch_size)
        top = min(max(0, top), h - patch_size)
        recon[i, :, top : top + patch_size, left : left + patch_size] = predicted_patches[i]

    return recon


def _feather_alpha_mask(patch_size: int, feather: int, device: torch.device) -> torch.Tensor:
    """Create a soft alpha mask that fades in from the patch border."""
    feather = max(0, min(feather, patch_size // 2))
    if feather == 0:
        return torch.ones((1, 1, patch_size, patch_size), device=device)

    coords = torch.arange(patch_size, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    dist_to_edge = torch.minimum(torch.minimum(xx, yy), torch.minimum(patch_size - 1 - xx, patch_size - 1 - yy))
    alpha = (dist_to_edge / float(feather)).clamp(0.0, 1.0)
    return alpha.view(1, 1, patch_size, patch_size)


def blend_patch_into_image(
    masked_images: torch.Tensor,
    predicted_patches: torch.Tensor,
    mask_features: torch.Tensor,
    feather: int = 2,
):
    """Blend the predicted patch into the masked image with soft edges."""
    recon = masked_images.clone()
    b, _, h, w = masked_images.shape
    patch_size = predicted_patches.shape[-1]
    alpha = _feather_alpha_mask(patch_size, feather, masked_images.device)

    for i in range(b):
        left = int(mask_features[i, 0].item() * max(1, w - 1))
        top = int(mask_features[i, 1].item() * max(1, h - 1))
        left = min(max(0, left), w - patch_size)
        top = min(max(0, top), h - patch_size)

        region = recon[i : i + 1, :, top : top + patch_size, left : left + patch_size]
        patch = predicted_patches[i : i + 1]
        recon[i : i + 1, :, top : top + patch_size, left : left + patch_size] = (
            alpha * patch + (1.0 - alpha) * region
        )

    return recon


def save_training_curve(losses, output_path: str = "training_curve.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MiniJEPA Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_prediction_panel(
    original: torch.Tensor,
    masked: torch.Tensor,
    predicted_patch: torch.Tensor,
    reconstructed: torch.Tensor,
    dataset: str,
    output_path: str = "prediction_panel.png",
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
):
    original = denormalize(original.unsqueeze(0), dataset, mean=mean, std=std).squeeze(0).cpu()
    masked = denormalize(masked.unsqueeze(0), dataset, mean=mean, std=std).squeeze(0).cpu()
    predicted_patch = denormalize(predicted_patch.unsqueeze(0), dataset, mean=mean, std=std).squeeze(0).cpu()
    reconstructed = denormalize(reconstructed.unsqueeze(0), dataset, mean=mean, std=std).squeeze(0).cpu()

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    imgs = [original, masked, predicted_patch, reconstructed]
    titles = ["Original", "Masked", "Predicted Patch", "Reconstructed"]

    for ax, img, title in zip(axes, imgs, titles):
        if img.shape[0] == 1:
            ax.imshow(img[0], cmap="gray")
        else:
            ax.imshow(img.permute(1, 2, 0))
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_completion_panel(
    masked: torch.Tensor,
    predicted_patch: torch.Tensor,
    reconstructed: torch.Tensor,
    dataset: str,
    output_path: str = "completion_panel.png",
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
):
    masked = denormalize(masked.unsqueeze(0), dataset, mean=mean, std=std).squeeze(0).cpu()
    predicted_patch = denormalize(predicted_patch.unsqueeze(0), dataset, mean=mean, std=std).squeeze(0).cpu()
    reconstructed = denormalize(reconstructed.unsqueeze(0), dataset, mean=mean, std=std).squeeze(0).cpu()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    imgs = [masked, predicted_patch, reconstructed]
    titles = ["Masked Input", "Predicted Patch", "Completed Image"]

    for ax, img, title in zip(axes, imgs, titles):
        if img.shape[0] == 1:
            ax.imshow(img[0], cmap="gray")
        else:
            ax.imshow(img.permute(1, 2, 0))
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
