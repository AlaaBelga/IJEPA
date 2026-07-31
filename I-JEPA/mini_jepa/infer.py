import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from models import ConvEncoder, JEPAEmbeddingPredictor, PatchDecoder, StrongPatchDecoder
from utils import DatasetConfig, blend_patch_into_image, find_black_square_region, get_dataloaders, get_device, save_completion_panel


def parse_args():
    parser = argparse.ArgumentParser(description="Run MiniJEPA inference")
    parser.add_argument("--checkpoint", type=str, default="outputs/mini_jepa.pt")
    parser.add_argument("--input", type=str, default="", help="Path to a masked image. If omitted, uses test sample.")
    parser.add_argument("--output", type=str, default="outputs/infer_prediction.png")
    parser.add_argument("--strong-decoder", action="store_true", default=True,
                        help="Use strong convolutional decoder (default: True)")
    parser.add_argument("--no-strong-decoder", dest="strong_decoder", action="store_false",
                        help="Disable strong decoder and use lightweight version")
    return parser.parse_args()


def load_single_image(path: str, channels: int):
    image = Image.open(path).convert("RGB" if channels == 3 else "L")
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32) if channels == 3 else (28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            if channels == 3
            else transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return transform(image).unsqueeze(0)


def main():
    args = parse_args()
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)

    dataset = ckpt["dataset"]
    patch_size = ckpt["patch_size"]
    embedding_dim = ckpt["embedding_dim"]
    channels = ckpt["channels"]

    context_encoder = ConvEncoder(in_channels=channels, embedding_dim=embedding_dim).to(device)
    predictor = JEPAEmbeddingPredictor(embedding_dim=embedding_dim).to(device)
    
    # Use strong decoder if requested
    if args.strong_decoder:
        decoder = StrongPatchDecoder(embedding_dim=embedding_dim, out_channels=channels, patch_size=patch_size).to(device)
    else:
        decoder = PatchDecoder(embedding_dim=embedding_dim, out_channels=channels, patch_size=patch_size).to(device)

    context_encoder.load_state_dict(ckpt["context_encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    
    # Only load if checkpoint has compatible weights
    if "decoder" in ckpt and ckpt["decoder"]:
        try:
            decoder.load_state_dict(ckpt["decoder"])
        except RuntimeError:
            print(f"Warning: Could not load decoder weights (architecture mismatch). Using random initialization.")

    context_encoder.eval()
    predictor.eval()
    decoder.eval()

    with torch.no_grad():
        if args.input:
            image = load_single_image(args.input, channels=channels).to(device)
        else:
            _, test_loader, _, _ = get_dataloaders(DatasetConfig(name=dataset, batch_size=1))
            image, _ = next(iter(test_loader))
            image = image.to(device)

        top, left = find_black_square_region(image, patch_size=patch_size)
        _, _, h, w = image.shape
        mask_features = torch.tensor(
            [[left / max(1, w - 1), top / max(1, h - 1), patch_size / w, patch_size / h]],
            device=device,
        )
        context_emb = context_encoder(image)
        pred_emb = predictor(context_emb, mask_features)
        pred_patch = decoder(pred_emb)

        recon = blend_patch_into_image(image, pred_patch, mask_features, feather=2)

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        save_completion_panel(image[0].cpu(), pred_patch[0].cpu(), recon[0].cpu(), dataset, output_path=args.output)
        print(f"Saved inference panel to: {args.output}")


if __name__ == "__main__":
    main()
