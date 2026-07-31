import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models import ConvEncoder, JEPAEmbeddingPredictor, PatchDecoder
from utils import (
    DatasetConfig,
    get_dataloaders,
    get_device,
    random_mask_batch,
    save_prediction_panel,
    save_training_curve,
    set_seed,
    update_ema,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MiniJEPA for hidden patch prediction")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--ema", type=float, default=0.99)
    parser.add_argument("--recon_weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = get_device()

    train_loader, _, channels, img_size = get_dataloaders(
        DatasetConfig(
            name=args.dataset,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
        )
    )

    if args.patch_size > img_size:
        raise ValueError(f"patch_size ({args.patch_size}) cannot exceed image size ({img_size})")

    context_encoder = ConvEncoder(in_channels=channels, embedding_dim=args.embedding_dim).to(device)
    target_encoder = ConvEncoder(in_channels=channels, embedding_dim=args.embedding_dim).to(device)
    predictor = JEPAEmbeddingPredictor(embedding_dim=args.embedding_dim).to(device)
    decoder = PatchDecoder(
        embedding_dim=args.embedding_dim,
        out_channels=channels,
        patch_size=args.patch_size,
    ).to(device)

    target_encoder.load_state_dict(context_encoder.state_dict())
    for p in target_encoder.parameters():
        p.requires_grad = False

    emb_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(context_encoder.parameters()) + list(predictor.parameters()) + list(decoder.parameters()),
        lr=args.lr,
    )

    epoch_losses = []

    for epoch in range(1, args.epochs + 1):
        context_encoder.train()
        predictor.train()
        decoder.train()

        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for images, _ in progress:
            images = images.to(device)
            masked_images, target_patches, mask_features = random_mask_batch(images, patch_size=args.patch_size)

            context_embedding = context_encoder(masked_images)
            pred_embedding = predictor(context_embedding, mask_features)

            with torch.no_grad():
                target_embedding = target_encoder(target_patches)

            pred_patch = decoder(pred_embedding)

            emb_loss = emb_criterion(pred_embedding, target_embedding)
            recon_loss = recon_criterion(pred_patch, target_patches)
            loss = emb_loss + args.recon_weight * recon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema(target_encoder, context_encoder, momentum=args.ema)

            running_loss += loss.item()
            progress.set_postfix(loss=loss.item(), emb=emb_loss.item(), recon=recon_loss.item())

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch:02d} | avg loss: {avg_loss:.6f}")

    checkpoint_path = os.path.join(args.output_dir, "mini_jepa.pt")
    torch.save(
        {
            "dataset": args.dataset,
            "patch_size": args.patch_size,
            "embedding_dim": args.embedding_dim,
            "channels": channels,
            "context_encoder": context_encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "decoder": decoder.state_dict(),
            "losses": epoch_losses,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}")

    save_training_curve(epoch_losses, output_path=os.path.join(args.output_dir, "training_curve.png"))

    # Save one qualitative panel from the last processed batch.
    with torch.no_grad():
        sample_original = images[0].detach().cpu()
        sample_masked = masked_images[0].detach().cpu()
        sample_patch = pred_patch[0].detach().cpu()
        sample_recon = sample_masked.clone()

        # Compute patch coordinates from normalized mask features for this sample.
        h, w = sample_recon.shape[-2], sample_recon.shape[-1]
        left = int(mask_features[0, 0].item() * max(1, w - 1))
        top = int(mask_features[0, 1].item() * max(1, h - 1))
        left = min(max(0, left), w - args.patch_size)
        top = min(max(0, top), h - args.patch_size)
        sample_recon[:, top : top + args.patch_size, left : left + args.patch_size] = sample_patch

        save_prediction_panel(
            sample_original,
            sample_masked,
            sample_patch,
            sample_recon,
            dataset=args.dataset,
            output_path=os.path.join(args.output_dir, "sample_prediction.png"),
        )

    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
