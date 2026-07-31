import argparse
import base64
import io
import os
from functools import lru_cache

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _infer_embedding_dim(predictor_state: dict) -> int:
    """Infer embedding_dim from predictor output layer (net.6)."""
    weight = predictor_state.get("net.6.weight")
    if weight is not None and weight.ndim == 2:
        return weight.shape[0]  # output dimension of final layer
    # Fallback: infer from input layer
    weight = predictor_state.get("net.0.weight")
    if weight is not None and weight.ndim == 2:
        return weight.shape[1] - 4
    return 512


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _file_to_data_url(path: str) -> str:
    with open(path, "rb") as file_handle:
        encoded = base64.b64encode(file_handle.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class MiniJEPABackend:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        import torch
        from models import (
            ConvEncoder,
            JEPAEmbeddingPredictor,
            LayerNormPredictor,
            PatchDecoder,
            PretrainedEncoder,
            TransposePatchDecoder,
        )
        from utils import denormalize, find_black_square_region, reconstruct_images, save_completion_panel

        self._torch = torch
        self._denormalize_fn = denormalize
        self._find_black_square_region = find_black_square_region
        self._reconstruct_images = reconstruct_images
        self._save_completion_panel = save_completion_panel

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.ckpt = torch.load(checkpoint_path, map_location=self.device)

        self.is_pretrained_checkpoint = "encoder" in self.ckpt
        if self.is_pretrained_checkpoint:
            self.dataset = self.ckpt.get("dataset", "cifar10")
            self.patch_size = self.ckpt.get("patch_size", 8)
            self.embedding_dim = self.ckpt.get("embedding_dim", _infer_embedding_dim(self.ckpt["predictor"]))
            self.channels = self.ckpt.get("channels", 3)
            self.norm_mean = IMAGENET_MEAN if self.channels == 3 else (0.5,)
            self.norm_std = IMAGENET_STD if self.channels == 3 else (0.5,)

            self.context_encoder = PretrainedEncoder(self.embedding_dim).to(self.device)
            self.predictor = LayerNormPredictor(self.embedding_dim).to(self.device)
            self.decoder = TransposePatchDecoder(self.embedding_dim, self.channels, self.patch_size).to(self.device)

            self.context_encoder.load_state_dict(self.ckpt["encoder"])
            self.predictor.load_state_dict(self.ckpt["predictor"])
            self.decoder.load_state_dict(self.ckpt["decoder"])
        else:
            self.dataset = self.ckpt["dataset"]
            self.patch_size = self.ckpt["patch_size"]
            self.embedding_dim = self.ckpt["embedding_dim"]
            self.channels = self.ckpt["channels"]
            self.norm_mean = (0.5, 0.5, 0.5) if self.channels == 3 else (0.5,)
            self.norm_std = (0.5, 0.5, 0.5) if self.channels == 3 else (0.5,)

            self.context_encoder = ConvEncoder(self.channels, self.embedding_dim).to(self.device)
            self.predictor = JEPAEmbeddingPredictor(self.embedding_dim).to(self.device)
            self.decoder = PatchDecoder(self.embedding_dim, self.channels, self.patch_size).to(self.device)

            self.context_encoder.load_state_dict(self.ckpt["context_encoder"])
            self.predictor.load_state_dict(self.ckpt["predictor"])

            if "decoder" in self.ckpt and self.ckpt["decoder"]:
                try:
                    self.decoder.load_state_dict(self.ckpt["decoder"])
                except RuntimeError:
                    print("Warning: Could not load decoder weights (architecture mismatch). Using random initialization.")

        self.context_encoder.eval()
        self.predictor.eval()
        self.decoder.eval()

    def _transform(self, image: Image.Image):
        from torchvision import transforms

        image = image.convert("RGB" if self.channels == 3 else "L")
        size = (32, 32) if self.channels == 3 else (28, 28)
        norm = transforms.Normalize(self.norm_mean, self.norm_std)
        transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), norm])
        return transform(image).unsqueeze(0)

    def _denormalize(self, images):
        return self._denormalize_fn(images, self.dataset, mean=self.norm_mean, std=self.norm_std)

    def predict(self, image: Image.Image):
        torch = self._torch
        x = self._transform(image).to(self.device)

        with torch.no_grad():
            try:
                top, left = self._find_black_square_region(x, self.patch_size)
                _, _, h, w = x.shape
                mask_features = torch.tensor(
                    [[left / max(1, w - 1), top / max(1, h - 1), self.patch_size / w, self.patch_size / h]],
                    device=self.device,
                )
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc

            context_emb = self.context_encoder(x)
            pred_emb = self.predictor(context_emb, mask_features)
            pred_patch = self.decoder(pred_emb)
            recon = self._reconstruct_images(x, pred_patch, mask_features)

            panel_path = "outputs/completion_panel.png"
            os.makedirs("outputs", exist_ok=True)
            self._save_completion_panel(
                x[0].cpu(),
                pred_patch[0].cpu(),
                recon[0].cpu(),
                self.dataset,
                panel_path,
                mean=self.norm_mean,
                std=self.norm_std,
            )

            completed_image = to_pil_image(self._denormalize(recon[0].cpu().unsqueeze(0)).squeeze(0))

        return completed_image, panel_path


def build_app(checkpoint_path: str, device: str = "cpu") -> FastAPI:
    app = FastAPI(title="MiniJEPA API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @lru_cache(maxsize=1)
    def get_backend() -> MiniJEPABackend:
        print("  Loading model weights on first request...")
        backend = MiniJEPABackend(checkpoint_path, device=device)
        print(f"  Checkpoint format: {'pretrained' if backend.is_pretrained_checkpoint else 'classic'}")
        print(f"  Normalization: {'ImageNet' if backend.is_pretrained_checkpoint else '0.5/0.5'}")
        return backend

    @app.get("/")
    def root():
        return HTMLResponse(
            "<h1>MiniJEPA API</h1><p>POST an image to <code>/api/predict</code> and check <code>/api/health</code> for status.</p>"
        )

    @app.get("/api/health")
    def health():
        return {"status": "ok", "model_loaded": get_backend.cache_info().currsize > 0}

    @app.post("/api/predict")
    async def predict(image: UploadFile = File(...)):
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Please upload an image file.")

        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="Could not decode the uploaded image.") from exc

        backend = get_backend()
        completed_image, panel_path = backend.predict(pil_image)

        return {
            "status": "ok",
            "dataset": backend.dataset,
            "patch_size": backend.patch_size,
            "channels": backend.channels,
            "completed_image": _image_to_data_url(completed_image),
            "panel_image": _file_to_data_url(panel_path),
            "panel_path": panel_path,
        }

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="FastAPI demo for MiniJEPA")
    parser.add_argument("--checkpoint", type=str, default="outputs/mini_jepa.pt")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="Use 0 to auto-select a free port")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Starting MiniJEPA API:")
    print(f"  Checkpoint: {args.checkpoint}")

    app = build_app(args.checkpoint, device=args.device)
    launch_port = args.port if args.port != 0 else 8000
    if args.port == 0:
        print(f"Launching at http://{args.host}:{launch_port}")
    else:
        print(f"Launching at http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=launch_port)


if __name__ == "__main__":
    main()
