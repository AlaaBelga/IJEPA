# MiniJEPA — Self-Supervised Image Patch Prediction

📊 **[View the Project Presentation (PDF)](JEPA_Presentation.pdf)**

A minimalist implementation of the **Joint Embedding Predictive Architecture (JEPA)** for hidden image patch prediction, trained on CIFAR-10.

The model learns to predict the **latent embedding** of a masked image region from its surrounding context — without relying on pixel-level reconstruction alone.

---

## Project Structure

```
IJEPA/
└── I JEPA/
    ├── MiniJEPA_Colab_Training.ipynb   # ⭐ Self-contained Colab training notebook
    └── mini_jepa/                       # Full local Python project
        ├── models/
        │   ├── encoder.py               # ResNet18 context encoder
        │   ├── predictor.py             # Embedding predictor MLP
        │   └── decoder.py              # Patch decoder (transposed conv)
        ├── train.py                     # Local training script
        ├── infer.py                     # Local inference script
        ├── app.py                       # FastAPI + Gradio demo
        ├── utils.py                     # Masking, blending, visualization
        ├── requirements.txt
        └── README.md
```

---

## How It Works

1. A random square patch is **masked** (hidden) from an image
2. The **context encoder** (ResNet18) encodes the visible region into an embedding
3. The **predictor** estimates the latent embedding of the hidden patch given mask position
4. The **decoder** reconstructs the missing pixels from the predicted embedding

**Loss**: Embedding MSE (JEPA objective) + pixel reconstruction MSE

---

## Quickstart — Google Colab ⭐ (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlaaBelga/IJEPA/blob/main/I%20JEPA/MiniJEPA_Colab_Training.ipynb)

The notebook includes:
- ✅ Model definitions (encoder, predictor, decoder)
- ✅ CIFAR-10 data loading
- ✅ Full training loop with EMA target encoder
- ✅ Visualization panel: Original → Masked → Predicted Patch → Reconstructed

---

## Local Setup

```bash
cd "I JEPA/mini_jepa"
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Train
```bash
python train.py --dataset cifar10 --epochs 20 --batch_size 128 --patch_size 8
```

### Inference
```bash
python infer.py --checkpoint outputs/mini_jepa.pt
```

### Run Demo App (FastAPI + Gradio)
```bash
python app.py --checkpoint outputs/mini_jepa.pt
```

---

## Architecture

| Component | Details |
|---|---|
| **Encoder** | ResNet18 pretrained on ImageNet (adapted for 32×32) |
| **Predictor** | 3-layer MLP with LayerNorm |
| **Decoder** | Transposed-conv decoder with residual blocks |
| **Target Encoder** | EMA copy of the context encoder (frozen) |
| **Dataset** | CIFAR-10 (auto-downloaded) |
| **Patch size** | 8×8 pixels |
| **Embedding dim** | 512 |

---

## Key References

- [I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243) — Assran et al., 2023

---

## Author

**Alaa Belga** — [@AlaaBelga](https://github.com/AlaaBelga)
