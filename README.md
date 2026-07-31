# MiniJEPA — Self-Supervised Image Patch Prediction

A minimalist implementation of the **Joint Embedding Predictive Architecture (JEPA)** for hidden image patch prediction, trained on CIFAR-10.

The model learns to predict the **latent embedding** of a masked image region from its surrounding context — without relying on pixel-level reconstruction alone.

---

## Project Structure

```
IJEPA/
├── I JEPA/
│   ├── MiniJEPA_Colab_Training.ipynb   # Self-contained Colab training notebook
│   └── mini_jepa/                       # Full local Python project
│       ├── models/                      # Encoder, Predictor, Decoder
│       ├── train.py                     # Local training script
│       ├── infer.py                     # Local inference script
│       ├── app.py                       # FastAPI + Gradio demo
│       ├── utils.py                     # Masking, blending, visualization
│       └── requirements.txt
│
└── JEPA/
    ├── app.py                           # Streamlit demo UI
    ├── jepa_inference.py                # Inference service
    └── requirements.txt
```

---

## How It Works

1. A random square patch is **masked** (hidden) from an image
2. The **context encoder** encodes the visible region
3. The **predictor** estimates the latent embedding of the hidden patch
4. The **decoder** reconstructs the missing pixels from the predicted embedding

**Loss**: Embedding MSE (JEPA objective) + pixel reconstruction MSE

---

## Quickstart — Google Colab (Recommended)

Open [`I JEPA/MiniJEPA_Colab_Training.ipynb`](I%20JEPA/MiniJEPA_Colab_Training.ipynb) in Google Colab.

It is fully self-contained — no extra files needed. Trains on CIFAR-10 with a GPU in ~20 minutes.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlaaBelga/IJEPA/blob/main/I%20JEPA/MiniJEPA_Colab_Training.ipynb)

---

## Local Setup

```bash
cd "I JEPA/mini_jepa"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train
```bash
python train.py --dataset cifar10 --epochs 20 --batch_size 128
```

### Inference
```bash
python infer.py --checkpoint outputs/mini_jepa.pt
```

### Run Demo App
```bash
python app.py --checkpoint outputs/mini_jepa.pt
```

---

## Streamlit Demo

The `JEPA/` folder contains a Streamlit web app for interactive demo.

> ⚠️ Requires the trained checkpoint (`mini_jepa_final.pt`). Download it after training from Colab or train locally first.

```bash
cd JEPA
pip install -r requirements.txt
streamlit run app.py
```

---

## Architecture

| Component | Details |
|---|---|
| **Encoder** | ResNet18 pretrained on ImageNet (adapted for 32×32) |
| **Predictor** | 3-layer MLP with LayerNorm |
| **Decoder** | Transposed-conv decoder with residual blocks |
| **Dataset** | CIFAR-10 (auto-downloaded) |
| **Patch size** | 8×8 pixels |

---

## Results

The model learns to predict plausible content for masked regions after 20 epochs of training on CIFAR-10 using a GPU.

---

## Author

**Alaa Belga** — [@AlaaBelga](https://github.com/AlaaBelga)