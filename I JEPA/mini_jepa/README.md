# MiniJEPA

MiniJEPA is a simplified Joint Embedding Predictive Architecture (JEPA)-inspired project for hidden image patch prediction.

## Features

- Randomly masks a patch from an image.
- Encodes visible context.
- Predicts latent embedding of the hidden patch.
- Decodes predicted embedding to pixels for visualization.
- Produces a qualitative panel: original, masked, predicted patch, reconstructed image.
- Includes a lightweight FastAPI demo API.

## Project Structure

```
mini_jepa/
├── data/
├── models/
│   ├── encoder.py
│   ├── predictor.py
│   └── decoder.py
├── notebooks/
├── train.py
├── infer.py
├── app.py
├── utils.py
├── requirements.txt
└── README.md
```

## Setup

```bash
cd mini_jepa
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

CIFAR-10 :

```bash
python train.py --dataset cifar10 --epochs 20 --batch_size 128 --patch_size 8
```

MNIST:

```bash
python train.py --dataset mnist --epochs 10 --batch_size 128 --patch_size 7
```

Training outputs are saved in `outputs/`:
- `mini_jepa.pt` checkpoint
- `training_curve.png`
- `sample_prediction.png`

## Inference

Use random test sample:

```bash
python infer.py --checkpoint outputs/mini_jepa.pt
```

Use your own image:

```bash
python infer.py --checkpoint outputs/mini_jepa.pt --input path/to/image.png --output outputs/custom_prediction.png
```

## FastAPI Demo

```bash
python app.py --checkpoint outputs/mini_jepa.pt
```

Open `http://127.0.0.1:8000` in your browser.

The server exposes:

- `GET /api/health` for a quick status check.
- `POST /api/predict` for image-upload inference.

The API expects a masked image with a clear missing square region. It will detect the hole, predict the missing patch, and return the completed image plus a preview panel as data URLs.

This backend is ready to sit behind a React/Vite client if you want a custom UI later.

## JEPA Objective

MiniJEPA minimizes embedding prediction error:

$$
\mathcal{L}_{\text{embed}} = \| z_{\text{pred}} - z_{\text{target}} \|_2^2
$$

This implementation also adds a lightweight reconstruction loss for clearer visualizations.

## Suggested Class Deliverables

1. Training script and model code.
2. Demo app screenshots or live demo.
3. Report comparing datasets and patch sizes.
4. Short presentation explaining JEPA intuition and results.
