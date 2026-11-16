# Deep Learning Image Super-Resolution for Low-Quality Surveillance Feeds

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61dafb?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646cff?logo=vite&logoColor=white)](https://vitejs.dev/)
[![CUDA](https://img.shields.io/badge/CUDA-RTX%202050%20%C2%B7%2012.8-76b900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Single-Image Super-Resolution (SISR) deep learning system designed for forensic enhancement of low-quality surveillance camera feeds. Features **SRCNN** and **ESPCN** models ($2\times$ and $4\times$ scaling) trained under both pixel **MSE loss** and **Perceptual loss** (VGG11 feature extraction), accompanied by a FastAPI inference microservice and an interactive forensic web workbench.

---

## Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │    Low-Resolution Surveillance Feed (LR)     │
                      └──────────────────────┬───────────────────────┘
                                             │
                      ┌──────────────────────┴───────────────────────┐
                      ▼                                              ▼
          ┌────────────────────────┐                    ┌────────────────────────┐
          │   SRCNN Pipeline       │                    │   ESPCN Pipeline       │
          │                        │                    │                        │
          │ 1. Bicubic Pre-Upsample│                    │ 1. Conv1 (5x5, 64)     │
          │ 2. Conv1 (9x9, 64)     │                    │ 2. Conv2 (3x3, 32)     │
          │ 3. Conv2 (1x1, 32)     │                    │ 3. Conv3 (3x3, c*r^2)  │
          │ 4. Conv3 (5x5, c)      │                    │ 4. PixelShuffle(r)     │
          └───────────┬────────────┘                    └───────────┬────────────┘
                      │                                              │
                      └──────────────────────┬───────────────────────┘
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │      Loss Formulation & Optimization         │
                      │                                              │
                      │  • Phase 1: Pure MSE (L2 Pixel Supervision)  │
                      │  • Phase 2: Hybrid MSE + λ·Perceptual (VGG)  │
                      └──────────────────────┬───────────────────────┘
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │    High-Resolution Reconstructed Frame (HR)  │
                      │       Evaluation: PSNR / SSIM on Y-Channel   │
                      └──────────────────────────────────────────────┘
```

- **SRCNN (Dong et al., 2014)**: Three-layer convolutional architecture (patch extraction, non-linear mapping, reconstruction) operating on bicubic upsampled inputs.
- **ESPCN (Shi et al., 2016)**: Sub-pixel convolution architecture extracting feature maps directly in low-resolution space and performing fast upscaling via `PixelShuffle`, achieving **sub-15ms real-time inference** on an RTX 2050 GPU.
- **Perceptual Loss**: Deep feature representation loss computed using frozen VGG11 layers (`relu2_2` and `relu3_3`), recovering high-frequency edges and textures in surveillance crops that MSE smoothing typically discards.

---

## Experimental Benchmark Matrix

All models are evaluated on the luminance ($Y$) channel after RGB $\to$ YCbCr conversion with borders shaved by the scale factor, strictly following the academic benchmark standard.

| Exp ID | Architecture | Scale | Loss Function | Set5 PSNR (dB) | Set14 PSNR (dB) | BSD100 PSNR (dB) | Inference Latency |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | SRCNN | $2\times$ | MSE | 34.20 dB | 30.12 dB | 29.80 dB | ~18 ms |
| **E2** | SRCNN | $2\times$ | MSE + Perceptual | 32.85 dB | 29.10 dB | 28.74 dB | ~18 ms |
| **E3** | SRCNN | $4\times$ | MSE | 28.75 dB | 26.24 dB | 25.92 dB | ~21 ms |
| **E4** | SRCNN | $4\times$ | MSE + Perceptual | 27.40 dB | 25.18 dB | 24.85 dB | ~21 ms |
| **E5** | ESPCN | $2\times$ | MSE | 34.62 dB | 30.45 dB | 30.05 dB | **~11 ms** |
| **E6** | ESPCN | $2\times$ | MSE + Perceptual | 33.10 dB | 29.35 dB | 28.95 dB | **~11 ms** |
| **E7** | ESPCN | $4\times$ | MSE | 29.15 dB | 26.58 dB | 26.20 dB | **~14 ms** |
| **E8** | ESPCN | $4\times$ | MSE + Perceptual | 27.80 dB | 25.42 dB | 25.10 dB | **~14 ms** |
| *Ref* | *Bicubic* | $2\times$ | *Baseline* | *33.66 dB* | *29.87 dB* | *29.56 dB* | *~4 ms* |
| *Ref* | *Bicubic* | $4\times$ | *Baseline* | *28.42 dB* | *26.00 dB* | *25.96 dB* | *~5 ms* |

> **Perceptual Loss Trade-Off**: Perceptual models yield lower pixel-level PSNR (due to minor phase/intensity variations) but produce significantly sharper textures, cleaner license plate boundaries, and recognizable facial features on degraded surveillance footage.

---

## Interactive Forensic Studio UI

The web application provides a forensic analyst workbench built with React, Vite, Tailwind CSS, Lucide icons, and Framer Motion:

1. **Interactive Curtain Split Slider**: Hardware-accelerated subpixel slider comparing raw pixelated surveillance input, bicubic interpolation, and neural reconstruction side-by-side.
2. **Surveillance Simulation Workflow**: Automatic detection of uploaded image dimensions. High-resolution photos are realistically downsampled ($4\times$) with sensor blur and compression noise before reconstruction, accompanied by real-time PSNR/SSIM calculation against the original ground truth.
3. **4-Way Experiment Matrix**: Simultaneous side-by-side evaluation across SRCNN MSE, SRCNN Perceptual, ESPCN MSE, and ESPCN Perceptual.
4. **$4\times$ Reticle Loupe**: Real-time synchronized ROI magnifier inspecting fine edge details across Input, Bicubic, Model, and Ground Truth.
5. **Residual Error Heatmap**: Pixel-difference heatmap ($|\text{Model} - \text{GroundTruth}|$) highlighting hallucinated and recovered structures.

---

## Quick Start

### 1. Prerequisites & Environment Setup
Clone the repository and install Python dependencies:
```bash
git clone https://github.com/krish10a/Super-resolution.git
cd Super-resolution

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.venv\Scripts\Activate.ps1      # On Windows PowerShell

# Install PyTorch and dependencies
pip install -r requirements.txt
```

### 2. Start the Backend API Server
The FastAPI backend serves the pre-trained checkpoints with CUDA acceleration:
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8001 --reload
```
API documentation is automatically available at `http://127.0.0.1:8001/docs`.

### 3. Launch the Interactive Frontend Workbench
In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
Open `http://127.0.0.1:5173` in your browser to access the SISR Studio.

---

## Command-Line Inference & Evaluation

### Single Image Super-Resolution
Enhance a low-resolution image using any trained checkpoint:
```bash
python infer/predict.py --model outputs/checkpoints/espcn_4x_mse.pt --img path/to/input_lr.png --out output_sr.png
```

### Run Benchmarks & Robustness Evaluation
```bash
# Run comprehensive evaluation across all 8 models on Set5, Set14, BSD100
python evaluate_all.py

# Evaluate robustness against surveillance sensor degradation (blur, noise, compression)
python evaluate_surveillance.py

# Quick evaluation for best models
python eval_srcnn.py
python eval_best.py
```

### Real-Time Webcam / Video Stream Demo
Super-resolve a live camera feed or surveillance RTSP stream in real-time:
```bash
python infer/webcam_demo.py --model outputs/checkpoints/espcn_2x_mse.pt --camera-id 0
```

### Unit Test Suite
Run the automated test suite covering models, losses, metrics, dataset pipelines, and API security:
```bash
python -m pytest tests/
```

---

## Training Pipeline

To reproduce training from scratch:
1. **Download and prepare datasets**:
   ```bash
   python data/download_sets.py
   python data/prepare_pairs.py
   ```
2. **Train model (e.g. Experiment E7 - ESPCN 4x MSE)**:
   ```bash
   python train/train_all.py --config configs/experiments.yaml --exp E7
   ```
3. **Train with degradation simulation (E7/E8)**:
   ```bash
   python train/train_all.py --config configs/experiments.yaml --exp E7 --degrade
   ```
4. **Google Colab**: Use the included [`colab_train.ipynb`](colab_train.ipynb) notebook for free T4 GPU training with automated Google Drive checkpoint synchronization.

---

## Repository Structure

```
.
├── backend/                  # FastAPI REST microservice & inference pipeline
│   ├── app/
│   │   ├── main.py           # API routes (/super-resolve, /bicubic, /health, /models)
│   │   ├── inference.py      # Pre/post-processing & model inference engine
│   │   ├── registry.py       # Checkpoint loader & metadata registry
│   │   └── security.py       # File validation & upload security checks
│   └── test_backend.py       # Backend pytest test suite
├── configs/                  # Hyperparameter configurations (experiments.yaml)
├── data/                     # Dataset loaders, degradation simulator & preprocessing
│   ├── dataset.py            # PyTorch PatchDataset with RAM caching
│   ├── degradation.py        # Surveillance sensor blur, noise & JPEG degradation
│   ├── download_sets.py      # Automated benchmark dataset downloader
│   └── prepare_pairs.py      # Precomputed LR/HR pair generator
├── frontend/                 # Interactive React + Vite + Tailwind workbench
│   ├── public/               # Synced evaluation logs and demo sample images
│   └── src/
│       ├── pages/            # Studio, Benchmarks, Models, Method, Overview
│       ├── hooks/            # useUpscale, useMagnifier, useResults
│       └── lib/              # API client, live canvas metrics, Turbo colormaps
├── infer/                    # CLI inference, benchmark, webcam demo & ONNX exporter
├── losses/                   # Loss functions (MSE, VGG11 Perceptual Loss)
├── metrics/                  # BT.601 Y-channel PSNR & SSIM with border shaving
├── models/                   # PyTorch neural architectures (SRCNN, ESPCN, VGG)
├── outputs/
│   ├── checkpoints/          # Canonical pre-trained model weights (E1–E8)
│   └── evaluation/           # Verified evaluation JSON, logs & summary tables
├── scripts/                  # Helper utilities and data alignment tools
├── tests/                    # 100+ automated unit tests (Pytest)
├── train/                    # Training curriculum & trainer implementation
├── colab_train.ipynb         # Google Colab GPU training notebook
├── evaluate_all.py           # Benchmark evaluator CLI
├── evaluate_surveillance.py   # Surveillance degradation robustness CLI
├── PROJECT_CONTEXT.md        # Single source of truth with complete technical details
└── README.md
```

---

## Citations & References

- **SRCNN**: Dong, C., Loy, C. C., He, K., & Tang, X. (2014). *Learning a Deep Convolutional Network for Image Super-Resolution*. In ECCV.
- **ESPCN**: Shi, W., Caballero, J., Huszár, F., et al. (2016). *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network*. In CVPR.
- **Perceptual Loss**: Johnson, J., Alahi, A., & Fei-Fei, L. (2016). *Perceptual Losses for Real-Time Style Transfer and Super-Resolution*. In ECCV.

---

## License

This project is licensed under the [MIT License](LICENSE).
