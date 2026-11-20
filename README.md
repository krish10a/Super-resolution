# Deep Learning Image Super-Resolution for Low-Quality Surveillance Feeds

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61dafb?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646cff?logo=vite&logoColor=white)](https://vitejs.dev/)
[![CUDA](https://img.shields.io/badge/CUDA-RTX%202050%20%C2%B7%2012.8-76b900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Single-Image Super-Resolution (SISR) deep learning system designed for forensic enhancement of low-quality surveillance camera feeds. Features **SRCNN**, **ESPCN**, **FSRCNN**, and **EDSR-Baseline** models ($2\times$ and $4\times$ scaling) trained on **DIV2K** under pixel **Charbonnier / L1 loss** and **Perceptual loss** (VGG11 feature extraction), accompanied by a FastAPI inference microservice and an interactive forensic web workbench.

---

## Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │    Low-Resolution Surveillance Feed (LR)     │
                      └──────────────────────┬───────────────────────┘
                                             │
         ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
         ▼                   ▼                               ▼                   ▼
┌──────────────────┐┌──────────────────┐            ┌──────────────────┐┌──────────────────┐
│  SRCNN Pipeline  ││  FSRCNN Pipeline │            │  ESPCN Pipeline  ││  EDSR Pipeline   │
│                  ││                  │            │                  ││                  │
│• Bicubic Upsample││• Feature Extract │            │• LR Conv Feature ││• 16 ResBlocks    │
│• 3 Conv Layers   ││• Shrink/Map/Expand│            │  Extraction      ││• 64 Features     │
│• Pixel Space SR  ││• Deconvolution   │            │• PixelShuffle SR ││• PixelShuffle SR │
└────────┬─────────┘└────────┬─────────┘            └────────┬─────────┘└────────┬─────────┘
         │                   │                               │                   │
         └───────────────────┼───────────────────────────────┴───────────────────┘
                             ▼
                      ┌──────────────────────────────────────────────┐
                      │      Loss Formulation & Optimization         │
                      │                                              │
                      │  • Phase 3: Charbonnier & L1 (DIV2K, 50 Ep)  │
                      │  • Phase 1/2: MSE & Hybrid Perceptual (VGG11)│
                      └──────────────────────┬───────────────────────┘
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │    High-Resolution Reconstructed Frame (HR)  │
                      │   Evaluation: BT.601 Y-Channel PSNR / SSIM   │
                      └──────────────────────┬───────────────────────┘
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │    Surveillance Pre-Filtering (Phase 4.3)    │
                      │  • Bilateral Denoising: Rescues Sensor Noise │
                      │    (+1.35 dB, 0.86 ms latency, >300 FPS)     │
                      └──────────────────────────────────────────────┘
```

- **ESPCN (Shi et al., 2016)**: Sub-pixel convolution architecture extracting feature maps directly in low-resolution space and performing fast upscaling via `PixelShuffle`, achieving **sub-6ms real-time inference** on an RTX 2050 GPU. Ideal for edge CCTV feeds.
- **EDSR-Baseline (Lim et al., 2017)**: Deep residual network without Batch Normalization (16 residual blocks, 64 feature maps), delivering maximum structural reconstruction and license plate/facial edge fidelity (**+3.10 dB over Bicubic** on Set5).
- **FSRCNN (Dong et al., 2016)**: Compact hourglass architecture with feature extraction, shrinking, mapping, and deconvolutional upscaling.
- **SRCNN (Dong et al., 2014)**: Classic three-layer convolutional architecture operating on bicubic upsampled inputs.

---

## Authoritative Canonical Benchmark Matrix

All models are evaluated on the luminance ($Y$) channel after RGB $\to$ YCbCr (BT.601) conversion with borders shaved by the scale factor, strictly following the academic benchmark standard.

### Scale 2× Benchmark Results

| Model ID | Architecture | Loss | Parameters | Set5 ($n=5$) PSNR / SSIM | Set14 ($n=14$) PSNR / SSIM | BSD100 ($n=100$) PSNR / SSIM | Inference Latency | Gain vs Bicubic (Set5) |
|:---:|:---|:---:|---:|:---:|:---:|:---:|:---:|:---:|
| **REF-B2** | *Bicubic Baseline* | — | 0 | 33.64 / 0.9369 | 30.31 / 0.8814 | 29.55 / 0.8569 | 1.2 ms | *Baseline* |
| **E9** | **ESPCN** | Charbonnier | 26,796 | **35.58 / 0.9536** | **31.80 / 0.9090** | **30.85 / 0.8927** | **5.7 ms** | **+1.94 dB** |
| **E10** | **FSRCNN** | Charbonnier | 24,683 | **34.81 / 0.9478** | **31.29 / 0.9043** | **30.51 / 0.8883** | **12.6 ms** | **+1.17 dB** |
| **E11** | **EDSR-Baseline** | L1 | 1,369,859 | **36.75 / 0.9607** | **32.43 / 0.9166** | **31.39 / 0.9000** | **43.0 ms** | **+3.10 dB** |
| *E5 (Legacy)* | ESPCN ($64\to 64$) | MSE | 48,716 | 31.99 / 0.9021 | 29.39 / 0.8589 | 29.29 / 0.8545 | 7.0 ms | -1.65 dB |
| *E1 (Legacy)* | SRCNN | MSE | 20,099 | 28.85 / 0.9044 | 27.34 / 0.8527 | 26.72 / 0.8342 | 66.5 ms | -4.79 dB |

### Scale 4× Benchmark Results

| Model ID | Architecture | Loss | Parameters | Set5 ($n=5$) PSNR / SSIM | Set14 ($n=14$) PSNR / SSIM | BSD100 ($n=100$) PSNR / SSIM | Inference Latency | Gain vs Bicubic (Set5) |
|:---:|:---|:---:|---:|:---:|:---:|:---:|:---:|:---:|
| **REF-B4** | *Bicubic Baseline* | — | 0 | 28.42 / 0.8230 | 26.08 / 0.7216 | 25.96 / 0.6856 | 1.0 ms | *Baseline* |
| **E12** | **ESPCN** | Charbonnier | 37,200 | **29.37 / 0.8444** | **26.74 / 0.7497** | **26.46 / 0.7171** | **4.7 ms** | **+0.95 dB** |
| **E13** | **FSRCNN** | Charbonnier | 24,683 | **29.06 / 0.8385** | **26.52 / 0.7459** | **26.36 / 0.7138** | **9.4 ms** | **+0.64 dB** |
| **E14** | **EDSR-Baseline** | L1 | 1,517,571 | **30.70 / 0.8809** | **27.63 / 0.7752** | **26.99 / 0.7352** | **19.5 ms** | **+2.28 dB** |
| *E7 (Legacy)* | ESPCN ($64\to 64$) | MSE | 69,488 | 27.15 / 0.7882 | 25.07 / 0.6908 | 25.11 / 0.6563 | 8.1 ms | -1.27 dB |
| *E3 (Legacy)* | SRCNN | MSE | 20,099 | 26.30 / 0.8170 | 24.58 / 0.7222 | 24.55 / 0.6903 | 17.8 ms | -2.11 dB |

---

## Statistical Robustness & Surveillance Validation (Phase 4.2)

Evaluated across 9,120 image inferences with **10,000 paired bootstrap resamples** (95% CI) under 5 realistic surveillance degradation regimes:

| Degradation Regime | Parameters | Bicubic (BSD100 2×) | E11 EDSR (BSD100 2×) | $\Delta\text{PSNR vs Bicubic}$ | 95% Bootstrap CI | Win Rate | Conclusion |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Clean** | Canonical antialiased | 29.55 dB | **31.39 dB** | **+1.84 dB** | `[+1.67, +2.04]` | 100% | Statistically significant superiority |
| **Motion Blur** | $k=5, \theta=45^\circ$ | 27.46 dB | **28.20 dB** | **+0.73 dB** | `[+0.69, +0.78]` | 100% | Decisive edge deblurring |
| **JPEG Compression** | Quality $Q=40$ | 27.40 dB | 27.27 dB | -0.14 dB | `[-0.17, -0.10]` | 16% | Artifact amplification |
| **Sensor Noise** | Gaussian $\sigma=10.0$ | 28.14 dB | 27.52 dB | -0.62 dB | `[-0.80, -0.46]` | 24% | High-frequency noise amplified |
| **Combined CCTV** | Blur + Noise + JPEG | 26.41 dB | 26.41 dB | -0.00 dB | `[-0.03, +0.03]` | 46% | Inconclusive without pre-filtering |

---

## Denoising Ablation Study (Phase 4.3)

Evaluated pre-filtering on degraded low-resolution inputs prior to SR across 4,104 inferences:

| Pre-Filtering Strategy | Mechanism | Added Latency | E11 PSNR (Sensor Noise) | Gain vs Direct SR (A0) | Gain vs Bicubic Baseline | Operational Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **A0: Direct SR** | None (raw input) | 0.00 ms | 27.52 dB | 0.00 dB | -0.62 dB (Loss) | Fails on noisy feeds |
| **A1: Non-Local Means** | OpenCV NLM ($h=3$) | 203.39 ms | 27.67 dB | +0.15 dB | -0.48 dB (Loss) | Too slow for real-time CCTV |
| **A2: Bilateral Filter** | OpenCV Bilateral ($d=5, \sigma=25$) | **0.86 ms** | **28.87 dB** | **+1.35 dB** (`[+1.17, +1.55]`) | **+0.73 dB** (`[+0.56, +0.90]`) | **Production standard for CCTV feeds** |

> **Key Operational Takeaway**: Prepending an ultra-fast Bilateral filter ($0.86\text{ ms}$) eliminates sensor noise and converts a $-0.62\text{ dB}$ degradation into a **$+0.73\text{ dB}$ statistically significant victory over Bicubic**, running at **$>300\text{ FPS}$ total pipeline latency**.

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
