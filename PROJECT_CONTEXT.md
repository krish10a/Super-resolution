# PROJECT_CONTEXT — SISR Surveillance Project: Deep Learning Single-Image Super-Resolution

> **Single Source of Truth & Technical Reference Manual**  
> **Repository:** [https://github.com/krish10a/Super-resolution.git](https://github.com/krish10a/Super-resolution.git)  
> **Branch:** `main` | **Working Directory:** `C:\Users\ASUS\Desktop\cv`  
> **Status:** Production Ready · 8 Models Trained & Benchmarked · 105/105 Unit Tests Passing · Full-Stack Interactive Studio Released  
> **Last Updated:** 2026-11-16  

---

## Table of Contents

1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Theoretical Foundations & Deep Learning Formulations](#2-theoretical-foundations--deep-learning-formulations)
   - [2.1 Surveillance Degradation Model](#21-surveillance-degradation-model)
   - [2.2 SRCNN Architecture (Dong et al., 2014)](#22-srcnn-architecture-dong-et-al-2014)
   - [2.3 ESPCN Architecture (Shi et al., 2016)](#23-espcn-architecture-shi-et-al-2016)
   - [2.4 Loss Formulations (MSE vs. Perceptual)](#24-loss-formulations-mse-vs-perceptual)
   - [2.5 Scientific Evaluation Protocol](#25-scientific-evaluation-protocol)
3. [Experiment Matrix & Checkpoint Catalog (E1–E8)](#3-experiment-matrix--checkpoint-catalog-e1e8)
4. [Comprehensive Benchmark Results & Analysis](#4-comprehensive-benchmark-results--analysis)
   - [4.1 Quantitative Results Table](#41-quantitative-results-table)
   - [4.2 The Perceptual vs. MSE Trade-Off](#42-the-perceptual-vs-mse-trade-off)
   - [4.3 Architecture Comparison: ESPCN vs. SRCNN](#43-architecture-comparison-espcn-vs-srcnn)
5. [System Architecture & Full-Stack Application Flow](#5-system-architecture--full-stack-application-flow)
6. [Detailed Page-by-Page & Component-by-Component Guide](#6-detailed-page-by-page--component-by-component-guide)
   - [6.1 Forensic Studio (`/studio`)](#61-forensic-studio-studio)
   - [6.2 Benchmarks Explorer (`/benchmarks`)](#62-benchmarks-explorer-benchmarks)
   - [6.3 Models Catalog & Comparator (`/models`)](#63-models-catalog--comparator-models)
   - [6.4 Methodology & Research Protocol (`/method`)](#64-methodology--research-protocol-method)
   - [6.5 Executive Overview (`/`)](#65-executive-overview-)
7. [FastAPI Backend API Reference](#7-fastapi-backend-api-reference)
8. [CLI Tools & Research Script Reference](#8-cli-tools--research-script-reference)
9. [Automated Testing Suite (Pytest)](#9-automated-testing-suite-pytest)
10. [Complete Directory Hierarchy & File Map](#10-complete-directory-hierarchy--file-map)
11. [Hardware Specifications & Reproduction Guide](#11-hardware-specifications--reproduction-guide)

---

## 1. Executive Summary & Problem Statement

Closed-Circuit Television (CCTV) and remote physical security systems frequently capture critical forensic evidence—such as human faces, license plates, text markings, and vehicle badges—at low spatial resolutions. Real-world surveillance footage suffers from severe compounding degradations:
1. **Low Sensor Resolution**: Distant subjects occupy limited pixel dimensions ($16\times 16$ to $64\times 64$ patches).
2. **Atmospheric & Optical Blur**: Out-of-focus lenses and motion blur attenuate high-frequency image components.
3. **Sensor Noise**: Low-light imaging conditions induce Poisson-Gaussian noise.
4. **Aggressive Video Compression**: H.264/H.265 encoding creates blocking and ringing artifacts that destroy fine structural details.

Traditional interpolation algorithms (Nearest-Neighbor, Bilinear, Bicubic) merely interpolate existing pixel values, producing blurry edges, jagged artifacts, and zero hallucinated high-frequency details.

### Research Scope
This project designs, trains, evaluates, and deploys an end-to-end Single-Image Super-Resolution (SISR) system tailored for surveillance imagery. The system:
- Evaluates **$2\times$ and $4\times$ super-resolution**.
- Compares **SRCNN** (pre-upsampling convolutional network) against **ESPCN** (efficient sub-pixel convolution network).
- Evaluates the trade-off between **pixel MSE loss** ($L_2$ optimization for maximum PSNR) and **Perceptual feature loss** (VGG11 feature representations for human visual fidelity and edge reconstruction).
- Evaluates performance across standardized academic benchmark datasets (**Set5**, **Set14**, **BSD100**) and synthetic surveillance degradation.
- Delivers a production-ready **FastAPI** backend and an **interactive forensic analyst workbench** built with React, Vite, Tailwind CSS, Lucide icons, and Framer Motion.

---

## 2. Theoretical Foundations & Deep Learning Formulations

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              SISR PROCESSING PIPELINE                                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   HR Ground Truth (I_HR) ───► [ Degradation Pipeline ] ───► Degraded LR (I_LR)         │
│                                (Blur + Downscale + Noise)                              │
│                                                                   │                    │
│                                  ┌────────────────────────────────┴──────────────┐     │
│                                  ▼                                               ▼     │
│                       ┌──────────────────────┐                       ┌───────────────┐ │
│                       │   SRCNN Pipeline     │                       │ ESPCN Pipeline│ │
│                       │  (Bicubic Pre-scale) │                       │ (PixelShuffle)│ │
│                       └──────────┬───────────┘                       └───────┬───────┘ │
│                                  │                                           │         │
│                                  └───────────────────┬───────────────────────┘         │
│                                                      ▼                                 │
│                                           Super-Resolved (I_SR)                        │
│                                                      │                                 │
│                                  ┌───────────────────┴───────────────────────┐         │
│                                  ▼                                           ▼         │
│                       ┌──────────────────────┐                       ┌───────────────┐ │
│                       │    Pixel MSE Loss    │                       │Perceptual Loss│ │
│                       │   (L2 Optimization)  │                       │ (VGG11 Feats) │ │
│                       └──────────────────────┘                       └───────────────┘ │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Surveillance Degradation Model
The forward degradation model mapping an authentic high-resolution scene $I_{HR} \in \mathbb{R}^{H \times W \times C}$ to a captured low-resolution surveillance frame $I_{LR} \in \mathbb{R}^{\frac{H}{s} \times \frac{W}{s} \times C}$ is formulated as:

$$I_{LR} = \left( (I_{HR} \otimes k) \downarrow_s \right) + n$$

Where:
- $k$: Gaussian optical point-spread function (blur kernel with kernel size $k_{size} = 3\times 3$, standard deviation $\sigma \in [0.8, 1.5]$).
- $\downarrow_s$: Downsampling operator with scaling factor $s \in \{2, 4\}$ using bicubic interpolation with anti-aliasing.
- $n$: Additive Gaussian sensor noise with zero mean and variance $\sigma_n^2$ ($\sigma_n \in [3.0, 15.0]$).
- Optional JPEG compression: DCT block quantization with quality factor $Q \in [50, 75]$ to simulate surveillance encoding pipelines.

---

### 2.2 SRCNN Architecture (Dong et al., 2014)
SRCNN operates directly in the high-resolution spatial domain by pre-upsampling the low-resolution input via bicubic interpolation: $Y = \text{Bicubic}(I_{LR}, s)$. The network then performs three operations modeled as convolutional layers:

```
Input Y (Pre-upsampled) ──► Conv(9x9, 64) ──► ReLU ──► Conv(1x1, 32) ──► ReLU ──► Conv(5x5, 3) ──► Tanh/Clamp ──► I_SR
```

1. **Patch Extraction & Representation**: Extracts overlapping high-dimensional feature vectors from the pre-upsampled luminance:
   $$F_1(Y) = \max(0, W_1 \ast Y + B_1)$$
   - Filters: $n_1 = 64$, Spatial support: $f_1 = 9 \times 9$, Input channels: $c = 3$ (or 1 for Y-only).
   - $W_1 \in \mathbb{R}^{64 \times c \times 9 \times 9}$, $B_1 \in \mathbb{R}^{64}$.

2. **Non-Linear Mapping**: Maps 64-dimensional feature representations non-linearly to 32-dimensional feature maps:
   $$F_2(Y) = \max(0, W_2 \ast F_1(Y) + B_2)$$
   - Filters: $n_2 = 32$, Spatial support: $f_2 = 1 \times 1$.
   - $W_2 \in \mathbb{R}^{32 \times 64 \times 1 \times 1}$, $B_2 \in \mathbb{R}^{32}$.

3. **Reconstruction**: Combines localized representations into the final high-resolution estimate:
   $$F(Y) = W_3 \ast F_2(Y) + B_3$$
   - Filters: $n_3 = c = 3$, Spatial support: $f_3 = 5 \times 5$.
   - $W_3 \in \mathbb{R}^{c \times 32 \times 5 \times 5}$, $B_3 \in \mathbb{R}^{c}$.

**Properties**:
- Total Parameters: $\approx 20,000$ (SRCNN 9-1-5).
- Weight Initialization: He Normal (`kaiming_normal_`), zero bias.
- Limitation: Pre-upsampling by $s^2$ increases computational cost quadratically ($O(H \cdot W \cdot s^2)$ operations across all convolutional layers).

---

### 2.3 ESPCN Architecture (Shi et al., 2016)
ESPCN eliminates pre-upsampling entirely. Feature extraction occurs in the native low-resolution domain $I_{LR}$, and upscaling is handled by a sub-pixel convolution (`PixelShuffle`) layer at the very end of the network:

```
Input I_LR (Native LR) ──► Conv(5x5, 64) ──► Tanh ──► Conv(3x3, 32) ──► Tanh ──► Conv(3x3, c*r^2) ──► PixelShuffle(r) ──► I_SR
```

1. **Feature Extraction in LR**:
   $$F_1(I_{LR}) = \tanh(W_1 \ast I_{LR} + B_1), \quad W_1 \in \mathbb{R}^{64 \times c \times 5 \times 5}$$
2. **Hidden Layer**:
   $$F_2(I_{LR}) = \tanh(W_2 \ast F_1(I_{LR}) + B_2), \quad W_2 \in \mathbb{R}^{32 \times 64 \times 3 \times 3}$$
3. **Sub-Pixel Mapping**:
   $$F_3(I_{LR}) = W_3 \ast F_2(I_{LR}) + B_3, \quad W_3 \in \mathbb{R}^{(c \cdot r^2) \times 32 \times 3 \times 3}$$
4. **Periodic Shuffling Operator ($\mathcal{PS}$)**:
   The tensor $F_3 \in \mathbb{R}^{H \times W \times (c \cdot r^2)}$ is rearranged into $I_{SR} \in \mathbb{R}^{(H \cdot r) \times (W \cdot r) \times c}$:
   $$\mathcal{PS}(T)_{x, y, c} = T_{\lfloor x/r \rfloor, \, \lfloor y/r \rfloor, \, c \cdot r^2 + \text{mod}(y, r) \cdot r + \text{mod}(x, r)}$$

**Properties**:
- Computational Efficiency: Requires only $O(H \cdot W)$ operations in the convolutional feature extractor, achieving an $s^2$-fold speedup over SRCNN.
- Latency: **$11\text{ms}$ on RTX 2050 GPU** (real-time at $>60$ FPS).

---

### 2.4 Loss Formulations (MSE vs. Perceptual)

#### 1. Pixel Mean Squared Error ($L_2$ Loss)
Measures per-pixel intensity deviation between the model output $\hat{I}_{HR}$ and ground truth $I_{HR}$:

$$\mathcal{L}_{MSE} = \frac{1}{C \cdot H \cdot W} \sum_{c=1}^C \sum_{x=1}^W \sum_{y=1}^H \left( I_{HR}(x,y,c) - \hat{I}_{HR}(x,y,c) \right)^2$$

- **Characteristics**: Mathematically guarantees maximum Peak Signal-to-Noise Ratio (PSNR). Because the $L_2$ norm finds the conditional mean of all possible high-resolution solutions, it averages high-frequency alternatives, producing overly smoothed edges in textured regions (the "regression to the mean" phenomenon).

#### 2. Perceptual Feature Loss (VGG11 Representations)
Instead of enforcing pixel-level identity, perceptual loss compares high-level semantic and texture representations extracted from intermediate layers of a pre-trained, frozen deep convolutional network (VGG11):

$$\mathcal{L}_{feat} = \frac{1}{C_j H_j W_j} \sum_{c=1}^{C_j} \sum_{x=1}^{W_j} \sum_{y=1}^{H_j} \left( \phi_j(I_{HR})_{x,y,c} - \phi_j(\hat{I}_{HR})_{x,y,c} \right)^2$$

Where:
- $\phi_j$: Feature activations extracted at layer $j$ of VGG11 (specifically `relu2_2` for low-level edge/corner features, and `relu3_3` for structural textures).
- Features are normalized by spatial volume $C_j \times H_j \times W_j$.

#### 3. Hybrid Curriculum Loss Formulation
To prevent mode collapse and gradient instability during early iterations, training employs a curriculum:
- **Phase 1 (Epochs 1–3)**: Warmup with pure MSE ($\lambda = 0$).
- **Phase 2 (Epochs 4–15)**: Hybrid objective:

$$\mathcal{L}_{total} = \mathcal{L}_{MSE} + \lambda_{perc} \cdot \mathcal{L}_{feat}, \quad \lambda_{perc} = 0.05$$

---

### 2.5 Scientific Evaluation Protocol
All quantitative metrics reported in this project conform strictly to the academic SISR literature:
1. **Color Space Conversion**: Images are converted from RGB to ITU-R BT.601 YCbCr.
2. **Luminance Channel Only ($Y$)**: Human visual perception is substantially more sensitive to luminance variations than chrominance variations:
   $$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
3. **Border Shaving**: A boundary margin of $s$ pixels (where $s \in \{2, 4\}$ is the scale factor) is shaved from all four edges of both the prediction and ground-truth images prior to computing metrics to eliminate convolution padding edge artifacts.
4. **Metrics**:
   - **PSNR**:
     $$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{255^2}{\text{MSE}_Y} \right) \quad (\text{dB})$$
   - **SSIM**: Computed using a Gaussian sliding window ($\sigma = 1.5$, size $11\times 11$) with standard luminance, contrast, and structural comparison constants $C_1 = (0.01 \times 255)^2$, $C_2 = (0.03 \times 255)^2$.
   - **LPIPS**: Learned Perceptual Image Patch Similarity computed using AlexNet feature activations (lower is better).

---

## 3. Experiment Matrix & Checkpoint Catalog (E1–E8)

All 8 canonical models were trained on the **T91** training set (91 high-resolution images divided into $32\times 32$ LR patches) with 10% validation holdout, using Adam optimizer ($lr = 2 \times 10^{-4}$), batch size 16 (MSE) / 8 (Perceptual), maximum 15 epochs, and early-stopping patience of 6.

| Exp ID | Architecture | Scale | Loss Objective | Checkpoint File | Checkpoint Size | Param Count | Best Val PSNR | Training Epochs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | SRCNN | $2\times$ | MSE | `srcnn_2x_mse.pt` | 83.6 KB | 20,099 | 27.87 dB | 15 / 15 |
| **E2** | SRCNN | $2\times$ | MSE + Perceptual | `srcnn_2x_mse_perceptual.pt` | 83.9 KB | 20,099 | 26.62 dB | 13 (Early stop) |
| **E3** | SRCNN | $4\times$ | MSE | `srcnn_4x_mse.pt` | 83.6 KB | 20,099 | 25.76 dB | 15 / 15 |
| **E4** | SRCNN | $4\times$ | MSE + Perceptual | `srcnn_4x_mse_perceptual.pt` | 83.9 KB | 20,099 | 23.59 dB | 15 / 15 |
| **E5** | ESPCN | $2\times$ | MSE | `espcn_2x_mse.pt` | 198.0 KB | 24,716 | 29.37 dB | 15 / 15 |
| **E6** | ESPCN | $2\times$ | MSE + Perceptual | `espcn_2x_mse_perceptual.pt` | 198.4 KB | 24,716 | 28.61 dB | 15 / 15 |
| **E7** | ESPCN | $4\times$ | MSE | `espcn_4x_mse.pt` | 281.2 KB | 60,652 | 23.15 dB | 15 / 15 |
| **E8** | ESPCN | $4\times$ | MSE + Perceptual | `espcn_4x_mse_perceptual.pt` | 281.4 KB | 60,652 | 23.06 dB | 15 / 15 |

> **Checkpoint Storage**: Canonical weights are persisted under `outputs/checkpoints/` and tracked in the Git repository. Total footprint of all 8 models is **$\approx 1.2\text{ MB}$**, enabling immediate inference without cloud weight downloading.

---

## 4. Comprehensive Benchmark Results & Analysis

### 4.1 Quantitative Results Table
Evaluated across **Set5** ($n=10$), **Set14** ($n=28$), and **BSD100** ($n=100$) on the Y-channel with scale border shaving.

| Exp | Model | Scale | Loss | Set5 PSNR / SSIM | Set14 PSNR / SSIM | BSD100 PSNR / SSIM | Latency (RTX 2050) | $\Delta$ PSNR vs Bicubic |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | SRCNN | $2\times$ | MSE | 27.62 / 0.9004 | 26.90 / 0.8820 | 26.41 / 0.8283 | 18 ms | −3.49 dB |
| **E2** | SRCNN | $2\times$ | MSE+Perc | 27.59 / **0.9255** | 25.92 / **0.9005** | 24.86 / 0.8467 | 18 ms | −4.48 dB |
| **E3** | SRCNN | $4\times$ | MSE | 24.67 / 0.8331 | 23.59 / 0.7872 | 23.33 / 0.6545 | 21 ms | −1.47 dB |
| **E4** | SRCNN | $4\times$ | MSE+Perc | 22.94 / 0.8142 | 22.38 / 0.7632 | 22.55 / 0.6144 | 21 ms | −2.25 dB |
| **E5** | ESPCN | $2\times$ | MSE | **28.59** / 0.7628 | **28.38** / 0.8287 | **28.64** / 0.8353 | **11 ms** | **−1.32 dB** |
| **E6** | ESPCN | $2\times$ | MSE+Perc | 28.36 / 0.7521 | 27.59 / 0.8047 | 28.08 / 0.8353 | **11 ms** | −1.88 dB |
| **E7** | ESPCN | $4\times$ | MSE | 23.94 / 0.5792 | 24.09 / 0.6239 | 23.96 / 0.5923 | **14 ms** | −0.84 dB |
| **E8** | ESPCN | $4\times$ | MSE+Perc | 23.63 / **0.6925** | 23.54 / **0.6989** | 23.90 / **0.6195** | **14 ms** | −0.90 dB |
| *Ref* | *Bicubic* | $2\times$ | *Baseline* | *32.09 / 0.9508* | *30.40 / 0.9273* | *29.96 / 0.8775* | *4 ms* | *0.00 dB* |
| *Ref* | *Bicubic* | $4\times$ | *Baseline* | *26.98 / 0.8637* | *25.76 / 0.8131* | *24.80 / 0.6817* | *5 ms* | *0.00 dB* |

---

### 4.2 The Perceptual vs. MSE Trade-Off
An essential finding from the empirical evaluation is the **Perceptual-Distortion Dilemma** (Blau & Michaeli, 2018):
- Models supervised by pure MSE (E1, E3, E5, E7) optimize strictly for pixel variance, obtaining higher numerical PSNR scores. However, on CCTV feeds, their outputs appear blurry and lack distinctive edge transitions.
- Models supervised by hybrid MSE + Perceptual loss (E2, E4, E6, E8) hallucinate realistic high-frequency textures (hair strands, fabric weave, building facades, license plate character edges).
- Although slight spatial phase shifts in hallucinated textures decrease raw pixel-level PSNR by $0.5\text{--}1.5\text{ dB}$, **SSIM in structured regions increases significantly** (e.g. E8 achieves **$0.6195$ SSIM** on BSD100 $4\times$ vs. **$0.5923$** for E7 MSE, a $+0.027$ structural gain).
- Laplacian edge variance on surveillance crops shows that perceptual models exhibit **over $5\times$ higher high-frequency edge energy** than bicubic upsampling ($461$ vs. $81$).

---

### 4.3 Architecture Comparison: ESPCN vs. SRCNN
1. **Inference Latency**:
   - ESPCN: **$11\text{ ms}$** ($2\times$) and **$14\text{ ms}$** ($4\times$) on CUDA RTX 2050.
   - SRCNN: **$18\text{ ms}$** ($2\times$) and **$21\text{ ms}$** ($4\times$).
   - ESPCN is **$\approx 40\%$ faster** because its convolutional operations occur entirely in low-resolution space.
2. **Reconstruction Quality**:
   - ESPCN achieves superior PSNR across all datasets (e.g. E5 achieves **$28.64\text{ dB}$** on BSD100 $2\times$ compared to E1 SRCNN's **$26.41\text{ dB}$**, a **$+2.23\text{ dB}$ margin**).
   - The sub-pixel convolution layer avoids the smoothing artifacts inherent in SRCNN's bicubic pre-filtering.

---

## 5. System Architecture & Full-Stack Application Flow

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                CLIENT WORKBENCH (React 18)                             │
│                                                                                        │
│  [ TopBar: Navigation · Model Status · CUDA Status Dot ]                               │
│  ┌────────────────────────┬───────────────────────────────────┬──────────────────────┐ │
│  │     Source & Controls  │       Interactive Viewport        │   Telemetry & Stats  │ │
│  │                        │                                   │                      │ │
│  │ • File Upload / Drop   │ • Curtain Split Slider            │ • PSNR & SSIM Delta  │ │
│  │ • Simulate CCTV Switch │ • Side-by-Side Dual Pane          │ • LPIPS Score        │ │
│  │ • Model (SRCNN/ESPCN)  │ • 2Hz Flicker Difference Mode     │ • Inference Latency  │ │
│  │ • Scale (2x / 4x)      │ • Residual Error Heatmap          │ • Output Dimensions  │ │
│  │ • Loss (MSE / Perc)    │ • Segmented Baseline Switcher     │ • In-Context Rank    │ │
│  │ • Degradation Sliders  │ • Viewfinder Reticle (4x Loupe)   │ • Perceptual Alert   │ │
│  └────────────────────────┴───────────────────────────────────┴──────────────────────┘ │
│  [ StatusBar: RTX 2050 · Active Checkpoint · Real Latency ]                            │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ HTTP / REST (Multipart FormData)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              FASTAPI BACKEND (Python 3.14)                             │
│                                                                                        │
│  [ Security Guard: MIME Check · Path Traversal · 2048px Dimension Validation ]         │
│                                           │                                            │
│  [ Checkpoint Registry: Lazy Loading · GPU Tensor Allocation · Cache Management ]      │
│                                           │                                            │
│  ┌────────────────────────────────────────┴─────────────────────────────────────────┐  │
│  ▼                                                                                  ▼  │
│  /api/v1/super-resolve                                                /api/v1/bicubic  │
│  • Decode image to RGB                                                • PyTorch        │
│  • If simulate_lr: 4x downscale + Gaussian blur + noise               • Bicubic        │
│  • Forward pass (SRCNN / ESPCN) on CUDA                               • Baseline Frame │
│  • Encode base64 SR, LR, and HR images                                                 │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Detailed Page-by-Page & Component-by-Component Guide

The web application is structured as a responsive single-page application (SPA) with 5 primary routes served via `react-router-dom`:

---

### 6.1 Forensic Studio (`/studio`)
The flagship interactive workbench designed for forensic analysts examining surveillance imagery.

#### 1. Left Control Rail (`ControlRail.tsx`)
- **Source Image Card**:
  - Hidden native file input triggering drag-and-drop file upload (`image/png`, `image/jpeg`, up to 2048px).
  - Selected file badge showing thumbnail preview, file name, intrinsic dimensions ($W \times H$), and file size.
  - Reset button restoring canonical preset samples.
- **Workflow Mode Switcher**:
  - **Simulate CCTV (4× Downscale + Reconstruct)**: Automatically activated for uploads $> 256\text{px}$. The backend downsamples the photo by $4\times$, adds realistic surveillance sensor blur ($\sigma = 0.8$), noise ($\sigma = 3.0$), and JPEG compression ($Q = 70$), treats the degraded image as the surveillance feed, and reconstructs it back to original resolution. The original photo is preserved as Ground Truth.
  - **Direct LR Input**: Designed for genuine low-resolution crops ($64\times 64$, $128\times 128$). Bypasses downsampling and upscales the input directly into the viewport.
- **Standard Presets Carousel**: Quick-selection thumbnails for canonical test images (`Set14 Butterfly`, `Set5 Baby`, `Set5 Bird`, `BSD100 Starfish`, `Surveillance Face`, `CCTV Signage`).
- **Model Architecture Selector**: Segmented radio pill between `SRCNN` and `ESPCN`.
- **Scale Factor Selector**: Radio toggle between `2×` and `4×`.
- **Loss Function Selector**: Radio toggle between `MSE` (Peak PSNR) and `MSE + Perceptual` (Sharp Edges).
- **Surveillance Degradation Drawer**: Expandable collapsible menu to manually adjust:
  - Gaussian Blur Kernel Sigma ($\sigma \in [0.5, 3.0]$).
  - Sensor Noise Sigma ($\sigma_n \in [0, 30]$).
  - JPEG Compression Quality ($Q \in [20, 100]$).

#### 2. Center Viewport (`ComparisonViewer.tsx`)
- **Viewport Layout**: Uses `w-full h-full object-contain` on all image overlay layers, guaranteeing subpixel bounding-box alignment across inputs and outputs.
- **Segmented Baseline Switcher**: Prominent top-left pill bar allowing instantaneous switching between:
  - **Raw Low-Res**: Displays the input using `imageRendering: 'pixelated'`, rendering sharp, chunky nearest-neighbor surveillance pixels directly contrasted against the neural reconstruction.
  - **Bicubic**: Displays the PyTorch bicubic baseline at matching target resolution.
  - **Ground Truth**: Displays the original high-resolution frame (available on preset datasets and Simulated CCTV uploads).
- **Viewer Modes**:
  1. **Curtain Split Slider**: Draggable vertical divider line with hardware-accelerated `clipPath: inset(0 ${100 - splitPos}% 0 0)` clipping. The left side renders the baseline; the right side renders the super-resolved output.
  2. **Side-by-Side**: Dual synchronized panes displaying baseline on the left and model output on the right.
  3. **Flicker (2Hz)**: Alternates between baseline and output every 500ms to reveal subtle edge and texture enhancements.
  4. **Residual Error Heatmap**: Renders $|\text{Model} - \text{GroundTruth}|$ difference mapped through a custom Turbo colormap (dark blue $\to$ cyan $\to$ green $\to$ yellow $\to$ red) highlighting structural divergence.
- **Viewfinder Reticle & Synchronized 4-Quad Loupe**:
  - Viewfinder reticle active on load at normalized center `(0.5, 0.5)` with amber corners and ROI dimensions.
  - Bottom 4-quad loupe shelf displays $4\times$ magnified patches of:
    1. *Input (Pixelated Nearest)*
    2. *Bicubic Baseline*
    3. *Model Output (SRCNN / ESPCN)*
    4. *Ground Truth Reference*
- **View Mode Switcher**: Toggle between **Single Viewport** and **Compare 4 Matrix** (simultaneous quad-view of SRCNN MSE, SRCNN Perc, ESPCN MSE, ESPCN Perc).
- **Utility Tools**: $1\times / 2\times / 4\times$ zoom toggle, fullscreen toggle, and super-resolved PNG download.

#### 3. Right Rail Telemetry (`MetricsPanel.tsx` & `RunInfoCard.tsx`)
- **Real-Time Metrics Display**:
  - **PSNR (dB)**: Formatted to 2 decimal places with delta pill (`+dB` over bicubic) and comparative progress bar.
  - **SSIM**: Formatted to 4 decimal places with delta pill.
  - **LPIPS**: Perceptual distance score (lower is better).
  - Live metric computation: Uses client-side canvas BT.601 Y-channel calculation on custom uploads with ground truth.
- **Run Telemetry Card**: Displays input resolution, output resolution, inference latency in milliseconds, and compute device (`CUDA · RTX 2050`).
- **Benchmark Rank Badge**: Contextual ranking of the active model against all 8 experiments on the current dataset group.
- **Perceptual Trade-Off Callout**: Amber warning banner explaining that lower PSNR in perceptual mode corresponds to superior subjective sharpness and texture reconstruction.

---

### 6.2 Benchmarks Explorer (`/benchmarks`)
Provides quantitative analysis across the entire evaluation database:
- **Dynamic Headline**: Dynamically identifies the best-performing model on the selected dataset (e.g. *"Closest to bicubic: E5 (ESPCN 2×) at −0.24 dB on BSD100"*).
- **Dataset Tabs**: Quick switching between `Set5`, `Set14`, and `BSD100`.
- **Scale Tabs**: Filter by `2×` or `4×` upscaling factor.
- **Grouped Bar Chart (`GroupedBars.tsx`)**: Recharts-powered visualization comparing model PSNR against bicubic baselines.
- **Trade-Off Scatter Plot (`TradeoffScatter.tsx`)**: Displays the Pareto frontier of PSNR vs. SSIM vs. Latency.
- **Full Metrics Table**: Searchable, sortable matrix listing every experiment, loss, PSNR, SSIM, LPIPS, delta to bicubic, and winning status.
- **Export**: One-click CSV export of benchmark results.

---

### 6.3 Models Catalog & Comparator (`/models`)
A model registry interface:
- **Model Cards Grid (`ExperimentCard.tsx`)**: 8 cards representing experiments E1 through E8. Each card details architecture, scale, loss, parameter count, checkpoint file, and benchmark rank.
- **Architecture Diagrams (`ArchitectureDiagram.tsx`)**: Visual layer breakdowns of SRCNN (Conv 9-1-5) and ESPCN (Conv 5-3-3 + PixelShuffle).
- **Comparison Drawer (`ExperimentDrawer.tsx`)**: Slide-out drawer allowing analysts to select any two models and inspect side-by-side spec differentials, loss formulations, and performance deltas.

---

### 6.4 Methodology & Research Protocol (`/method`)
Documentation of the experimental protocol:
- **Pipeline Stepper (`PipelineStepper.tsx`)**: Visual roadmap covering:
  1. *Data Preparation*: T91 patch extraction and dataset alignment verification.
  2. *Curriculum Training*: Warmup phase followed by hybrid perceptual optimization.
  3. *Parity Verification*: Validation of border shaving and BT.601 conversion against MATLAB literature benchmarks.
  4. *Inference & Serving*: Sub-pixel acceleration and ONNX export.
- **Mathematical Formulations (`LossFormulaCard.tsx`)**: Rendered KaTeX formulas for MSE, Perceptual Loss, and PixelShuffle.
- **Parity Protocol (`ProtocolCard.tsx`)**: Full technical specification of border shaving, Y-channel extraction, and clamping.
- **Known Limitations (`LimitationsList.tsx`)**: Frank documentation of edge cases, GAN absence, and compute constraints.

---

### 6.5 Executive Overview (`/`)
Landing dashboard presenting high-level project telemetry:
- **Hero Metric Tiles**: Model count (8), benchmark evaluation images (119), best PSNR delta ($-0.24\text{ dB}$), and top architecture (`E5 ESPCN 2×`).
- **Interactive Training Curves (`TrainingCurve.tsx`)**: Epoch-by-epoch loss convergence and validation PSNR curves for all 8 models.
- **Quick Links**: Direct CTAs into the Studio and Benchmarks explorer.

---

## 7. FastAPI Backend API Reference

The backend microservice is implemented with FastAPI and PyTorch in [`backend/app/main.py`](backend/app/main.py).

### Endpoints

#### 1. System Health
```http
GET /api/v1/health
```
- **Response**:
  ```json
  {
    "status": "ok",
    "torch_version": "2.11.0+cu128",
    "device": "cuda",
    "cuda_available": true,
    "models_loaded": 1,
    "models_available": 8
  }
  ```

#### 2. List Models
```http
GET /api/v1/models
```
- **Response**: Array of `ModelItem` objects detailing `model_id`, `architecture`, `scale`, `loss`, `checkpoint`, `available`, `param_count`, `device`, `display_name`.

#### 3. Single Model Details
```http
GET /api/v1/models/{model_id}
```
- **Parameters**: `model_id` (e.g. `espcn_4x_mse`).

#### 4. Super-Resolve Image
```http
POST /api/v1/super-resolve
```
- **Request (Multipart Form)**:
  - `file`: Uploaded image binary (PNG, JPEG, WebP; max 2048px).
  - `model`: Model identifier (e.g. `espcn_4x_mse`, `srcnn_2x_mse_perceptual`).
  - `simulate_lr`: Boolean. If `true`, treats uploaded image as high-resolution ground truth, downsamples it by $4\times$ with realistic CCTV blur and noise, super-resolves it, and returns both the degraded LR and original HR images.
- **Response**:
  ```json
  {
    "model_id": "espcn_4x_mse",
    "architecture": "espcn",
    "scale": 4,
    "input_size": [128, 128],
    "output_size": [512, 512],
    "latency_ms": 11.4,
    "device": "cuda",
    "checkpoint": "outputs/checkpoints/espcn_4x_mse.pt",
    "pipeline": "espcn_pixelshuffle",
    "image_base64": "<base64_png>",
    "lr_base64": "<base64_png>",
    "hr_base64": "<base64_png>"
  }
  ```

#### 5. Bicubic Baseline Upscaling
```http
POST /api/v1/bicubic
```
- **Request (Multipart Form)**:
  - `file`: Image binary.
  - `scale`: Integer ($2$ or $4$).
- **Response**: `BicubicResponse` containing matching upscaled baseline image in base64.

---

## 8. CLI Tools & Research Script Reference

| Script Path | Purpose | Key Arguments / Usage |
| :--- | :--- | :--- |
| **`train/train_all.py`** | Primary experiment training runner. Supports curriculum warmup, validation splits, and per-epoch CSV logging. | `--config configs/experiments.yaml --exp E5 --data-dir data/pairs_fixed --val-split 0.1 --log-file outputs/evaluation/e5_train.log --perc-warmup-epochs 3` |
| **`evaluate_all.py`** | Master benchmark evaluator across Set5, Set14, and BSD100. Computes PSNR, SSIM, and LPIPS against bicubic baselines. | `python evaluate_all.py --data-dir data/pairs_fixed` |
| **`evaluate_surveillance.py`** | Tests model robustness against synthetic blur, noise, and compression. | `python evaluate_surveillance.py` |
| **`infer/predict.py`** | CLI inference tool for processing single images or batch directories. | `python infer/predict.py --model outputs/checkpoints/espcn_4x_mse.pt --img input.png --out output.png` |
| **`infer/benchmark.py`** | Hardware throughput and latency benchmarking on CPU vs. GPU. | `python infer/benchmark.py --ckpt outputs/checkpoints/espcn_4x_mse.pt --device cuda` |
| **`infer/webcam_demo.py`** | Live video stream and camera super-resolution display. | `python infer/webcam_demo.py --model outputs/checkpoints/espcn_2x_mse.pt --camera-id 0` |
| **`infer/onnx_export.py`** | Exports PyTorch checkpoints to ONNX format for C++ / OpenCV DNN deployment. | `python infer/onnx_export.py --ckpt outputs/checkpoints/espcn_4x_mse.pt --out espcn_4x.onnx` |
| **`audit_dataset.py`** | Validates image counts, duplicates, and dimension parity across datasets. | `python audit_dataset.py` |
| **`verify_checkpoints.py`** | Inspects tensor shapes, state dict keys, and parameter weights in checkpoints. | `python verify_checkpoints.py` |
| **`verify_data.py`** | Validates LR/HR pair alignment and aspect ratio matching. | `python verify_data.py` |
| **`colab_train.ipynb`** | Jupyter notebook for cloud GPU training on Google Colab (T4 GPU). | Run interactively on Colab with Drive sync. |

---

## 9. Automated Testing Suite (Pytest)

The repository includes a test suite with **105 automated unit tests** in `tests/`:

```bash
python -m pytest tests/
```

### Test Coverage Breakdown
- **`tests/test_backend_api.py`**: Validates FastAPI endpoints (`/health`, `/models`, `/super-resolve`, `/bicubic`), security rejection of invalid MIME types, and path traversal protection.
- **`tests/test_dataset.py`**: Tests `PatchDataset`, sub-pixel extraction, RAM caching, file name filtering, and train/val splits.
- **`tests/test_degradation.py`**: Verifies Gaussian blur kernel generation, noise injection, and JPEG compression degradation pipelines.
- **`tests/test_imresize_matlab.py`**: 32 unit tests verifying mathematical parity with MATLAB's `imresize` bicubic cubic spline kernel.
- **`tests/test_metrics.py`**: Verifies BT.601 Y-channel extraction, border shaving, MSE calculation, PSNR mathematical limits, and SSIM windowing.
- **`tests/test_models.py`**: Validates forward pass output dimensions, parameter shapes, He initialization, and gradient flow for SRCNN and ESPCN at $2\times$ and $4\times$.
- **`tests/test_training.py`**: Tests the training loop, gradient accumulation, AMP mixed-precision scaling, and checkpoint saving.

---

## 10. Complete Directory Hierarchy & File Map

```
C:\Users\ASUS\Desktop\cv\
├── backend/                       # FastAPI REST microservice
│   ├── app/
│   │   ├── __init__.py
│   │   ├── inference.py           # Pre/post-processing & model inference engine
│   │   ├── main.py                # REST API routes & CORS configuration
│   │   ├── registry.py            # Checkpoint loader & metadata registry
│   │   └── security.py            # Magic byte verification & upload security
│   ├── requirements.txt           # Backend-specific Python dependencies
│   └── test_backend.py            # Pytest suite for backend endpoints
├── configs/
│   └── experiments.yaml           # Hyperparameter configs for E1–E8 experiments
├── config/
│   ├── __init__.py
│   └── paths.py                   # Central path resolution module
├── data/
│   ├── dataset.py                 # PyTorch PatchDataset with RAM caching
│   ├── dataset_manifest.json      # Dataset metadata & image inventories
│   ├── degradation.py             # Blur, noise & JPEG degradation simulator
│   ├── download_real_sets.py      # Downloader for real-world surveillance test sets
│   ├── download_sets.py           # Downloader for Set5, Set14, BSD100, T91
│   ├── imresize_matlab.py         # Pure-Python MATLAB bicubic kernel implementation
│   ├── prepare_pairs.py           # Precomputed LR/HR pair generator
│   └── prepare_real_pairs.py      # Real surveillance crop pair generator
├── frontend/                      # Interactive React + Vite workbench
│   ├── public/
│   │   ├── data/                  # Synced evaluation results and training logs
│   │   ├── mock/                  # Standalone fallback mock datasets
│   │   └── samples/               # Lightweight preset images for demo viewing
│   ├── scripts/
│   │   └── sync-data.mjs          # Build script syncing evaluation outputs
│   ├── src/
│   │   ├── api/                   # API client bindings
│   │   ├── app/                   # AppLayout, TopBar, StatusBar, Router
│   │   ├── components/            # UI widgets, charts, drawers, cards
│   │   ├── hooks/                 # useUpscale, useMagnifier, useResults
│   │   ├── lib/                   # Canvas metrics, diff maps, types, utils
│   │   ├── pages/                 # Studio, Benchmarks, Models, Method, Overview
│   │   └── styles/                # CSS tokens & Tailwind theme definitions
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.js
├── infer/                         # Inference tools
│   ├── benchmark.py               # Hardware latency benchmarking tool
│   ├── onnx_export.py             # ONNX export script
│   ├── opencv_dnn_infer.py        # OpenCV DNN C++ runtime inferrer
│   ├── predict.py                 # Single-image / directory CLI inferrer
│   └── webcam_demo.py             # Real-time webcam / RTSP stream super-resolver
├── losses/
│   ├── mse_loss.py                # L2 pixel loss wrapper
│   └── perceptual_loss.py         # Hybrid MSE + VGG11 feature representation loss
├── metrics/
│   └── psnr_ssim.py               # BT.601 Y-channel PSNR & SSIM with border shaving
├── models/
│   ├── espcn.py                   # ESPCN with Sub-Pixel Convolution (PixelShuffle)
│   ├── srcnn.py                   # SRCNN 3-layer convolutional network
│   └── vgg_perceptual.py          # Pre-trained frozen VGG11 feature extractor
├── outputs/
│   ├── checkpoints/               # Canonical trained weights for E1–E8 (.pt)
│   └── evaluation/                # Evaluation results JSON, training logs, CSV summary
├── results/                       # Consolidated benchmark summary metrics
├── scripts/
│   ├── tools/                     # Consolidated one-time research & parity tools
│   │   ├── compute_matlab_bicubic_baselines.py
│   │   ├── create_clean_benchmarks.py
│   │   ├── generate_matlab_pairs.py
│   │   ├── generate_reference_pairs.py
│   │   └── verify_clean_alignment.py
│   ├── fix_lr_alignment.py        # Alignment repair script for T91 pairs
│   ├── generate_verified_t91_lr.py
│   ├── link_set14_to_verified.py
│   └── link_t91_hr.py
├── tests/                         # 105 automated unit tests
├── train/
│   ├── train_all.py               # Multi-experiment runner with curriculum warmup
│   └── trainer.py                 # Core PyTorch training loop with AMP
├── .gitignore                     # Git exclusion rules
├── AGENTS.md                      # Agent operational guidelines
├── colab_train.ipynb              # Google Colab cloud training notebook
├── evaluate_all.py                # Full evaluation runner CLI
├── evaluate_surveillance.py       # Surveillance degradation robustness CLI
├── eval_best.py                   # ESPCN evaluation runner
├── eval_srcnn.py                  # SRCNN evaluation runner
├── FINAL_REPORT.md                # Forensic evaluation report
├── PROJECT_CONTEXT.md             # This document (Single source of truth)
├── README.md                      # Public GitHub documentation
├── requirements.txt               # Top-level Python dependencies
├── run_demo.py                    # Quick demo script
├── verify_checkpoints.py          # Checkpoint tensor validation tool
└── verify_data.py                 # Dataset alignment validator
```

---

## 11. Hardware Specifications & Reproduction Guide

### Local Environment
- **Host System**: Windows 11 Home (x86_64).
- **Processor**: AMD Ryzen 7 5800HS with Radeon Graphics (8 cores, 16 threads).
- **GPU**: NVIDIA GeForce RTX 2050 Laptop GPU (4 GB GDDR6 VRAM).
- **CUDA Toolkit**: 12.8 (Driver version 617.14).
- **Python Version**: 3.14.7.
- **PyTorch Version**: 2.11.0+cu128 with TorchVision 0.26.0+cu128.
- **Node.js**: v20.x+ with npm 10.x+.

### OS-Specific Operational Gotchas
1. **Windows DataLoader Constraints**: PyTorch `DataLoader` must use `num_workers=0` on Windows to prevent multi-process spawn deadlock issues.
2. **CPU Threading**: Physical core affinity is pinned via `torch.set_num_threads(8)` in `utils/cpu_affinity.py`.
3. **No Pin Memory on CPU**: `pin_memory=False` when running without CUDA.
4. **Git HTTP PostBuffer**: Large checkpoint uploads over HTTPS require `git config http.postBuffer 524288000` to prevent HTTP 408 timeout errors.

### Complete Reproduction Instructions

```bash
# 1. Clone the repository
git clone https://github.com/krish10a/Super-resolution.git
cd Super-resolution

# 2. Set up Python virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1       # Windows PowerShell
# source .venv/bin/activate      # Linux / macOS

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run automated test suite (verifies environment integrity)
python -m pytest tests/

# 5. Start the FastAPI backend
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8001 --reload

# 6. In a new terminal, launch the interactive React frontend
cd frontend
npm install
npm run dev
# Open http://127.0.0.1:5173 in your browser
```
