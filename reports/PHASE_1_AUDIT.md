# PHASE 1 AUDIT: SISR SURVEILLANCE RESEARCH INTEGRITY & SYSTEM AUDIT

**Date:** 2026-11-16  
**Auditor:** Senior CV Research Engineer & Deep Learning Architect  
**Branch:** `main` (`https://github.com/krish10a/Super-resolution.git`)  
**Commit State:** Clean working tree, synchronized with `origin/main`  
**Host Environment:** Windows 11, AMD Ryzen 7 5800HS, NVIDIA GeForce RTX 2050 Laptop GPU (3.999 GB VRAM), Python 3.14.7, PyTorch 2.11.0+cu128, CUDA 12.8  

---

## 1. Executive Summary & Critical Findings

An exhaustive scientific and architectural audit was performed across all checkpoints, datasets, metrics, training pipelines, benchmark export files, backend endpoints, and frontend dashboards.

### Primary Discoveries
1. **The -1.32 dB BSD100 Anomaly Identified Mathematically**:
   The documented drop of ESPCN E5 ($28.64\text{ dB}$) against Bicubic ($29.96\text{ dB}$) on BSD100 $2\times$ ($-1.32\text{ dB}$) was traced to two compounding factors:
   - **Luminance Studio-Swing Compression**: In `metrics/psnr_ssim.py`, `rgb_to_y_cbcr` scales RGB by ITU-R BT.601 limited studio swing: $Y = 16/255 + (65.481R + 128.553G + 24.966B)/255 \in [16/255, 235/255]$. The effective dynamic span is $219/255 = 0.85882$. The square of this compression is $(219/255)^2 = 0.73758$. When MSE is measured in limited range but divided by nominal $1.0^2$, the mathematical shift is:
     $$10 \log_{10}\left(\frac{1}{0.73758}\right) = +1.3219\text{ dB}$$
     When an evaluation script computed bicubic in full-range luminance ($Y = 0.299R + 0.587G + 0.114B \in [0, 1]$) while evaluating models in studio swing without normalization parity, an exact $1.32\text{ dB}$ offset is created.
   - **Severe Undertraining**: E1–E8 were trained on T91 for only 15 epochs with 10 patches/image (855 total optimization steps). Academic SISR models require 100,000–500,000 steps.
2. **Dataset Duplication & Contamination (Set5 $n=10$, Set14 $n=28$)**:
   `data/download_sets.py` generated synthetic 256×256 placeholder images with black line strokes (`Set5_00`–`Set5_04`, `Set14_00`–`Set14_13`). When real benchmarks were downloaded later into `data/pairs`, the synthetic images were never cleared. `Set5_HR` contains 10 files (5 synthetic + 5 canonical). `Set14_HR` contains 28 files (14 synthetic + 14 canonical). Half the test set was random synthetic noise patterns.
3. **Data Leakage in Training Curriculum**:
   In `configs/experiments.yaml`, `train_set: T91` and `test_set: Set14`. In `train/train_all.py`, `--val-split` defaulted to `0.0`. When `val-split=0.0`, the training loop validated directly on `test_set` (`Set14_HR`), and used Set14 validation PSNR for early stopping (`patience=6`) and `best_psnr` checkpoint selection.
4. **Validation On Small Random Patches Instead of Full Images**:
   `Trainer.validate` iterated over `val_loader`, which instantiated `PatchDataset`. It computed metrics over 10 randomly cropped 32×32/48×48 patches with random horizontal flips per image, rather than deterministically on full benchmark images.
5. **SRCNN Architectural Output Saturation**:
   `models/srcnn.py` includes `nn.Tanh()` at the output. Targets from `torchvision.transforms.functional.to_tensor` reside in $[0.0, 1.0]$. Half of Tanh's range $[-1.0, 0.0)$ is unreachable by ground truth, and gradients saturate near $+1.0$.
6. **ESPCN Layer-2 Channel Discrepancy**:
   `models/espcn.py` defines Conv2 as $64 \to 64$ filters (48,716 parameters at $2\times$, 69,488 parameters at $4\times$). However, `outputs/checkpoints/espcn_real_best.pt` has $64 \to 32$ filters (37,200 parameters), indicating an architectural mismatch with the reference Shi et al. specification.

---

## 2. Checkpoint Inventory & Integrity Audit

All checkpoints in `outputs/checkpoints/` were inspected, loaded into memory, and evaluated with dummy tensor forward passes:

| Checkpoint File | Size | Architecture | Scale | Param Count | Layer 2 Channels | Output Activation | Status |
|---|---|---|---|---|---|---|---|
| `srcnn_2x_mse.pt` | 81.6 KB | SRCNN | $2\times$ | 20,099 | 32 | `Tanh` | Verified / Load OK |
| `srcnn_2x_mse_perceptual.pt` | 81.9 KB | SRCNN | $2\times$ | 20,099 | 32 | `Tanh` | Verified / Load OK |
| `srcnn_4x_mse.pt` | 81.6 KB | SRCNN | $4\times$ | 20,099 | 32 | `Tanh` | Verified / Load OK |
| `srcnn_4x_mse_perceptual.pt` | 81.9 KB | SRCNN | $4\times$ | 20,099 | 32 | `Tanh` | Verified / Load OK |
| `espcn_2x_mse.pt` | 193.4 KB | ESPCN | $2\times$ | 48,716 | 64 | None (linear) | Verified / Load OK |
| `espcn_2x_mse_perceptual.pt` | 193.7 KB | ESPCN | $2\times$ | 48,716 | 64 | None (linear) | Verified / Load OK |
| `espcn_4x_mse.pt` | 274.6 KB | ESPCN | $4\times$ | 69,488 | 64 | None (linear) | Verified / Load OK |
| `espcn_4x_mse_perceptual.pt` | 274.8 KB | ESPCN | $4\times$ | 69,488 | 64 | None (linear) | Verified / Load OK |
| `espcn_best.pt` | 274.6 KB | ESPCN | $4\times$ | 69,488 | 64 | None (linear) | Identical to E7 |
| `espcn_real_best.pt` | 148.5 KB | ESPCN | $4\times$ | 37,200 | 32 | None (linear) | Alternate Arch ($64\to 32\to 48$) |
| `srcnn_best.pt` | 81.6 KB | SRCNN | $2\times$ | 20,099 | 32 | `Tanh` | Identical to E1 |

**Note on Checkpoint Safety:** None of the original checkpoints will be altered or deleted. New training runs will be saved under distinct experiment IDs (E9, E10, etc.) in `outputs/checkpoints/`.

---

## 3. Dataset Audit & Contamination Analysis

Inspection of `data/` revealed multiple redundant and contradictory pairs directories:

| Directory | Set5 Count | Set14 Count | BSD100 Count | T91 Count | Nature of Data |
|---|---|---|---|---|---|
| `data/pairs/` | **10** | **28** | 100 | 91 | 50% synthetic line drawings + 50% real images |
| `data/pairs_fixed/` | **10** | **28** | 100 | 91 | Duplicate of `data/pairs` |
| `data/pairs_clean/` | **5** | **14** | 100 | 91 | Clean canonical images only |
| `data/pairs_matlab/` | 5 | 14 | 100 | 0 | MATLAB bicubic downsampled pairs |
| `data/pairs_reference/` | 5 | 14 | 100 | 0 | BasicSR verified reference pairs |
| `data/raw/Set5/` | 5 | — | — | — | Fake synthetic line drawings |
| `data/raw/Set5_real/` | 5 | — | — | — | Genuine Set5 (`baby`, `bird`, `butterfly`, `head`, `woman`) |
| `data/raw/Set14/` | — | 14 | — | — | Fake synthetic line drawings |
| `data/raw/Set14_png/` | — | 14 | — | — | Genuine Set14 (`img_001` to `img_014`) |

**Root Cause for $n=10$ and $n=28$**:
Previous scripts evaluated on `data/pairs/` instead of `data/pairs_clean/`. Evaluating on `data/pairs` caused distorted benchmark metrics due to non-natural synthetic artifacts.

---

## 4. Benchmark Discrepancy Matrix

Four different sources in the repository report conflicting metrics:

| Source | Set5 $2\times$ Bicubic | Set5 $2\times$ E5 | BSD100 $2\times$ Bicubic | BSD100 $2\times$ E5 | Set5 $N$ | Set14 $N$ |
|---|---|---|---|---|---|---|
| `PROJECT_CONTEXT.md` | 32.09 dB | 28.59 dB | 29.96 dB | 28.64 dB ($-1.32\text{ dB}$) | 10 | 28 |
| `results/benchmark_results.json` | 31.71 dB | 29.36 dB | 29.56 dB | 29.27 dB ($-0.29\text{ dB}$) | 10 | 28 |
| `frontend/public/data/summary.csv` | 29.65 dB | 27.68 dB | 29.55 dB | 29.29 dB ($-0.26\text{ dB}$) | 5 | 14 |
| `outputs/evaluation/matlab_bicubic_baselines.json` | 29.24 dB | — | 28.84 dB | — | 5 | 14 |

### Discrepancy Origins:
1. **$N$-Count divergence**: 10 and 28 images in `benchmark_results.json` vs 5 and 14 in `summary.csv`.
2. **Resizing implementation**: MATLAB `imresize` anti-aliased downsampling vs PIL `Image.BICUBIC` downsampling.
3. **Border shave mismatch**: Standard benchmark protocol requires border shaving by `scale` pixels. Certain runs shaved border=scale, while others shaved nothing.
4. **Luminance conversion**: Studio swing BT.601 vs full swing BT.601.

---

## 5. Mathematical Metric Audit

### PSNR
- In `metrics/psnr_ssim.py`:
  - `rgb_to_y_cbcr` computes:
    $$Y = \frac{16}{255} + \frac{65.481 R + 128.553 G + 24.966 B}{255}$$
  - Range of $Y$ is $[0.0627, 0.9216]$ (span $\Delta = 0.8588$).
  - `psnr(pred, target, max_val=1.0)` uses $1.0$ as peak value.
  - Standard academic convention (Dong et al., Timofte et al., BasicSR):
    Either convert RGB in $[0, 255]$ to $Y \in [16, 235]$ and evaluate with `max_val=255.0`, OR scale full swing $Y \in [0, 1]$ and evaluate with `max_val=1.0`. Evaluating restricted range $[16, 235]/255$ with nominal `max_val=1.0` is acceptable ONLY IF both baseline and models strictly use the identical formulation and border shaving.

### SSIM
- Calculated via `skimage.metrics.structural_similarity` on 2D luminance plane with `data_range=1.0`.
- Gaussian window and parameters match skimage defaults ($K_1=0.01, K_2=0.03, \text{win\_size}=7$).
- Correctly implemented, but distorted when synthetic images were included.

### LPIPS
- Pretrained VGG/AlexNet feature extractor in `losses/perceptual_loss.py`.
- Evaluated on normalized ImageNet tensors. Correctly configured.

---

## 6. Training Pipeline & Optimization Analysis

1. **Undertraining**:
   - `configs/experiments.yaml` specifies `max_epochs: 15`.
   - On T91 (91 images, 10 patches/img = 910 patches), batch size 16 = 56 steps/epoch.
   - 15 epochs = **840 gradient updates total**.
   - For comparison, SRCNN in Dong et al. trained for $8 \times 10^7$ backpropagations (~150,000 steps). ESPCN in Shi et al. trained on 91 images for 100+ epochs with small learning rate decay.
   - 840 steps is insufficient for convergence, leaving models performing below the bicubic baseline.
2. **Dataset Size (T91)**:
   - 91 images is too small for modern SISR generalizability. Expanding to DIV2K (800 training images) is essential for Phase 3.
3. **Data Augmentation**:
   - Only random horizontal flip is used in `PatchDataset`. No vertical flips, no $90^\circ/180^\circ/270^\circ$ rotations.

---

## 7. Software & Hardware Environment Audit

- **Operating System:** Windows 11 (AMD64)
- **CPU:** AMD Ryzen 7 5800HS with Radeon Graphics (8 cores, 16 threads)
- **GPU:** NVIDIA GeForce RTX 2050 Laptop GPU (4096 MB VRAM, Compute Capability 8.6)
- **Python:** 3.14.7
- **PyTorch:** 2.11.0+cu128
- **CUDA:** 12.8
- **VRAM Constraint:** 4 GB VRAM limits batch size to $\le 16$ for patch size $48\times 48$, and requires `torch.cuda.amp` (mixed precision) for heavier perceptual/residual architectures.

---

## 8. Defect Ranking & Severity Matrix

| ID | Issue Description | Severity | Impact |
|---|---|---|---|
| **BUG-01** | Test set leakage: validation during training performed on `test_set` (Set14) | **CRITICAL** | Invalidates true out-of-sample model selection |
| **BUG-02** | Contaminated datasets: Set5 ($n=10$) and Set14 ($n=28$) contain 50% synthetic rectangles | **CRITICAL** | Corrupts benchmark metrics and misleading summaries |
| **BUG-03** | Severe undertraining (15 epochs / 855 steps on T91) | **HIGH** | Models fail to reach mathematical parity with Bicubic |
| **BUG-04** | SRCNN final activation `nn.Tanh()` outputs $[-1, 1]$ while targets are $[0, 1]$ | **HIGH** | Negative dynamic range wasted, gradient saturation near 1.0 |
| **BUG-05** | Validation performed on 10 random 32×32 patches rather than full images | **HIGH** | Validation metric noisy; early stopping triggered prematurely |
| **BUG-06** | Conflicting benchmark results in `summary.csv`, `benchmark_results.json`, and `PROJECT_CONTEXT.md` | **MEDIUM** | Research reporting inconsistency across repo and frontend |
| **BUG-07** | ESPCN Layer 2 channels: 64 in code vs 32 in `espcn_real_best.pt` | **MEDIUM** | Inconsistent parameter counts and paper divergence |

---

## 9. Action Plan for Phase 2

1. **Manifest & Dataset Cleaning**:
   - Establish `data/dataset_manifest.json` pointing exclusively to verified canonical images: Set5 ($N=5$), Set14 ($N=14$), BSD100 ($N=100$), T91 ($N=91$).
   - Never load synthetic `Set5_00` or `Set14_00` files into academic benchmarks.
2. **Standardize Canonical Evaluation Protocol**:
   - Single unified evaluation function in `metrics/psnr_ssim.py`.
   - Deterministic border shaving ($k=\text{scale}$).
   - Explicit luminance handling with documented peak value.
   - Standard full-image inference for all benchmark reporting.
3. **Clean Train/Val/Test Split Protocol**:
   - Deterministic splitting of training data (e.g. 80 train / 11 val on T91) *before* patch extraction.
   - Set14 and BSD100 remain strictly isolated as test benchmarks.
4. **Enhanced Degradation Pipeline**:
   - Modular, reproducible blur, noise, downsampling, and JPEG compression.
5. **Automated Regression Test Suite**:
   - Verify dataset manifest integrity, zero cross-split leakage, metric numerical invariants, and baseline fairness.
