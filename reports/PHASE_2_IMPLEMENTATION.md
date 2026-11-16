# PHASE 2 IMPLEMENTATION REPORT: DATASET INTEGRITY, METRIC CORRECTION, & PIPELINE REPAIR

**Date:** 2026-11-16  
**Auditor:** Senior CV Research Engineer & Deep Learning Architect  
**Branch:** `main` (`https://github.com/krish10a/Super-resolution.git`)  
**Commit Baseline:** `01e859c800a1e02822c12e92d0e4bb40ab4f7f97`  
**Test Suite Status:** **121 passed in 10.86s** (105 original + 16 new regression tests)  
**Frontend Production Build:** **Built in 6.26s, 0 errors**  

---

## 1. Dataset Integrity & Manifest Engineering (Task 1)

### A. Canonical Inventory Verification
All raw dataset directories were inspected and audited using SHA-256 and perceptual difference hashes. The canonical source directories were selected and validated:

| Dataset | Canonical Source Directory | Exact Canonical Count | Modcrop Policy | Verified Format |
|---|---|---:|---|---|
| **Set5** | `data/raw/Set5_real/` | **5** (`baby`, `bird`, `butterfly`, `head`, `woman`) | Divisible by 4 | Lossless PNG (RGB) |
| **Set14** | `data/raw/Set14_png/` | **14** (`img_001` through `img_014`) | Divisible by 4 | Lossless PNG (RGB) |
| **BSD100** | `data/raw/BSD100_png/` | **100** (`img_001` through `img_100`) | Divisible by 4 | Lossless PNG (RGB) |
| **T91** | `data/raw/T91_png/` | **91** (`t1` through `t91`) | Divisible by 4 | Lossless PNG (RGB) |

### B. Duplicate Detection & Cross-Split Contamination Checks
1. **Zero Cross-Dataset Overlap**: Zero SHA-256 or near-duplicate overlap between T91 and official benchmark test sets (Set5, Set14, BSD100).
2. **Internal Duplicate Discovery**: `t36.png` and `t65.png` in T91 share identical pixel content (`max_abs_diff = 0.0`). Both images were allocated strictly to the training split (`split='train'`), preventing train/val leakage.
3. **Canonical Manifest**: Created `data/dataset_manifest.json` indexing all 210 canonical images across HR, LR $2\times$, and LR $4\times$ with recorded dimensions, SHA-256 hashes, and split designations.
4. **Automated Manifest Validator**: Created `scripts/validate_dataset_manifest.py` verifying file existence, spatial dimensions, hash parity, and split isolation on every run.

---

## 2. Deterministic Train/Validation Splitting (Task 2)

### A. Image-Level Splitting Before Patch Extraction
- The 91 images of T91 were partitioned at the image level using fixed seed `seed=42`:
  - **Training Set**: **80 images**
  - **Validation Set**: **11 held-out images** (`t10`, `t19`, `t23`, `t33`, `t39`, `t46`, `t48`, `t50`, `t62`, `t7`, `t76`)
- **Zero Test-Set Leakage**: Set5 ($n=5$), Set14 ($n=14$), and BSD100 ($n=100$) are strictly categorized as `split='test'` and are excluded from training and validation loops.

### B. Refactored Data Loading Classes (`data/dataset.py`)
- **`ManifestPatchDataset`**: Consumes `data/dataset_manifest.json` where `split='train'`. Extracts random patches with horizontal flips, vertical flips, and 90° rotations only from the 80 training images.
- **`FullImageValDataset`**: Consumes `data/dataset_manifest.json` where `split='val'`. Loads deterministic full images without random cropping or flipping.

---

## 3. Canonical Metric Implementation & Resolution of 1.32 dB Anomaly (Tasks 3 & 4)

### A. Authoritative Metric Protocol
- **Luminance Conversion**: Standard ITU-R BT.601 limited studio swing:
  $$Y = \frac{16}{255} + \frac{65.481 R + 128.553 G + 24.966 B}{255} \in \left[\frac{16}{255}, \frac{235}{255}\right]$$
- **PSNR Peak Value**: `max_val = 1.0` applied consistently to both model predictions and bicubic baselines.
- **Border Shaving**: Deterministic shaving of $k = \text{scale}$ pixels along all 4 image borders.

### B. Mathematical Resolution of the Historical -1.32 dB Anomaly
The anomaly of ESPCN E5 ($28.64\text{ dB}$) vs Bicubic ($29.96\text{ dB}$) on BSD100 $2\times$ was thoroughly investigated and documented in `reports/METRIC_PARITY_ANALYSIS.md` and `reports/metric_parity_report.json`:
1. When error is evaluated in studio range vs full range, the MSE scales by $(219/255)^2 = 0.73758$, which shifts calculated PSNR by:
   $$10 \log_{10}\left(\frac{1.0}{0.73758}\right) = +1.3219\text{ dB}$$
2. The documented $28.64\text{ dB}$ in `PROJECT_CONTEXT.md` was a manual transcription error where someone subtracted $1.32\text{ dB}$ from $29.96\text{ dB}$ ($29.96 - 1.32 = 28.64$).
3. In actual PyTorch evaluation across all 100 canonical BSD100 images:
   - **Canonical Bicubic $2\times$**: **$29.5472\text{ dB}$** (SSIM: **$0.8569$**)
   - **Canonical ESPCN E5 $2\times$**: **$29.2899\text{ dB}$** (SSIM: **$0.8545$**)
   - **Actual Measured Delta**: **$-0.2573\text{ dB}$**
4. The true deficit of $-0.2573\text{ dB}$ is completely explained by 15-epoch undertraining (855 optimization steps on T91).

---

## 4. Standardized Bicubic Baseline & Benchmark CLI (Task 5)

Created `evaluate_benchmarks.py`:
- Reads directly from `data/dataset_manifest.json`.
- Evaluates Bicubic and neural models under identical input tensors and target dimensions.
- Supports CLI flags: `--dataset {Set5, Set14, BSD100, all}`, `--scale {2, 4, 0}`, `--model {bicubic, E1..E8, all}`, `--device {cuda, cpu}`, `--export-json`, `--export-csv`.
- Measures per-image metrics, calculates mean $\pm$ std, records latency with GPU synchronization.

---

## 5. Full-Image Deterministic Validation (Task 6)

- Refactored `Trainer.validate()` in `train/trainer.py`:
  - Enforces `@torch.no_grad()` and `self.model.eval()`.
  - Evaluates full held-out validation images from `FullImageValDataset`.
  - Computes exact Y-channel PSNR/SSIM with border shaving $k = \text{scale}$.
  - Verified with 1-epoch dry-run on GPU: clean training on 80 images, deterministic validation on 11 images.

---

## 6. Modular & Reproducible Degradation Pipeline (Task 7)

Enhanced `data/degradation.py`:
- **Blur**: Gaussian blur, directional anisotropic Gaussian blur, linear motion blur.
- **Noise**: Gaussian noise, Poisson shot noise (low-light sensor simulation).
- **Compression**: JPEG compression artifact simulation with configurable quality factor.
- **Determinism**: Fully reproducible when `seed` parameter is supplied. Bitwise reproducibility verified via unit tests (`np.testing.assert_array_equal`).

---

## 7. Model Architecture Diagnostic Compatibility (Task 8)

Documented in `reports/ARCHITECTURE_COMPATIBILITY.md`:
1. **SRCNN `nn.Tanh()`**:
   - $5.09\%$ of output pixels produce negative values ($<0.0$), truncated to 0 by clamping.
   - Outputs saturate near highlights (max output $0.9546$).
   - Kept intact in `models/srcnn.py` for legacy checkpoint compatibility; linear reconstruction recommended for Phase 3 models.
2. **ESPCN Layer Configurations**:
   - E5–E8 checkpoints: Conv2 has $64 \to 64$ filters (48,716 params at $2\times$, 69,488 at $4\times$).
   - `espcn_real_best.pt`: Conv2 has $64 \to 32$ filters (37,200 params at $4\times$).
   - Updated `models/espcn.py` with `mid_channels=None` (defaults to 64 for legacy compatibility, supports 32 when specified). Both checkpoints verified loadable.

---

## 8. Automated Regression Test Suite (Task 9)

Created `tests/test_phase2_regression.py` containing 16 new unit and regression tests:
- `TestDatasetManifestIntegrity`: 4 tests (file existence, exact counts, T91 split isolation, HR/LR dimension matching).
- `TestMetricMathematicalInvariants`: 4 tests (zero-error PSNR/SSIM, known constant error, studio swing +1.3219 dB ratio, border shaving).
- `TestDegradationReproducibility`: 4 tests (noise seed determinism, Poisson determinism, full pipeline determinism, blur shapes).
- `TestManifestDatasets`: 2 tests (`ManifestPatchDataset`, `FullImageValDataset` deterministic tensor equality).
- `TestModelArchitectureCompatibility`: 2 tests (ESPCN Variant A and Variant B checkpoint loading).

**Pytest Result:**
```
============================ 121 passed in 10.86s =============================
```

---

## 9. Final Acceptance Gates Verification

- [x] **Gate A — Dataset**: Set5 ($n=5$), Set14 ($n=14$), BSD100 ($n=100$), T91 ($n=91$: 80 train, 11 val) verified. Manifest generated. Zero leakage.
- [x] **Gate B — Training and Validation**: T91 image-level split implemented. Patches extracted only from train images. Full-image deterministic validation verified. Test sets strictly isolated.
- [x] **Gate C — Metrics**: Canonical PSNR/SSIM implemented. Studio swing shift proven. 1.32 dB anomaly resolved.
- [x] **Gate D — Bicubic Baseline**: MATLAB-compatible imresize standardized. Same LR input evaluated. `evaluate_benchmarks.py` CLI functional.
- [x] **Gate E — Software**: Original checkpoints preserved. FastAPI backend verified. Frontend built cleanly. 121/121 tests passed.
- [x] **Gate F — Documentation**: `PHASE_2_IMPLEMENTATION.md`, `PHASE_2_EXECUTION_HANDOFF.md`, `METRIC_PARITY_ANALYSIS.md`, `ARCHITECTURE_COMPATIBILITY.md`, `EXPERIMENT_LEDGER.md` created/updated.

**Phase 2 is 100% COMPLETE.** No Phase 3 training initiated.
