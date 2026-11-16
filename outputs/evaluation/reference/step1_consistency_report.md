# Step 1 — ML Consistency Verification: Consolidated Technical Report

**SISR Surveillance Super-Resolution Project**  
**Repository:** `C:\Users\ASUS\Desktop\cv`  
**Date:** 2026-11-01  
**Scope:** Step 1 only (no retraining, no checkpoint/result overwrites)

---

## Executive Summary

All Step 1 verification units completed. The project's ML implementation is **internally consistent and matches BasicSR within acceptable tolerance**:

| Unit | Finding | Status |
|------|---------|--------|
| 1. VGG16 / Perceptual architecture | VGG16 + IMAGENET1K_V1, frozen, eval, features[:9]=relu2_2 (index 8), correct gradient flow | ✅ Verified |
| 2. Perceptual checkpoint provenance | Config+code establish perceptual intent for E2/E4/E6/E8; checkpoints are raw state_dicts (no training metadata) — independent proof not possible | ⚠️ Documented limitation |
| 3. PSNR parity | Project matches BasicSR within 0.014 dB (float32 vs float64 + uint8 Y quantization) | ✅ Parity achieved |
| 4. SSIM parity | Project matches BasicSR within 0.0323 max (skimage vs cv2 border-crop + window + precision) | ✅ Within tolerance |
| 5. Y-channel BT.601 | Correct formula, correct channel order, end-to-end PSNR=100/SSIM=1.0 | ✅ Verified |

**No code changes were required.** No checkpoints, training logs, or evaluation results were modified or overwritten. No retraining performed.

---

## Unit 1 — VGG16 / Perceptual Loss Architecture

**Report:** `vgg_verification_report.md`  
**Diagnostic:** `diag_vgg_verify.py` (all checks passed)

- Model: `torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)` — **VGG16, not VGG11** (AGENTS.md's "VGG11" claim was incorrect).
- Weights: IMAGENET1K_V1, ImageNet classes confirmed.
- Normalization: ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`, registered as buffers, move correctly to CUDA.
- Freezing: all VGG parameters `requires_grad=False`.
- Mode: `eval()`.
- Feature extraction: `features[:9]` = modules 0–8 ending at **relu2_2** (ReLU at index 8). Index 9 = MaxPool2d (subsequent pooling), correctly **excluded**.
- Gradient flow: prediction features propagate gradients to SR model; target features under `torch.no_grad()`; VGG params accumulate no gradients; HybridLoss forward finite.
- MSE-only path (lambda_perc=0) works correctly.

**AGENTS.md correction:** AGENTS.md incorrectly states "VGG11". The code uses VGG16. No code change needed; documentation note only.

---

## Unit 2 — Perceptual Checkpoint Provenance

**Report:** `checkpoint_provenance_report.md`

Experiment matrix (from `configs/experiments.yaml`):

| Exp | Architecture | Scale | Loss | Checkpoint |
|-----|--------------|-------|------|------------|
| E1 | SRCNN | 2× | MSE | `srcnn_2x_mse.pt` |
| E2 | SRCNN | 2× | MSE+perceptual | `srcnn_2x_mse_perceptual.pt` |
| E3 | SRCNN | 4× | MSE | `srcnn_4x_mse.pt` |
| E4 | SRCNN | 4× | MSE+perceptual | `srcnn_4x_mse_perceptual.pt` |
| E5 | ESPCN | 2× | MSE | `espcn_2x_mse.pt` |
| E6 | ESPCN | 2× | MSE+perceptual | `espcn_2x_mse_perceptual_verified.pt` |
| E7 | ESPCN | 4× | MSE | `espcn_4x_mse_verified.pt` |
| E8 | ESPCN | 4× | MSE+perceptual | `espcn_4x_mse_perceptual_verified.pt` |

- Code path: `train/train_all.py` maps `mse_perceptual` → `HybridLoss(lambda_perc=0.05)` (perceptual active from epoch 1, no warmup).
- Train logs (`e2/e4/e6/e8_train.log`) confirm 15-epoch runs with hybrid-style loss magnitudes.
- **Limitation:** checkpoints are raw `state_dict`s (weights only) — no epoch, loss type, lambda, or optimizer metadata stored. Independent provenance proof not possible; attribution is config-documented intent only.

**Recommendation:** future training should embed metadata (epoch, loss, lambda, dataset hash) in checkpoints.

---

## Unit 3 — PSNR Parity

**Report:** `psnr_parity_report.md`  
**Diagnostic:** `diag_psnr_parity.py`

Compared project `psnr()` vs BasicSR `calculate_psnr` / `calculate_psnr_pt` on identical arrays:

- [0,1] float RGB: diff 0.00003 dB (float32 vs float64 accumulation).
- [0,255] uint8: diff 0.000001 dB (exact).
- Y-channel BT.601, shave=scale: diff 0.00009 dB (pt) / 0.01419 dB (np uint8 quantization).
- Identical images: project caps at 100 dB; BasicSR returns `inf` (design convention difference, not a bug).
- Max observed diff: **0.01419 dB** — well within the reported 0.0964 dB tolerance.

**Conclusion:** PSNR parity achieved. No change needed.

---

## Unit 4 — SSIM Parity

**Report:** `ssim_parity_report.md`  
**Diagnostic:** `diag_ssim_parity.py`

Compared project skimage `structural_similarity` (data_range=1.0) vs BasicSR cv2 SSIM (11×11 Gaussian σ=1.5, filter2D, [5:-5] crop, data_range=255) on identical float BT.601 Y arrays:

- Max synthetic diff: **0.0141** (smooth gradients).
- Max real-benchmark diff (prior report): **0.0323**.
- Identical images: 1.0 both.
- Root cause: BasicSR crops 5px border ([5:-5]); skimage includes reflect-padded border. Combined with window truncation and float32 vs float64 precision. Inherent skimage-vs-cv2 implementation difference.

**Conclusion:** SSIM parity within acceptable tolerance (<0.04). No change needed.

**Vendored caveat:** vendored `basicsr_reference.rgb2ycbcr`/`_basicSR_to_y_channel` cast Y to uint8 (latin bug in vendored reference only). Diagnostic used manual float BT.601 for both paths. Does not affect project results.

---

## Unit 5 — Y-Channel Verification

**Report:** `ychannel_verification_report.md`  
**Diagnostic:** `diag_ychannel.py`

- BT.601 formula verified for black, white, red, green, blue — all exact matches.
- Channel 0 = R (weight 65.481), channel 2 = B (weight 24.966) — correct RGB convention.
- `compute_psnr_ssim` end-to-end: PSNR=100, SSIM=1.0 for all colors.
- Callers use RGB order (PIL `.convert('RGB')`). No BGR swap bug.

---

## Files Created (Step 1, versioned diagnostics + reports)

All under `outputs/evaluation/reference/`:

| File | Purpose |
|------|---------|
| `vgg_verification_report.md` | Unit 1 report |
| `checkpoint_provenance_report.md` | Unit 2 report |
| `psnr_parity_report.md` | Unit 3 report |
| `ssim_parity_report.md` | Unit 4 report |
| `ychannel_verification_report.md` | Unit 5 report |
| `step1_consistency_report.md` | This consolidated report |
| `diag_vgg_verify.py` | Unit 1 diagnostic script |
| `diag_psnr_parity.py` | Unit 3 diagnostic script |
| `diag_ssim_parity.py` | Unit 4 diagnostic script |
| `diag_ychannel.py` | Unit 5 diagnostic script |

Existing files preserved (no modification): `evaluation_results.json`, `summary.csv`, training logs, all checkpoints, `tests/`, source models/metrics/losses.

---

## Step 1 Acceptance Criteria

- [x] Actual perceptual model identified: **VGG16** (not VGG11)
- [x] VGG feature extraction independently verified: features[:9]=relu2_2, frozen, eval, IMAGENET1K_V1
- [x] Existing perceptual checkpoint provenance assessed: config-documented intent, state_dict limitation documented
- [x] PSNR parity discrepancy investigated: max 0.014 dB, within tolerance
- [x] SSIM parity discrepancy investigated: max 0.0323, root cause documented (border crop + window + precision)
- [x] Colour conversion verified: BT.601 correct for all primary colors
- [x] Metric implementation correct if necessary: no change required
- [x] Existing checkpoints preserved: none modified or overwritten
- [x] No retraining performed
- [x] Final consistency report produced: this document

---

## Next Step

Proceed to **Step 3** — Complete React + FastAPI Application ("SISR Surveillance Super-Resolution Studio"). See plan file `C:\Users\ASUS\.claude\plans\greedy-bouncing-pudding.md`, Work units 6–13.
