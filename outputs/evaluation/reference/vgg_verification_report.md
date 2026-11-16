# VGG16 / Perceptual Loss Architecture Verification Report

**Step 1 — Unit 1** | ML Consistency Verification
**Repository:** `C:\Users\ASUS\Desktop\cv`
**Date:** 2026-11-01

---

## Executive Summary

The perceptual architecture is confirmed to be **torchvision VGG16 with IMAGENET1K_V1 weights**, extracting features through `relu2_2` via `features[:9]`. All verification checks passed: correct weights, correct ImageNet normalization, frozen parameters, eval mode, correct relu2_2 indexing (index 8 = ReLU, index 9 = MaxPool — correctly NOT mislabeled), gradient flow through the prediction path, target computed under `torch.no_grad()`, and correct device movement of mean/std buffers. No code changes were required.

---

## Verified Facts

### Model identity

| Check | Result |
|-------|--------|
| Model class | `torchvision.models.VGG` (VGG16) |
| Weights enum | `VGG16_Weights.IMAGENET1K_V1` |
| Not VGG11 | Confirmed |

The code uses `models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)`. The earlier AGENTS.md claim of "VGG11" is incorrect; the actual implementation uses VGG16.

### Feature extraction indexing

Torchvision VGG16 feature module indices (verified programmatically):

| Index | Module | Role |
|-------|--------|------|
| 0 | Conv2d(3,64,k3,p1) | conv1_1 |
| 1 | ReLU | relu1_1 |
| 2 | Conv2d(64,64,k3,p1) | conv1_2 |
| 3 | ReLU | relu1_2 |
| 4 | MaxPool2d | pool1 |
| 5 | Conv2d(64,128,k3,p1) | conv2_1 |
| 6 | ReLU | relu2_1 |
| 7 | Conv2d(128,128,k3,p1) | conv2_2 |
| 8 | ReLU | **relu2_2** |
| 9 | MaxPool2d | pool2 (subsequent pooling) |

`features[:9]` selects modules 0 through 8, ending at **relu2_2** (the ReLU at index 8). Index 9 is the subsequent MaxPool2d and is correctly excluded. The code comment in `models/vgg_perceptual.py` accurately documents this.

### Freezing and mode

| Check | Result |
|-------|--------|
| All VGG params `requires_grad=False` | PASS |
| Module in `eval()` mode | PASS |
| ImageNet mean buffer present & correct | PASS (`[0.485, 0.456, 0.406]`) |
| ImageNet std buffer present & correct | PASS (`[0.229, 0.224, 0.225]`) |
| Buffers move correctly to CUDA | PASS |

### Gradient flow

| Check | Result |
|-------|--------|
| Prediction features propagate gradients to SR model input | PASS |
| Target features computed under `torch.no_grad()` | PASS |
| VGG params accumulate no gradients | PASS |
| HybridLoss forward finite (lambda_perc=0.07) | PASS |
| MSE-only path (lambda_perc=0) finite | PASS |

The `HybridLoss.forward` computes `feat_pred` with grad tracking (propagates back to the SR model) and `feat_target` inside `torch.no_grad()` (no gradient tracking on ground truth), exactly as intended for a perceptual loss.

---

## Conclusion

The perceptual implementation is **correct and consistent**:

- VGG16 (not VGG11), IMAGENET1K_V1 weights, frozen, eval mode.
- `features[:9]` correctly ends at relu2_2 (index 8), not the MaxPool at index 9.
- ImageNet normalization buffers correct and device-aware.
- Gradient routing correct: prediction path trains, target path frozen under no_grad.

No implementation changes are needed. Existing perceptual checkpoints (E2, E4, E6, E8) were trained with this code, but their provenance cannot be independently proven from raw state_dicts — see the checkpoint provenance report (Unit 2).

---

## Diagnostic Script

A one-off diagnostic script was run: `outputs/evaluation/reference/diag_vgg_verify.py`. All assertions passed. It is retained as versioned diagnostic output.
