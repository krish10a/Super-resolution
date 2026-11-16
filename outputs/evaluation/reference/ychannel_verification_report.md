# Y-Channel (BT.601) Verification Report

**Step 1 — Unit 5** | ML Consistency Verification
**Repository:** `C:\Users\ASUS\Desktop\cv`
**Date:** 2026-11-01

---

## Executive Summary

The project's BT.601 Y-channel conversion in `metrics/psnr_ssim.rgb_to_y_cbcr` is **correct**:

`Y = 16/255 + (65.481·R + 128.553·G + 24.966·B) / 255`

All primary colors (black, white, red, green, blue) match the expected values exactly. Channel 0 is correctly treated as R (weight 65.481), channel 2 as B (weight 24.966) — consistent with BT.601 RGB convention. `compute_psnr_ssim` applies Y conversion before PSNR/SSIM when `y_channel=True`. End-to-end identical-image checks yield PSNR=100, SSIM=1.0 for all colors. No code change required.

---

## Verified Values

BT.601 Y for each primary color (channel values in [0,1]):

| Color | R | G | B | Expected Y | Project Y | Match |
|-------|---|---|---|------------|-----------|-------|
| Black | 0 | 0 | 0 | 0.062745 | 0.062745 | ✅ |
| White | 1 | 1 | 1 | 0.921569 | 0.921569 | ✅ |
| Red | 1 | 0 | 0 | 0.319533 | 0.319533 | ✅ |
| Green | 0 | 1 | 0 | 0.566875 | 0.566875 | ✅ |
| Blue | 0 | 0 | 1 | 0.160651 | 0.160651 | ✅ |

Formula check: `Y = 16/255 + (65.481·R + 128.553·G + 24.966·B)/255`
- Red: 16/255 + 65.481/255 = 81.481/255 = 0.319533 ✅
- Green: 16/255 + 128.553/255 = 144.553/255 = 0.566875 ✅
- Blue: 16/255 + 24.966/255 = 40.966/255 = 0.160651 ✅
- White: 16/255 + 219/255 = 235/255 = 0.921569 ✅

---

## Channel-Order Sensitivity

`rgb_to_y_cbcr` treats tensor channel 0 as R (weight 65.481) and channel 2 as B (weight 24.966):

- Pure RED passed as RGB (channel 0 = R): Y = 0.319533 ✅
- Pure RED passed as BGR (channel 0 = B): Y = 0.160651 (the blue weight) ✅ — differs as expected

No BGR swap bug: the function does not swap R/B weights. Callers must pass RGB-order tensors. All project callers (`evaluate_all.py`, `compute_psnr_ssim`) use PIL `.convert('RGB')`, so input is consistently RGB order.

---

## compute_psnr_ssim End-to-End

For identical images (pred == target) at all 5 colors, `scale=0, y_channel=True`:

| Color | PSNR | SSIM |
|-------|------|------|
| Black | 100.0 | 1.000000 |
| White | 100.0 | 1.000000 |
| Red | 100.0 | 1.000000 |
| Green | 100.0 | 1.000000 |
| Blue | 100.0 | 1.000000 |

PSNR capped at 100 (MSE < 1e-12), SSIM = 1.0. Correct.

---

## Vendored Reference Caveat

The vendored `outputs/evaluation/basicsr_reference.py` `rgb2ycbcr` / `bgr2ycbcr` functions have a **known cast quirk** in their uint8 and float paths (they do not reliably produce [0,1] Y for all inputs). This is vendored reference code used only for parity comparison, not project code. The project's own `rgb_to_y_cbcr` is correct and independent. This caveat does not affect project results.

---

## Conclusion

BT.601 Y-channel conversion is **verified correct** in the project. It is used consistently in `compute_psnr_ssim` with border shaving and clamp [0,1]. No implementation change required.

Diagnostic script: `outputs/evaluation/reference/diag_ychannel.py`.
