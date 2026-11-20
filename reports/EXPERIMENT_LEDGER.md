# SISR SURVEILLANCE EXPERIMENT LEDGER

This ledger tracks the provenance, training configuration, execution status, and verified benchmark metrics of every model in the project.

**Status Key:**
- `NOT STARTED`: Configuration created, no execution.
- `IN PROGRESS`: Currently running or training.
- `AWAITING USER EXECUTION`: Pipeline, configs, models implemented and tested; awaiting manual user terminal execution.
- `LOCALLY VERIFIED`: Code/shape/smoke-test verified on local machine.
- `EXPERIMENT COMPLETED`: Trained, weights saved, verified on canonical benchmark.
- `FAILED`: Training diverged or broken.
- `LEGACY / PRESERVED`: Original repository checkpoint (preserved without modification).

---

## Canonical Baseline Reference (Verified BasicSR / MATLAB-Compatible)

*Evaluated on canonical datasets with border shaving $k=\text{scale}$, ITU-R BT.601 luminance.*

| ID | Method | Scale | Interpolation / Pipeline | Set5 ($n=5$) PSNR / SSIM | Set14 ($n=14$) PSNR / SSIM | BSD100 ($n=100$) PSNR / SSIM | Latency (ms) | Status |
|---|---|---|---|---|---|---|---|---|
| **REF-B2** | Bicubic Baseline | $2\times$ | MATLAB-compatible / PIL Bicubic | 33.66 / 0.9299 | 30.24 / 0.8688 | 29.55 / 0.8431 | 1.8 ms | LOCALLY VERIFIED |
| **REF-B4** | Bicubic Baseline | $4\times$ | MATLAB-compatible / PIL Bicubic | 28.42 / 0.8104 | 26.00 / 0.7027 | 25.96 / 0.6675 | 1.9 ms | LOCALLY VERIFIED |

---

## Original Experiments (E1–E8) — Preserved Legacy

*Trained on T91 ($n=91$) for 15 epochs. Historical training suffered from Set14 validation leakage and random patch validation. Checkpoints preserved intact in `outputs/checkpoints/`.*

| ID | Architecture | Scale | Loss Function | Checkpoint File | Parameters | Train Set | Validation Set | Canonical BSD100 PSNR / SSIM | Canonical Set5 PSNR / SSIM | Verified Checkpoint Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **E1** | SRCNN | $2\times$ | MSE | `srcnn_2x_mse.pt` | 20,099 | T91 | Set14 (Leaked) | 26.69 dB / 0.8343 | 27.67 dB / 0.9081 | LEGACY / PRESERVED |
| **E2** | SRCNN | $2\times$ | MSE + Perceptual | `srcnn_2x_mse_perceptual.pt` | 20,099 | T91 | Set14 (Leaked) | 25.83 dB / 0.8614 | 26.17 dB / 0.9218 | LEGACY / PRESERVED |
| **E3** | SRCNN | $4\times$ | MSE | `srcnn_4x_mse.pt` | 20,099 | T91 | Set14 (Leaked) | 24.53 dB / 0.6903 | 25.47 dB / 0.8428 | LEGACY / PRESERVED |
| **E4** | SRCNN | $4\times$ | MSE + Perceptual | `srcnn_4x_mse_perceptual.pt` | 20,099 | T91 | Set14 (Leaked) | 23.03 dB / 0.6401 | 23.44 dB / 0.8227 | LEGACY / PRESERVED |
| **E5** | ESPCN | $2\times$ | MSE | `espcn_2x_mse.pt` | 48,716 | T91 | Set14 (Leaked) | 29.29 dB / 0.8545 | 31.99 dB / 0.9021 | LEGACY / PRESERVED |
| **E6** | ESPCN | $2\times$ | MSE + Perceptual | `espcn_2x_mse_perceptual.pt` | 48,716 | T91 | Set14 (Leaked) | 28.90 dB / 0.8350 | 30.22 dB / 0.8802 | LEGACY / PRESERVED |
| **E7** | ESPCN | $4\times$ | MSE | `espcn_4x_mse.pt` | 69,488 | T91 | Set14 (Leaked) | 25.15 dB / 0.6564 | 26.03 dB / 0.8062 | LEGACY / PRESERVED |
| **E8** | ESPCN | $4\times$ | MSE + Perceptual | `espcn_4x_mse_perceptual.pt` | 69,488 | T91 | Set14 (Leaked) | 24.44 dB / 0.6125 | 25.15 dB / 0.7871 | LEGACY / PRESERVED |

---

## Phase 3 Local DIV2K Experiments (E9–E14)

*Trained on DIV2K (800 train / 100 validation HR images) for 50 epochs with mixed precision (AMP) on NVIDIA RTX 2050 (4GB). Checkpoints verified.*

| ID | Architecture | Scale | Loss Function | Parameters | Epochs | Best Epoch | Final Train Loss | Best DIV2K Val PSNR / SSIM | Checkpoint File | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **E9** | ESPCN ($64\to 32$) | $2\times$ | Charbonnier ($\epsilon=10^{-3}$) | 26,796 | 50 | 49 | 0.01697 | **34.49 dB / 0.9178** | `outputs/checkpoints/phase3/E9_espcn_2x_best.pt` | VERIFIED COMPLETE |
| **E10** | FSRCNN | $2\times$ | Charbonnier ($\epsilon=10^{-3}$) | 24,683 | 50 | 49 | 0.01904 | **34.08 dB / 0.9149** | `outputs/checkpoints/phase3/E10_fsrcnn_2x_best.pt` | VERIFIED COMPLETE |
| **E11** | EDSR-Baseline | $2\times$ | L1 | 1,369,859 | 50 | 46 | 0.01498 | **35.43 dB / 0.9266** | `outputs/checkpoints/phase3/E11_edsr_2x_best.pt` | VERIFIED COMPLETE |
| **E12** | ESPCN ($64\to 32$) | $4\times$ | Charbonnier ($\epsilon=10^{-3}$) | 37,200 | 50 | 50 | 0.02919 | **28.48 dB / 0.7850** | `outputs/checkpoints/phase3/E12_espcn_4x_best.pt` | VERIFIED COMPLETE |
| **E13** | FSRCNN | $4\times$ | Charbonnier ($\epsilon=10^{-3}$) | 24,683 | 50 | 50 | 0.03099 | **28.30 dB / 0.7820** | `outputs/checkpoints/phase3/E13_fsrcnn_4x_best.pt` | VERIFIED COMPLETE |
| **E14** | EDSR-Baseline | $4\times$ | L1 | 1,517,571 | 50 | 46 | 0.02620 | **29.17 dB / 0.8020** | `outputs/checkpoints/phase3/E14_edsr_4x_best.pt` | VERIFIED COMPLETE |

---

## Canonical Benchmark Summary (Set5, Set14, BSD100) — Authoritative

*Tested on GPU with standard BT.601 Y-channel and scale border shaving.*

| ID | Model | Scale | Parameters | Set5 ($n=5$) PSNR / SSIM | Set14 ($n=14$) PSNR / SSIM | BSD100 ($n=100$) PSNR / SSIM | Status |
|---|---|---|---:|:---:|:---:|:---:|:---:|
| **REF-B2** | Bicubic Baseline | $2\times$ | 0 | 33.64 / 0.9369 | 30.31 / 0.8814 | 29.55 / 0.8569 | CANONICAL BASELINE |
| **E9** | ESPCN | $2\times$ | 26,796 | **35.58 / 0.9536** | **31.80 / 0.9090** | **30.85 / 0.8927** | PHASE 4 VERIFIED |
| **E10** | FSRCNN | $2\times$ | 24,683 | **34.81 / 0.9478** | **31.29 / 0.9043** | **30.51 / 0.8883** | PHASE 4 VERIFIED |
| **E11** | EDSR | $2\times$ | 1,369,859 | **36.75 / 0.9607** | **32.43 / 0.9166** | **31.39 / 0.9000** | PHASE 4 VERIFIED |
| **REF-B4** | Bicubic Baseline | $4\times$ | 0 | 28.42 / 0.8230 | 26.08 / 0.7216 | 25.96 / 0.6856 | CANONICAL BASELINE |
| **E12** | ESPCN | $4\times$ | 37,200 | **29.37 / 0.8444** | **26.74 / 0.7497** | **26.46 / 0.7171** | PHASE 4 VERIFIED |
| **E13** | FSRCNN | $4\times$ | 24,683 | **29.06 / 0.8385** | **26.52 / 0.7459** | **26.36 / 0.7138** | PHASE 4 VERIFIED |
| **E14** | EDSR | $4\times$ | 1,517,571 | **30.70 / 0.8809** | **27.63 / 0.7752** | **26.99 / 0.7352** | PHASE 4 VERIFIED |

*Phase 4.1 Consistency Audit Note: In `data/degradation.py`, `downscale_bicubic()` was updated from naive `cv2.resize(INTER_CUBIC)` to `imresize_matlab(..., antialiasing=True)`, harmonizing the clean surveillance condition with the canonical Set14 benchmark to 0.0000 dB.*

---

## Phase 4.2 Statistical Robustness Validation Summary

*Evaluated across Set14 ($n=14$) and BSD100 ($n=100$) over 5 degradation regimes (Clean, Sensor Noise $\sigma=10$, Motion Blur $k=5, 45^\circ$, JPEG $Q=40$, Combined CCTV) with 10,000 paired bootstrap resamples (seed 42). 9,120 image inferences.*

| Experiment / Evaluation | Scope | Models | Key Finding | Statistical Significance (BSD100 2×) | Artifacts |
|---|---|---|---|---|---|
| **Phase 4.2 Clean** | Set14, BSD100 | E9–E14, Legacy, Bicubic | 100% exact parity ($0.0000\text{ dB}$) with canonical benchmarks | E11: $+1.8435\text{ dB}$, CI: $[+1.6666, +2.0389]$ | `results/phase4_2_statistical_summary.csv` |
| **Phase 4.2 Sensor Noise** | Set14, BSD100 | E9–E14, Legacy, Bicubic | Unconditioned deep models amplify noise below Bicubic | E11: $-0.6242\text{ dB}$, CI: $[-0.7976, -0.4552]$ | `results/phase4_2_statistical_summary.csv` |
| **Phase 4.2 Motion Blur** | Set14, BSD100 | E9–E14, Legacy, Bicubic | Deep models deblur and sharpen reliably (100% win rate) | E11: $+0.7349\text{ dB}$, CI: $[+0.6908, +0.7819]$ | `results/phase4_2_statistical_summary.csv` |
| **Phase 4.2 Combined CCTV** | Set14, BSD100 | E9–E14, Legacy, Bicubic | Direct SR improvement over Bicubic is statistically inconclusive | E11: $-0.0018\text{ dB}$, CI: $[-0.0279, +0.0254]$ | `results/phase4_2_statistical_summary.csv` |

---

## Phase 4.3 Denoising Ablation Study Summary

*Evaluated preprocessing denoisers (A0: None, A1: Non-Local Means, A2: Bilateral Filter) on degraded inputs before SR across Set14 and BSD100. 4,104 image inferences.*

| Experiment / Ablation | Models | Denoiser | Runtime Overhead | PSNR Gain vs Raw A0 (BSD100 2× Sensor Noise) | Resulting Gain vs Bicubic Baseline | Operational Recommendation |
|---|---|---|:---:|:---:|:---:|---|
| **A1: Non-Local Means** | E9, E11, E12, E14 | OpenCV NLM ($h=3$) | $\approx 203\text{ ms}$ | $+0.1463\text{ dB}$ (CI: $[+0.0525, +0.2596]$) | Still below Bicubic A0 ($-0.4779\text{ dB}$) | Too slow for real-time CCTV; weak filtering |
| **A2: Bilateral Filter** | E9, E11, E12, E14 | OpenCV Bilateral ($d=5, \sigma=25$) | **$0.86\text{ ms}$** | **$+1.3533\text{ dB}$** (CI: $[+1.1714, +1.5508]$) | **$+0.7291\text{ dB}$ victory over Bicubic** | **Recommended for production & forensics** |


