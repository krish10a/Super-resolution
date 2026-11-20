# PHASE 4.3: DENOISING ABLATION STUDY REPORT

**Project:** Deep Learning-Based Single-Image Super-Resolution for CCTV Surveillance Imagery  
**Hardware:** NVIDIA GeForce RTX 2050 (4GB VRAM, CUDA AMP), AMD Ryzen 7 5800HS, 16GB RAM  
**Datasets:** Set14 ($n=14$), BSD100 ($n=100$)  
**Scaling Factors:** $2\times$ and $4\times$  
**Degradation Regimes:** Sensor Noise ($\sigma=10.0$) and Combined CCTV Pipeline  
**Date:** November 16, 2026  
**Status:** COMPLETE & EMPIRICALLY VERIFIED  

---

## 1. Executive Summary

Phase 4.3 investigates whether classical edge-preserving spatial denoisers applied prior to deep SISR inference resolve the severe noise-amplification failure identified in Phase 4.2. 

We systematically ablate three preprocessing regimes across 4,104 paired model inferences:
- **A0 (Baseline):** No preprocessing; direct SR inference on degraded low-resolution input.
- **A1 (Non-Local Means):** OpenCV `fastNlMeansDenoisingColored` ($h=3, h_{\text{Color}}=3$, template size $7$, search window $21$).
- **A2 (Bilateral Filter):** OpenCV `bilateralFilter` ($d=5, \sigma_{\text{Color}}=25, \sigma_{\text{Space}}=25$).

### Key Empirical Findings:

1. **Bilateral Filtering (A2) Rescues Deep SISR Under Sensor Noise:**
   - Under raw Gaussian sensor noise ($\sigma=10$), direct inference (**A0**) caused EDSR ($2\times$) to suffer a $-0.6242\text{ dB}$ degradation relative to Bicubic on BSD100.
   - Inserting Bilateral Filtering (**A2**) yields a dramatic **$+1.3533\text{ dB}$ PSNR gain** over A0 ($95\%\text{ CI: } [+1.1714, +1.5508]\text{ dB}$).
   - This elevates EDSR from $27.5204\text{ dB}$ to **$28.8737\text{ dB}$**, converting a $-0.62\text{ dB}$ loss into a **$+0.7291\text{ dB}$ decisive victory over Bicubic** ($95\%\text{ CI: } [+0.5591, +0.9022]\text{ dB}$).
   - On Set14, EDSR gains **$+1.3934\text{ dB}$** ($95\%\text{ CI: } [+0.8809, +1.9461]\text{ dB}$), reaching $29.4682\text{ dB}$.

2. **Extreme Efficiency of Bilateral Filtering:**
   - Bilateral filtering requires only **$0.83\text{ ms} - 1.02\text{ ms}$** of CPU compute on $2\times$ LR frames and **$0.35\text{ ms}$** on $4\times$ LR frames.
   - Combined with ESPCN ($2.4\text{ ms}$ GPU latency), the entire pipeline operates in **$\approx 3.2\text{ ms}$ total latency** ($>300\text{ FPS}$), fully viable for real-time edge CCTV streams.

3. **Non-Local Means (A1) is Computationally Infeasible for Real-Time Use:**
   - NLM requires **$201\text{ ms} - 267\text{ ms}$** per frame on CPU ($>200\times$ slower than Bilateral).
   - Despite its heavy computational cost, NLM yields only a modest $+0.1463\text{ dB}$ gain on EDSR under sensor noise, heavily trailing Bilateral filtering ($+1.3533\text{ dB}$).

4. **Combined CCTV Improvement:**
   - Under the full combined CCTV degradation, Bilateral filtering (**A2**) delivers a statistically verified $+0.1611\text{ dB}$ (ESPCN) and $+0.1705\text{ dB}$ (EDSR) gain on BSD100 ($95\%\text{ CI: } [+0.1319, +0.2150]\text{ dB}$).

---

## 2. Experimental Results: BSD100 ($n=100$)

### Scale 2× Ablation (BSD100)

| Condition | Denoiser | Model | Mean PSNR | $\Delta\text{PSNR vs A0}$ | 95% Bootstrap CI (vs A0) | $\Delta\text{PSNR vs Bic A0}$ | Denoiser Latency | Model Latency | Total Latency |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sensor Noise** | **A0 (None)** | Bicubic | 28.1446 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.00 ms | 2.16 ms | 2.16 ms |
| ($\sigma=10$) | | E9 (ESPCN) | 27.5849 | 0.0000 | [0.0000, 0.0000] | -0.5598 | 0.00 ms | 2.52 ms | 2.52 ms |
| | | E11 (EDSR) | 27.5204 | 0.0000 | [0.0000, 0.0000] | -0.6242 | 0.00 ms | 49.14 ms | 49.14 ms |
| | **A1 (NLM)** | Bicubic | 28.1973 | +0.0526 | [+0.0055, +0.1135] | +0.0526 | 203.39 ms | 2.00 ms | 205.39 ms |
| | | E9 (ESPCN) | 27.7327 | +0.1479 | [+0.0613, +0.2528] | -0.4119 | 203.39 ms | 2.42 ms | 205.81 ms |
| | | E11 (EDSR) | 27.6667 | +0.1463 | [+0.0525, +0.2596] | -0.4779 | 203.39 ms | 48.95 ms | 252.34 ms |
| | **A2 (Bilateral)** | Bicubic | 28.5224 | +0.3778 | [+0.2862, +0.4828] | +0.3778 | **0.86 ms** | 2.15 ms | **3.01 ms** |
| | | E9 (ESPCN) | **28.7406** | **+1.1557** | **[+0.9907, +1.3365]** | **+0.5960** | **0.86 ms** | 2.41 ms | **3.27 ms** |
| | | E11 (EDSR) | **28.8737** | **+1.3533** | **[+1.1714, +1.5508]** | **+0.7291** | **0.86 ms** | 48.96 ms | **49.82 ms** |
| **Combined CCTV** | **A0 (None)** | Bicubic | 26.4114 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.00 ms | 2.18 ms | 2.18 ms |
| | | E9 (ESPCN) | 26.4076 | 0.0000 | [0.0000, 0.0000] | -0.0038 | 0.00 ms | 2.41 ms | 2.41 ms |
| | | E11 (EDSR) | 26.4096 | 0.0000 | [0.0000, 0.0000] | -0.0018 | 0.00 ms | 49.10 ms | 49.10 ms |
| | **A1 (NLM)** | Bicubic | 26.4270 | +0.0156 | [-0.0010, +0.0369] | +0.0156 | 201.67 ms | 2.04 ms | 203.71 ms |
| | | E9 (ESPCN) | 26.4691 | +0.0615 | [+0.0379, +0.0905] | +0.0577 | 201.67 ms | 2.40 ms | 204.07 ms |
| | | E11 (EDSR) | 26.4705 | +0.0609 | [+0.0376, +0.0895] | +0.0591 | 201.67 ms | 49.01 ms | 250.68 ms |
| | **A2 (Bilateral)** | Bicubic | 26.4714 | +0.0600 | [+0.0332, +0.0916] | +0.0600 | **0.83 ms** | 2.12 ms | **2.95 ms** |
| | | E9 (ESPCN) | **26.5687** | **+0.1611** | **[+0.1231, +0.2052]** | **+0.1573** | **0.83 ms** | 2.45 ms | **3.28 ms** |
| | | E11 (EDSR) | **26.5801** | **+0.1705** | **[+0.1319, +0.2150]** | **+0.1687** | **0.83 ms** | 49.13 ms | **49.96 ms** |

---

## 3. Experimental Results: Set14 ($n=14$)

### Scale 2× Ablation (Set14)

| Condition | Denoiser | Model | Mean PSNR | $\Delta\text{PSNR vs A0}$ | 95% Bootstrap CI (vs A0) | $\Delta\text{PSNR vs Bic A0}$ | Denoiser Latency |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Sensor Noise** | **A0 (None)** | Bicubic | 28.7337 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.00 ms |
| ($\sigma=10$) | | E9 (ESPCN) | 28.1336 | 0.0000 | [0.0000, 0.0000] | -0.6002 | 0.00 ms |
| | | E11 (EDSR) | 28.0748 | 0.0000 | [0.0000, 0.0000] | -0.6590 | 0.00 ms |
| | **A1 (NLM)** | Bicubic | 28.7849 | +0.0512 | [-0.0104, +0.1345] | +0.0512 | 266.38 ms |
| | | E9 (ESPCN) | 28.2718 | +0.1382 | [+0.0132, +0.2929] | -0.4619 | 266.38 ms |
| | | E11 (EDSR) | 28.2141 | +0.1393 | [+0.0076, +0.3013] | -0.5196 | 266.38 ms |
| | **A2 (Bilateral)** | Bicubic | 29.1171 | +0.3833 | [+0.2638, +0.5284] | +0.3833 | **1.04 ms** |
| | | E9 (ESPCN) | **29.3364** | **+1.2028** | **[+0.7937, +1.7456]** | **+0.6027** | **1.04 ms** |
| | | E11 (EDSR) | **29.4682** | **+1.3934** | **[+0.8809, +1.9461]** | **+0.7345** | **1.04 ms** |
| **Combined CCTV** | **A0 (None)** | Bicubic | 26.6860 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.00 ms |
| | | E9 (ESPCN) | 26.7336 | 0.0000 | [0.0000, 0.0000] | +0.0476 | 0.00 ms |
| | | E11 (EDSR) | 26.7339 | 0.0000 | [0.0000, 0.0000] | +0.0479 | 0.00 ms |
| | **A2 (Bilateral)** | Bicubic | 26.7785 | +0.0925 | [+0.0269, +0.1648] | +0.0925 | **1.02 ms** |
| | | E9 (ESPCN) | **26.9247** | **+0.1911** | **[+0.1052, +0.2838]** | **+0.2387** | **1.02 ms** |
| | | E11 (EDSR) | **26.9374** | **+0.2035** | **[+0.1164, +0.2977]** | **+0.2514** | **1.02 ms** |

---

## 4. Scale 4× Ablation Summary (BSD100)

| Condition | Denoiser | Model | Mean PSNR | $\Delta\text{PSNR vs A0}$ | 95% Bootstrap CI (vs A0) | Denoiser Latency |
|---|---|---|:---:|:---:|:---:|:---:|
| **Sensor Noise** | **A0 (None)** | Bicubic | 25.2719 | 0.0000 | [0.0000, 0.0000] | 0.00 ms |
| | | E12 (ESPCN) | 25.0670 | 0.0000 | [0.0000, 0.0000] | 0.00 ms |
| | | E14 (EDSR) | 25.1544 | 0.0000 | [0.0000, 0.0000] | 0.00 ms |
| | **A2 (Bilateral)** | Bicubic | 25.4512 | +0.1793 | [+0.1170, +0.2483] | **0.35 ms** |
| | | E12 (ESPCN) | **25.5683** | **+0.5013** | **[+0.3804, +0.6300]** | **0.35 ms** |
| | | E14 (EDSR) | **25.7533** | **+0.5989** | **[+0.4571, +0.7490]** | **0.35 ms** |
| **Combined CCTV** | **A0 (None)** | Bicubic | 24.1610 | 0.0000 | [0.0000, 0.0000] | 0.00 ms |
| | | E12 (ESPCN) | 24.1793 | 0.0000 | [0.0000, 0.0000] | 0.00 ms |
| | | E14 (EDSR) | 24.2386 | 0.0000 | [0.0000, 0.0000] | 0.00 ms |
| | **A2 (Bilateral)** | Bicubic | 24.2322 | +0.0712 | [+0.0381, +0.1084] | **0.35 ms** |
| | | E12 (ESPCN) | **24.3168** | **+0.1375** | **[+0.0967, +0.1818]** | **0.35 ms** |
| | | E14 (EDSR) | **24.3980** | **+0.1594** | **[+0.1152, +0.2079]** | **0.35 ms** |

---

## 5. Architectural & Deployment Recommendations

1. **Mandatory Denoising Stage for Surveillance SISR:**
   Deploying deep SISR models directly onto noisy surveillance streams without a front-end noise filter leads to sub-bicubic visual degradation. Adding a lightweight bilateral filter is non-negotiable for operational CCTV deployments.

2. **The Ideal Real-Time Pipeline ($>300\text{ FPS}$):**
   $$\text{Raw LR Frame} \xrightarrow[\approx 0.8\text{ms}]{\text{Bilateral Filter}} \text{Denoised LR} \xrightarrow[\approx 2.4\text{ms}]{\text{ESPCN } (2\times)} \text{High-Resolution Frame}$$
   - Total latency: **$3.2\text{ ms}$** on consumer GPU/CPU.
   - Quality: $28.74\text{ dB}$ on BSD100 ($+0.60\text{ dB}$ over Bicubic).

3. **The Ideal Forensic Pipeline:**
   $$\text{Raw LR Frame} \xrightarrow[\approx 0.8\text{ms}]{\text{Bilateral Filter}} \text{Denoised LR} \xrightarrow[\approx 49\text{ms}]{\text{EDSR } (2\times)} \text{Forensic Reconstruction}$$
   - Quality: $28.87\text{ dB}$ on BSD100 ($+0.73\text{ dB}$ over Bicubic), with maximum structural edge definition.

---

## 6. Files & Artifacts Generated

- Per-image results: `results/phase4_3_denoising_per_image_metrics.csv` (4,104 rows)
- Summary statistics: `results/phase4_3_denoising_summary.csv` (72 condition/denoiser/model pairs)
- JSON dataset: `results/phase4_3_denoising_ablation.json`
