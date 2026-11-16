# ARCHITECTURE COMPATIBILITY & DIAGNOSTIC REPORT

**Date:** 2026-11-16  
**Auditor:** Senior CV Research Engineer & Deep Learning Architect  

---

## 1. SRCNN Architecture & Output Activation Analysis

### Official Academic Formulation (Dong et al., TPAMI 2016)
- **Patch Extraction & Representation**: $\text{Conv}(3, 64, 9\times 9, \text{pad}=4) \to \text{ReLU}$
- **Non-linear Mapping**: $\text{Conv}(64, 32, 1\times 1, \text{pad}=0) \to \text{ReLU}$ (or $5\times 5$ in 9-5-5 variant)
- **Reconstruction**: $\text{Conv}(32, 3, 5\times 5, \text{pad}=2) \to \text{Linear}$ (No non-linear output activation)

### Existing Local Implementation (`models/srcnn.py`)
```python
self.net = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=9, padding=4),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 32, kernel_size=1, padding=0),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 3, kernel_size=5, padding=2),
    nn.Tanh(),  # <--- Non-standard output activation
)
```

### Empirical Distribution Analysis (`srcnn_2x_mse.pt` on Set5)
- **Target Range**: $[0.0, 1.0]$ (via `torchvision.transforms.functional.to_tensor`)
- **Raw Network Output Minimum**: **$-0.2008$**
- **Raw Network Output Maximum**: **$+0.9546$**
- **Fraction of Negative Pixels ($<0.0$)**: **$5.09\%$**
- **Fraction of Near-Saturation Pixels ($>0.95$)**: $<0.001\%$

### Architectural Diagnostic Findings:
1. **Dynamic Range Mismatch**: Because Tanh maps to $[-1.0, 1.0]$, $5.1\%$ of generated pixels produce negative intensities. Clamping to $[0, 1]$ truncates these to 0, destroying shadow gradients.
2. **Gradient Saturation Near Highlights**: As outputs approach $1.0$, $\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x) \to 0$. Gradients vanish for bright highlights, hindering convergence.
3. **Phase 2 / Phase 3 Resolution Policy**:
   - For Phase 2: Preserve existing `models/srcnn.py` with `nn.Tanh()` to maintain strict backwards-compatibility with legacy checkpoints `srcnn_2x_mse.pt`, `srcnn_4x_mse.pt`, `srcnn_2x_mse_perceptual.pt`, `srcnn_4x_mse_perceptual.pt`.
   - For Phase 3: Introduce `SRCNNLinear` (or `activation=None`) for newly trained experiments (E9+).

---

## 2. ESPCN Architecture & Channel Configuration Analysis

### Official Academic Formulation (Shi et al., CVPR 2016)
- **Feature Extraction**: $\text{Conv}(3, 64, 5\times 5, \text{pad}=2) \to \text{ReLU}$
- **Feature Mapping**: $\text{Conv}(64, 32, 3\times 3, \text{pad}=1) \to \text{ReLU}$  ($32$ channels)
- **Sub-Pixel Convolution**: $\text{Conv}(32, 3 \cdot r^2, 3\times 3, \text{pad}=1) \to \text{PixelShuffle}(r)$

### Existing Local Checkpoints Analysis
Two distinct ESPCN architectures exist in `outputs/checkpoints/`:

| Architecture Variant | Conv1 Channels | Conv2 Channels | Conv3 Channels ($2\times$) | Conv3 Channels ($4\times$) | Total Params ($2\times$) | Total Params ($4\times$) | Associated Checkpoints |
|---|---|---|---|---|---|---|---|
| **Variant A (Wide)** | $3 \to 64$ | **$64 \to 64$** | $64 \to 12$ | $64 \to 48$ | **48,716** | **69,488** | `espcn_2x_mse.pt`<br>`espcn_2x_mse_perceptual.pt`<br>`espcn_4x_mse.pt`<br>`espcn_4x_mse_perceptual.pt`<br>`espcn_best.pt` |
| **Variant B (Standard)** | $3 \to 64$ | **$64 \to 32$** | $32 \to 12$ | $32 \to 48$ | **26,796** | **37,200** | `espcn_real_best.pt` |

### Architectural Diagnostic Findings:
1. `models/espcn.py` in the repository currently instantiates **Variant A** (`channels=64` for both Conv1 and Conv2).
2. Variant A strictly matches the 8 original checkpoints (E5–E8).
3. `espcn_real_best.pt` requires Variant B (`mid_channels=32`).
4. **Resolution Policy**:
   - Parameterize `ESPCN(scale=2, in_channels=3, channels=64, mid_channels=64)` so that it defaults to 64 for backward compatibility with E1–E8, but allows `mid_channels=32` when loading `espcn_real_best.pt` or training standard Shi et al. models in Phase 3.
