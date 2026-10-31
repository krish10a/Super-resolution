# PROJECT_CONTEXT — SISR Surveillance (SRCNN / ESPCN)

Single source of truth for model design, losses, training curriculum, and experiment results.

## Scope
- Single-Image Super-Resolution (SISR), 2× and 4×, for low-quality surveillance feeds.
- Architectures: SRCNN (9-1-5 conv, He init, Tanh output), ESPCN (PixelShuffle upscale, He init).
- Losses: MSE vs Hybrid MSE + λ·Perceptual (VGG11 frozen, relu2_2/relu3_3, λ in [0.05, 0.1]).
- Metrics: PSNR/SSIM on Y-channel (RGB→YCbCr), borders shaved by scale pixels.
- Cross-dataset: train on Set5, test on Set14. Bicubic baseline required for every experiment.

## Model specs
| Model | Scale | Upscale method | Output activation | Init |
|-------|-------|----------------|-------------------|------|
| SRCNN | 2, 4  | Bicubic upsample input (spec §6) | Tanh | He (kaiming) |
| ESPCN | 2, 4  | PixelShuffle (sub-pixel conv) | Sigmoid/clamp [0,1] | He (kaiming) |

SRCNN conv stack: Conv2d(3→64, 9×9, p=4) → ReLU → Conv2d(64→32, 1×1) → ReLU → Conv2d(32→3, 5×5, p=2) → Tanh.
ESPCN: feature extraction convs → sub-pixel conv (channels × scale²) → PixelShuffle(scale).

## Losses
- `MSELoss`: plain nn.MSELoss on RGB tensors [0,1].
- `HybridLoss(lambda_perc=0.07)`: MSE + 0.07 · MSE(VGG11 features(pred), VGG11 features(target)).
- VGG11 perceptual: frozen, feature_layer ∈ {relu2_2 (children[:9]), relu3_3 (children[:16])}.

## Training
- Optimizer: Adam lr 1e-4.
- Scheduler: ReduceLROnPlateau / CosineAnnealing on train loss.
- AMP + grad accum (steps=4) on RTX 2050 (4GB VRAM).
- Early stopping: patience 15 on val PSNR.
- Windows: DataLoader num_workers=0, no pin_memory on CPU.
- Patch size (LR): 32 for 2×, 48 for 4×; random crop + hflip augmentation.
- Batch size: 4 (local MSE), 2 (local perceptual), 8 (Colab).

## Experiments (configs/experiments.yaml)
- E1: SRCNN 2× MSE, train Set5 → test Set14
- E2: SRCNN 4× MSE, train Set5 → test Set14
- E3: ESPCN 2× MSE, train Set5 → test Set14
- E4: ESPCN 4× MSE, train Set5 → test Set14
- E5: ESPCN 2× MSE+Perceptual (Colab)
- E6: ESPCN 4× MSE+Perceptual (Colab)
- E7: ESPCN 4× MSE, train Set5_degraded → test Set14
- E8: ESPCN 4× MSE+Perceptual, train Set5_degraded → test Set14

Degraded training (E7/E8): on-the-fly LR generation via `data/degradation.py` (blur → downscale → noise → JPEG).

## Results
To be filled in as experiments complete. Each entry must include bicubic baseline PSNR/SSIM for comparison.

| Exp | Model | Scale | Loss | Train | Test | Val PSNR | Val SSIM | Bicubic PSNR |
|-----|-------|-------|------|-------|------|----------|----------|--------------|

## Portability
- All paths resolve via `config/paths.py` relative to project root — no hardcoded `C:\` paths.
- Checkpoints: `outputs/checkpoints/{model}_{scale}x_{loss}.pt`.
- Data: `data/pairs/{Set5,Set14}_LR_x{scale}` and `_HR`.
