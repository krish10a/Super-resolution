# Perceptual Checkpoint Provenance Report

**Step 1 — Unit 2** | ML Consistency Verification
**Repository:** `C:\Users\ASUS\Desktop\cv`
**Date:** 2026-11-01

---

## Executive Summary

The config-to-code mapping proves **intent** that E2, E4, E6, and E8 used perceptual (MSE+perceptual) loss: `configs/experiments.yaml` assigns them `loss: mse_perceptual`, and `train/train_all.py` maps `mse_perceptual` to `HybridLoss(lambda_perc=0.05)` (default `perc_warmup_epochs=0`). Train logs confirm 15-epoch runs with loss magnitudes consistent with a hybrid MSE+perceptual objective. However, **checkpoint provenance cannot be independently proven from the checkpoint files themselves**, because they are raw PyTorch `state_dict`s containing only model weights — no training metadata (epoch, loss type, lambda, optimizer state) is stored. This limitation is documented below.

---

## Experiment-to-Config Mapping (`configs/experiments.yaml`)

| Exp | Architecture | Scale | Loss (config) | Checkpoint |
|-----|--------------|-------|---------------|------------|
| E1 | SRCNN | 2× | mse | `srcnn_2x_mse.pt` |
| E2 | SRCNN | 2× | mse_perceptual | `srcnn_2x_mse_perceptual.pt` |
| E3 | SRCNN | 4× | mse | `srcnn_4x_mse.pt` |
| E4 | SRCNN | 4× | mse_perceptual | `srcnn_4x_mse_perceptual.pt` |
| E5 | ESPCN | 2× | mse | `espcn_2x_mse.pt` |
| E6 | ESPCN | 2× | mse_perceptual | `espcn_2x_mse_perceptual_verified.pt` |
| E7 | ESPCN | 4× | mse | `espcn_4x_mse_verified.pt` |
| E8 | ESPCN | 4× | mse_perceptual | `espcn_4x_mse_perceptual_verified.pt` |

Perceptual experiments: **E2, E4, E6, E8**.

## Code Path (`train/train_all.py`, lines 153-159)

```python
if loss_name in ('mse', 'mse_loss'):
    loss_fn = MSELoss()
elif loss_name in ('mse_perceptual', 'mse+perceptual', 'hybrid'):
    # Start with lambda_perc=0 for warmup (MSE-only), enable after warmup epochs
    loss_fn = HybridLoss(lambda_perc=0.0 if args.perc_warmup_epochs > 0 else perceptual_lambda)
else:
    raise ValueError(f'Unknown loss: {loss_name}')
```

- Default `perc_warmup_epochs=0` → `lambda_perc = perceptual_lambda = 0.05` (from `configs/experiments.yaml` hyperparams).
- Perceptual loss active from epoch 1 for all perceptual experiments (no warmup).
- `perceptual_lambda` in config is `0.05` (note: `losses/perceptual_loss.py` default is `0.07`; training used `0.05`).

## Train Log Evidence

Logs exist for all four perceptual experiments (`e2_train.log`, `e4_train.log`, `e6_train.log`, `e8_train.log`).

| Exp | Epochs run | Best val PSNR | Notes |
|-----|------------|---------------|-------|
| E2 | 15 | ~26.62 | Loss decreases 0.087→0.110 |
| E4 | 11 (early stop) | ~21.97 | Early-stopped at patience |
| E6 | 15 | ~28.62 | Loss decreases 0.128→0.131 |
| E8 | 15 | ~23.06 | Loss decreases 0.195→0.236 |

The loss magnitudes and non-monotonic behavior (jumps around epoch 4-5) are consistent with a hybrid MSE+perceptual objective where the perceptual term introduces higher-frequency gradient signals. Logs do not explicitly record `lambda_perc` or "perceptual" per line, so they corroborate but do not independently prove perceptual loss usage.

## Limitation: Checkpoints are raw state_dicts

All checkpoints are raw `torch.save(model.state_dict())` files. They contain:

- Model weights only (Conv2d weights/biases, PixelShuffle has no params).
- **No** training metadata: no epoch, no loss type, no `lambda_perc`, no optimizer state, no validation history.

This means:
- Loading `srcnn_2x_mse_perceptual.pt` into SRCNN and running inference works, but the file itself cannot prove it was trained with perceptual loss vs. MSE loss.
- A checkpoint could have been produced by any training config; filename is the only provenance signal, and filenames are convention, not cryptographic proof.

## Conclusion

**Provenance status: INTENT established, INDEPENDENT PROOF not possible.**

- The config + training code together establish that E2/E4/E6/E8 were *intended* to use MSE+perceptual loss with `lambda_perc=0.05`.
- Train logs corroborate 15-epoch hybrid-style runs.
- Checkpoint files cannot independently verify which loss produced them (raw state_dicts, no metadata).

**Recommendation:** Treat the perceptual attribution of E2/E4/E6/E8 as config-documented intent, not independently verifiable fact. Do not silently relabel or re-attribute existing checkpoints. Future training should embed metadata (epoch, loss, lambda, dataset hash) in checkpoint files to enable independent provenance.

No checkpoints were modified, overwritten, or retrained.
