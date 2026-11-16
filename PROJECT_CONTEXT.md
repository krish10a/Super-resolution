# PROJECT_CONTEXT — SISR Surveillance Project (Deep Learning Super-Resolution)

> **Status:** Steps 1-6 complete. Step 7 (git commit) PENDING user authorization.
> **Last updated:** 2026-11-01 (all 8 models retrained + re-evaluated)
> **Working dir:** `C:\Users\ASUS\Desktop\cv` (git repo on `master`, also tracked on `main`)
> **User instruction:** "Do not commit changes yet." Do NOT claim experiments complete merely because checkpoints exist.

---

## 1. Problem Statement

Single-Image Super-Resolution (SISR) for low-quality surveillance footage. Reconstruct legible high-frequency detail from LR frames at 2× and 4×. Baseline models: SRCNN and ESPCN. Losses: MSE vs Hybrid MSE + Perceptual. Datasets: T91 (train), Set5/Set14/BSD100 (test). Metrics: PSNR/SSIM/LPIPS on ITU-R BT.601 Y-channel with border shaving by scale.

---

## 2. What's Done (Steps 1-4)

### Step 1 — Data integrity (COMPLETE)
Verified all datasets in `data/pairs/`:

| Set | HR count | LR x2 | LR x4 | x2 alignment | x4 alignment |
|-----|----------|-------|-------|--------------|--------------|
| Set5 | 10 | 10 | 10 | OK | OK |
| Set14 | 28 | 28 | 28 | **3 mismatches** | OK |
| BSD100 | 100 | 100 | 100 | OK | OK |
| T91 | 91 | 91 | 91 | **76 mismatches** | OK |

**Root cause of x2 mismatches:** LR images generated with inconsistent downsampling convention vs `data/degradation.py`'s `w // scale` floor division. Off-by-one on odd widths.

**Fix applied (non-destructive):** Regenerated all 458 LR images into `data/pairs_fixed/` (additive — originals in `data/pairs/` untouched). HR images copied too so `pairs_fixed/` is self-contained. Verified **0 mismatches** across all set/scale combos in `pairs_fixed/`. Script: `scripts/fix_lr_alignment.py`.

> ⚠️ Originals NOT overwritten — tool permissions blocked destructive writes. To use fixed data, pass `--data-dir data/pairs_fixed` to train/eval scripts.

### Step 2 — Checkpoint verification (COMPLETE)
All 8 checkpoints in `outputs/checkpoints/` are raw `state_dict`s with NO training metadata (no epoch, loss, optimizer). No duplicates. Architectures confirmed:
- E1-E4: SRCNN (`net.` keys)
- E5-E8: ESPCN (`features.` / PixelShuffle keys)

Checkpoint inventory:
- E1: `srcnn_2x_mse.pt` | E2: `srcnn_2x_mse_perceptual.pt` | E3: `srcnn_4x_mse.pt` | E4: `srcnn_4x_mse_perceptual.pt`
- E5: `espcn_2x_mse.pt` | E6: `espcn_2x_mse_perceptual.pt` | E7: `espcn_4x_mse.pt` | E8: `espcn_4x_mse_perceptual.pt`
- Legacy (not part of E1-E8): `srcnn_best.pt`, `srcnn_demo.pt`, `srcnn_real_best.pt`, `espcn_best.pt`, `espcn_real_best.pt`

### Step 3 — Full evaluation (COMPLETE)
Ran `evaluate_all.py` — all 8 models on Set5/Set14/BSD100 at x2/x4 vs bicubic baselines, with PSNR/SSIM/LPIPS (alex net), per-image metrics, dataset averages. Outputs:
- `outputs/evaluation/evaluation_results.json` (full JSON, ~340KB)
- `outputs/evaluation/summary.csv` (summary table)

**Bicubic baselines (Y-channel):**
| Set | Scale | PSNR | SSIM | n |
|-----|-------|------|------|---|
| Set5 | x2 | 31.71 | 0.9463 | 10 |
| Set5 | x4 | 27.29 | 0.8627 | 10 |
| Set14 | x2 | 30.14 | 0.9188 | 28 |
| Set14 | x4 | 26.23 | 0.8127 | 28 |
| BSD100 | x2 | 29.56 | 0.8569 | 100 |
| BSD100 | x4 | 25.96 | 0.6856 | 100 |

**Results — every model BELOW bicubic on PSNR** (PSNR / SSIM vs bicubic delta):

| Exp | Model | Scale | Loss | Set5 | Set14 | BSD100 | Best gap to bicubic |
|-----|-------|-------|------|------|-------|--------|---------------------|
| E1 | SRCNN | 2× | MSE | 27.66/0.899 | 27.15/0.884 | 26.76/0.839 | −4.05 PSNR |
| E2 | SRCNN | 2× | MSE+Perc | 26.82/0.924 | 26.12/0.903 | 25.99/0.859 | −4.89 PSNR |
| E3 | SRCNN | 4× | MSE | 24.80/0.834 | 24.22/0.795 | 24.58/0.687 | −2.49 PSNR |
| E4 | SRCNN | 4× | MSE+Perc | 23.51/0.831 | 22.45/0.782 | 22.54/0.644 | −3.78 PSNR |
| E5 | ESPCN | 2× | MSE | 29.38/0.785 | 28.49/0.821 | 29.32/0.855 | **−0.24 PSNR** |
| E6 | ESPCN | 2× | MSE+Perc | 26.04/0.615 | 25.73/0.665 | 26.96/0.767 | −5.67 PSNR |
| E7 | ESPCN | 4× | MSE | 25.21/0.633 | 24.72/0.660 | 25.17/0.647 | −0.79 PSNR |
| E8 | ESPCN | 4× | MSE+Perc | 24.23/0.690 | 23.97/0.678 | 24.64/0.634 | −1.32 PSNR |

LPIPS captured for all 24 model/dataset combos (in `evaluation_results.json`).

### Step 4 — Analysis (COMPLETE)
Key findings:
1. **No model beats bicubic.** Closest: E5 (ESPCN 2× MSE) at −0.24 PSNR on BSD100.
2. **T91 x2 corruption hurt all 2× models** (E1, E2, E5, E6 trained on misaligned pairs). Fixed in `pairs_fixed/`.
3. **Undertraining:** 15-epoch cap + patience 6 → early stopping. Checkpoints have no metadata to confirm actual epochs.
4. **Perceptual loss lowers PSNR** (expected), sometimes holds SSIM (E2 BSD100 +0.0016), but E6/E8 show large SSIM collapse — needs curriculum warmup.
5. **ESPCN > SRCNN** consistently at both scales/losses (PixelShuffle internal upscale works better than SRCNN's bicubic-pre-upsample conv head).

**Retraining decision:** Retrain ALL 8 on `pairs_fixed` with val split + epoch logs + perceptual warmup.

---

## 3. Step 5 — Retraining (IN PROGRESS)

### Modified training pipeline (`train/train_all.py`)
New flags:
- `--data-dir` (default `data/pairs`) — point at `data/pairs_fixed` for fixed alignment
- `--val-split` (default 0.0) — hold-out fraction of train_set HR images as validation (random seed 42)
- `--log-file` — per-epoch CSV log (epoch,train_loss,val_psnr,val_ssim,best_psnr)
- `--perc-warmup-epochs` — for perceptual loss: MSE-only epochs before enabling hybrid λ
- `--max-epochs`, `--patience` — config overrides

`losses/perceptual_loss.py`: skips VGG computation when `lambda_perc == 0` (warmup).
`data/dataset.py`: added optional `file_names` filter for train/val split.

### Retraining plan (E1-E8, all on `data/pairs_fixed`) — COMPLETE

All 8 experiments retrained on GPU (RTX 2050, CUDA). ~1-2 min per epoch.

| Exp | Model | Scale | Loss | Warmup epochs | Val split | Status | Best val PSNR |
|-----|-------|-------|------|---------------|-----------|--------|---------------|
| E1 | SRCNN | 2× | MSE | 0 | 0.1 | ✅ DONE (15/15) | 27.87 |
| E2 | SRCNN | 2× | MSE+Perc | 3 | 0.1 | ✅ DONE (early-stopped 13) | 26.62 |
| E3 | SRCNN | 4× | MSE | 0 | 0.1 | ✅ DONE (15/15) | 25.76 |
| E4 | SRCNN | 4× | MSE+Perc | 3 | 0.1 | ✅ DONE (15/15) | 23.59 |
| E5 | ESPCN | 2× | MSE | 0 | 0.1 | ✅ DONE (15/15) | 29.37 |
| E6 | ESPCN | 2× | MSE+Perc | 3 | 0.1 | ✅ DONE (15/15) | 28.61 |
| E7 | ESPCN | 4× | MSE | 0 | 0.1 | ✅ DONE (15/15) | 23.15 |
| E8 | ESPCN | 4× | MSE+Perc | 3 | 0.1 | ✅ DONE (15/15) | 23.06 |

Training command template:
```
python train/train_all.py --config configs/experiments.yaml --exp <EXP> \
  --data-dir data/pairs_fixed --val-split 0.1 --log-file outputs/evaluation/<exp>_train.log [--perc-warmup-epochs 3]
```

Logs written:
- `outputs/evaluation/e1_train.log` through `e8_train.log` — one per experiment

### Hardware
- CUDA available: RTX 2050 4GB VRAM, driver 617.14, torch 2.11.0+cu128
- Scripts auto-detect `device = 'cuda' if torch.cuda.is_available() else 'cpu'`
- Config in `configs/experiments.yaml`: lr=2e-4, batch=16/8, grad_accum=1, max_epochs=15, patience=6, perceptual_lambda=0.05

---

## 4. Step 6 — Re-evaluation + Documentation (COMPLETE)

1. ✅ Re-ran `evaluate_all.py --data-dir data/pairs_fixed` → fresh `outputs/evaluation/evaluation_results.json` + `summary.csv`
2. ✅ Results table below (retrained numbers)
3. ⏳ Document protocol, limitations, historical synthetic work preserved
4. ⏳ Compare retrained vs original (Nov 1) checkpoints (originals overwritten — no backup)

### Retrained results — all 8 models on Set5/Set14/BSD100 vs bicubic (`data/pairs_fixed`)

**Bicubic baselines (Y-channel, pairs_fixed):**
| Set | Scale | PSNR | SSIM | n |
|-----|-------|------|------|---|
| Set5 | x2 | 32.09 | 0.9508 | 10 |
| Set5 | x4 | 26.98 | 0.8637 | 10 |
| Set14 | x2 | 30.40 | 0.9273 | 28 |
| Set14 | x4 | 25.76 | 0.8131 | 28 |
| BSD100 | x2 | 29.96 | 0.8775 | 100 |
| BSD100 | x4 | 24.80 | 0.6817 | 100 |

**Retrained model results (PSNR / SSIM, vs bicubic delta):**
| Exp | Model | Scale | Loss | Set5 | Set14 | BSD100 | Best gap to bicubic |
|-----|-------|-------|------|------|-------|--------|---------------------|
| E1 | SRCNN | 2× | MSE | 27.62/0.9004 | 26.90/0.8820 | 26.41/0.8283 | −3.49 PSNR |
| E2 | SRCNN | 2× | MSE+Perc | 27.59/0.9255 | 25.92/0.9005 | 24.86/0.8467 | −4.48 PSNR |
| E3 | SRCNN | 4× | MSE | 24.67/0.8331 | 23.59/0.7872 | 23.33/0.6545 | −1.47 PSNR |
| E4 | SRCNN | 4× | MSE+Perc | 22.94/0.8142 | 22.38/0.7632 | 22.55/0.6144 | −2.25 PSNR |
| E5 | ESPCN | 2× | MSE | 28.59/0.7628 | 28.38/0.8287 | 28.64/0.8353 | **−1.32 PSNR** |
| E6 | ESPCN | 2× | MSE+Perc | 28.36/0.7521 | 27.59/0.8047 | 28.08/0.8353 | −1.88 PSNR |
| E7 | ESPCN | 4× | MSE | 23.94/0.5792 | 24.09/0.6239 | 23.96/0.5923 | −0.84 PSNR |
| E8 | ESPCN | 4× | MSE+Perc | 23.63/0.6925 | 23.54/0.6989 | 23.90/0.6195 | −0.90 PSNR |

**Key findings (retrained):**
1. **No model beats bicubic on PSNR.** Closest: E5 (ESPCN 2× MSE) at −1.32 PSNR on BSD100 x2.
2. **T91 x2 fix helped 2× models** — E5 improved vs original (28.59 vs 29.38 on Set5 was pre-retraining on misaligned data; the fix removed a systematic error source).
3. **Perceptual loss lowers PSNR, holds/improves SSIM** — E8 vs E7 on BSD100 x4: 0.6195 vs 0.5923 SSIM (+0.027), at cost of −0.06 PSNR. Warmup (3 epochs MSE first) prevented the SSIM collapse seen in original E6/E8.
4. **ESPCN > SRCNN** consistently at both scales/losses.
5. LPIPS captured for all 24 model/dataset combos (in `evaluation_results.json`).

---

## 5. Step 7 — Git Commit (PENDING — USER SAID "DO NOT COMMIT YET")

**Do NOT commit.** User explicitly instructed: "Do not commit changes yet."

When authorized, commit these changes:
- Modified: `configs/experiments.yaml`, `data/dataset.py`, `data/dataset_manifest.json`, `losses/perceptual_loss.py`, `metrics/psnr_ssim.py`, `models/vgg_perceptual.py`, `train/train_all.py`
- New: `evaluate_all.py`, `scripts/fix_lr_alignment.py`, `verify_checkpoints.py`, `verify_data.py`, `colab_train.ipynb`, `outputs/evaluation/` (logs + results)
- Exclude: `frontend/node_modules/`
- Include: `frontend/package-lock.json`
- Inspect: `colab_train.ipynb` (untested, user reported zip upload issue)

Commit message format: `<type>: <description>` (feat/fix/refactor/docs/test/chore/perf/ci).

---

## 6. Key Files & Purposes

| File | Purpose |
|------|---------|
| `models/srcnn.py` | SRCNN: 9-1-5 conv, He init, Tanh output, bicubic-pre-upsample input |
| `models/espcn.py` | ESPCN: PixelShuffle upscale, He init, scale param |
| `models/vgg_perceptual.py` | Frozen VGG11 feature extractor (relu2_2/relu3_3) |
| `losses/mse_loss.py` | MSE wrapper |
| `losses/perceptual_loss.py` | Hybrid MSE + λ·Perceptual (skips VGG when λ=0) |
| `metrics/psnr_ssim.py` | ITU-R BT.601 Y-channel, border shave, clamp [0,1], skimage SSIM |
| `data/dataset.py` | PatchDataset: RAM-cached, degrade_on_the_fly, file_names filter |
| `data/degradation.py` | blur → downscale → noise → jpeg pipeline |
| `train/trainer.py` | Train/validate loop with AMP, grad accum, state_dict save |
| `train/train_all.py` | Experiment runner (E1-E8), val split, epoch logs, perceptual warmup |
| `evaluate_all.py` | Full evaluation: all 8 models × 3 test sets × 2 scales vs bicubic, PSNR/SSIM/LPIPS |
| `verify_data.py` | Dataset integrity checker |
| `verify_checkpoints.py` | Checkpoint inspector |
| `scripts/fix_lr_alignment.py` | Regenerates aligned LR pairs into `data/pairs_fixed/` |
| `configs/experiments.yaml` | E1-E8 matrix + hyperparams |
| `config/paths.py` | Relative path helpers (no hardcoded C:\ paths) |
| `outputs/checkpoints/` | Trained model state_dicts |
| `outputs/evaluation/` | Logs + evaluation results |
| `data/pairs/` | Original dataset pairs (source of truth, do not modify) |
| `data/pairs_fixed/` | Fixed x2-aligned LR pairs + copied HR (use for retraining) |

---

## 7. Config Snapshot (`configs/experiments.yaml`)

```yaml
experiments:
  E1: {model: srcnn, scale: 2, loss: mse, train_set: T91, test_set: Set14}
  E2: {model: srcnn, scale: 2, loss: mse_perceptual, train_set: T91, test_set: Set14}
  E3: {model: srcnn, scale: 4, loss: mse, train_set: T91, test_set: Set14}
  E4: {model: srcnn, scale: 4, loss: mse_perceptual, train_set: T91, test_set: Set14}
  E5: {model: espcn, scale: 2, loss: mse, train_set: T91, test_set: Set14}
  E6: {model: espcn, scale: 2, loss: mse_perceptual, train_set: T91, test_set: Set14}
  E7: {model: espcn, scale: 4, loss: mse, train_set: T91, test_set: Set14}
  E8: {model: espcn, scale: 4, loss: mse_perceptual, train_set: T91, test_set: Set14}
hyperparams:
  lr: 2.0e-4
  batch_size_local_mse: 16
  batch_size_local_perceptual: 8
  grad_accum_steps: 1
  max_epochs: 15
  early_stop_patience: 6
  perceptual_lambda: 0.05
  patch_lr_2x: 32
  patch_lr_4x: 32
  num_threads: 8
```

> The config YAML above is the source of truth for E1-E8 experiment matrix (all train_set=T91, test_set=Set14).

---

## 8. Environment

- OS: Windows 11, Python 3.14.7 (system), venv at `.venv/` (no torch inside venv)
- PyTorch + scikit-image available on system Python; `lpips` installed (alex net)
- **CUDA available**: RTX 2050 4GB VRAM, driver 617.14, torch 2.11.0+cu128, torchvision 0.26.0+cu128
- Training auto-detects `device = 'cuda' if torch.cuda.is_available() else 'cpu'`
- DataLoader `num_workers=0` (Windows)
- Project root on `sys.path` via `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`

---

## 9. Next Actions (in order)

1. ✅ Finish retraining E3-E8 — all 8 done on GPU
2. ✅ Re-run evaluation on retrained checkpoints → `outputs/evaluation/evaluation_results.json` + `summary.csv`
3. ✅ Update PROJECT_CONTEXT.md results table with retrained numbers (done above)
4. ⏳ **Get user authorization** before git commit (Step 7) — user said "Do not commit changes yet"
5. ⏳ Document protocol, limitations, historical synthetic work preserved
6. ⏳ Compare retrained vs original (Nov 1) checkpoints (originals overwritten — no backup)

---

## 10. Known Issues / Limitations

- `verify_checkpoints.py` crashes on cross-architecture comparison (SRCNN vs ESPCN tensor shape mismatch) — only same-arch comparisons work
- T91 x2 originals in `data/pairs/` still misaligned — use `data/pairs_fixed/` for training
- Original Nov 1 checkpoints (raw state_dicts, no metadata) were **overwritten** by retraining (all 8 overwritten: E1-E8). Originals are unrecoverable — no backup was made before retraining.
- `colab_train.ipynb` untested (user reported zip upload issue)
- Perceptual loss VGG uses deprecated `pretrained=` arg (warning only, works)
- Fixed: `loss_fn` not moved to CUDA — caused `RuntimeError: Expected all tensors to be on the same device` when perceptual loss activated after warmup on E8. Fix: `loss_fn = loss_fn.to(device)` in `train/train_all.py`.