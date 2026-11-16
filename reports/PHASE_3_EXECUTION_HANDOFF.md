# PHASE 3 EXECUTION HANDOFF: DIV2K PREPROCESSING & MODEL TRAINING (E9–E14)

**Host Platform:** Windows 11 (PowerShell)  
**Target Directory:** `C:\Users\ASUS\Desktop\cv`  
**GPU:** NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM)  
**Python Environment:** Python 3.14.7 with PyTorch 2.11.0+cu128 (CUDA 12.8)  

---

## 1. Prerequisites & Environment Check

In your local terminal, verify your environment:
```powershell
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
*Expected Output:*
`PyTorch: 2.11.0+cu128 | CUDA available: True | Device: NVIDIA GeForce RTX 2050`

---

## 2. Step 1: Preprocess DIV2K (Generate LR/HR Pairs)

Your Kaggle download in `datasets/Div2k dataset/` contains 800 training HR images and 100 validation HR images. Run the preprocessing script in your foreground terminal to modcrop and generate MATLAB-compatible $2\times$ and $4\times$ downsampled pairs:

```powershell
python scripts/prepare_div2k.py
```

### What You Will See:
- Continuous `tqdm` progress bars with image filenames, processing speed, and ETA:
  ```
  Processing DIV2K_train:  45%|█████████        | 360/800 [04:12<05:08, 1.42img/s, img=0360]
  Processing DIV2K_valid:  10%|██               | 10/100  [00:07<01:03, 1.41img/s, img=0810]
  ```
- **Execution Time:** ~10–12 minutes total for all 900 images.
- **Output Generated:**
  - `data/processed/DIV2K/DIV2K_train_HR/` (800 images)
  - `data/processed/DIV2K/DIV2K_train_LR_x2/` (800 images)
  - `data/processed/DIV2K/DIV2K_train_LR_x4/` (800 images)
  - `data/processed/DIV2K/DIV2K_valid_HR/` (100 images)
  - `data/processed/DIV2K/DIV2K_valid_LR_x2/` (100 images)
  - `data/processed/DIV2K/DIV2K_valid_LR_x4/` (100 images)
  - `data/processed/DIV2K/div2k_manifest.json`

---

## 3. Step 2: Training Individual Experiments (Recommended Order)

Each experiment is fully self-contained. You can train them individually or run them sequentially.

### Experiment E9: ESPCN $2\times$ (Charbonnier Loss)
Lightweight sub-pixel CNN ($26\text{k}$ params), fast convergence:
```powershell
python train/train_div2k.py --exp E9
```
- **Estimated Duration:** ~15–20 minutes on RTX 2050.
- **Expected VRAM Usage:** $\sim 250\text{ MB}$.
- **Checkpoint Saved:** `outputs/checkpoints/phase3/E9_espcn_2x_best.pt`

### Experiment E10: FSRCNN $2\times$ (Charbonnier Loss)
Fast super-resolution CNN with deconvolution ($24.6\text{k}$ params):
```powershell
python train/train_div2k.py --exp E10
```
- **Estimated Duration:** ~15–20 minutes on RTX 2050.
- **Expected VRAM Usage:** $\sim 250\text{ MB}$.
- **Checkpoint Saved:** `outputs/checkpoints/phase3/E10_fsrcnn_2x_best.pt`

### Experiment E11: EDSR-Baseline $2\times$ (L1 Loss)
Deep residual network ($1.37\text{M}$ params), high reconstruction fidelity:
```powershell
python train/train_div2k.py --exp E11
```
- **Estimated Duration:** ~45–60 minutes on RTX 2050.
- **Expected VRAM Usage:** $\sim 1.4\text{ GB}$ (batch size 8, AMP enabled).
- **Checkpoint Saved:** `outputs/checkpoints/phase3/E11_edsr_2x_best.pt`

### Experiment E12: ESPCN $4\times$ (Charbonnier Loss)
```powershell
python train/train_div2k.py --exp E12
```
- **Estimated Duration:** ~18–25 minutes.
- **Checkpoint Saved:** `outputs/checkpoints/phase3/E12_espcn_4x_best.pt`

### Experiment E13: FSRCNN $4\times$ (Charbonnier Loss)
```powershell
python train/train_div2k.py --exp E13
```
- **Estimated Duration:** ~18–25 minutes.
- **Checkpoint Saved:** `outputs/checkpoints/phase3/E13_fsrcnn_4x_best.pt`

### Experiment E14: EDSR-Baseline $4\times$ (L1 Loss)
```powershell
python train/train_div2k.py --exp E14
```
- **Estimated Duration:** ~50–65 minutes.
- **Expected VRAM Usage:** $\sim 1.5\text{ GB}$.
- **Checkpoint Saved:** `outputs/checkpoints/phase3/E14_edsr_4x_best.pt`

---

## 4. Alternative: Sequential Execution of All Six Models

If you want to train all six models back-to-back unattended:
```powershell
python train/train_div2k.py --exp all
```
*Total Estimated Execution Time: ~3 to 3.5 hours on RTX 2050.*

---

## 5. How to Stop and Resume Safely

- **Stopping**: Press `Ctrl + C` in the terminal. The training script will exit safely.
- **Resuming**: Recovery checkpoints (`{exp_id}_{model}_{scale}x_recovery.pt`) are automatically saved every 5 epochs. To resume where you left off, add `--resume`:
  ```powershell
  python train/train_div2k.py --exp E11 --resume
  ```
  The script will detect the recovery checkpoint, restore model weights, optimizer state, learning rate schedule, and epoch index, and continue training seamlessly.

---

## 6. What Output Files to Return for Phase 4

After training completes, return the following to proceed to Phase 4:
1. `outputs/logs/phase3/` (contains `{exp_id}_history.csv` for each experiment).
2. Confirmation that checkpoints exist in `outputs/checkpoints/phase3/`:
   - `E9_espcn_2x_best.pt`
   - `E10_fsrcnn_2x_best.pt`
   - `E11_edsr_2x_best.pt`
   - `E12_espcn_4x_best.pt`
   - `E13_fsrcnn_4x_best.pt`
   - `E14_edsr_4x_best.pt`

Phase 4 will then independently evaluate all six new models on canonical Set5, Set14, and BSD100 benchmarks against the Bicubic baselines, evaluate surveillance degradation robustness, and integrate the models into the Forensic Studio.
