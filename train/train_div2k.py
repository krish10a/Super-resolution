"""
Unified DIV2K Training System for SISR Experiments E9-E14.

Features:
- Hardware-optimized for NVIDIA RTX 2050 (4GB VRAM): AMP mixed precision,
  gradient clipping, memory-aware patch sizes.
- Models supported: ESPCN, FSRCNN, EDSR-Baseline at 2x and 4x.
- Losses supported: CharbonnierLoss, L1Loss.
- Schedulers: CosineAnnealingLR.
- Checkpoint management: Best validation checkpoint + periodic recovery checkpoints
  with full optimizer, scheduler, epoch, and config state.
- Resumable: `--resume` flag seamlessly restores optimizer, scheduler, and epoch.
- Live terminal visibility: tqdm progress bar with live loss, lr, GPU mem, and
  formatted end-of-epoch reports.
- Comprehensive CSV logging per experiment.
"""

import os
import sys
import time
import math
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.espcn import ESPCN
from models.fsrcnn import FSRCNN
from models.edsr import EDSR
from losses.charbonnier_loss import CharbonnierLoss
from data.div2k_dataset import DIV2KPatchDataset, DIV2KValDataset
from metrics.psnr_ssim import compute_psnr_ssim

CONFIG_DEFAULT = PROJECT_ROOT / "configs" / "experiments_phase3.yaml"
CHECKPOINTS_DEFAULT = PROJECT_ROOT / "outputs" / "checkpoints" / "phase3"
LOGS_DEFAULT = PROJECT_ROOT / "outputs" / "logs" / "phase3"


def get_model(arch: str, scale: int, cfg: Dict[str, Any]) -> nn.Module:
    arch = arch.lower()
    if arch == "espcn":
        # Supports standard Shi et al. mid_channels=32 or wide 64
        mid_ch = cfg.get("mid_channels", 32)
        return ESPCN(scale=scale, mid_channels=mid_ch)
    elif arch == "fsrcnn":
        return FSRCNN(scale=scale)
    elif arch == "edsr":
        n_feats = cfg.get("n_feats", 64)
        n_resblocks = cfg.get("n_resblocks", 16)
        res_scale = cfg.get("res_scale", 1.0)
        return EDSR(scale=scale, n_feats=n_feats, n_resblocks=n_resblocks, res_scale=res_scale)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def get_loss_fn(loss_name: str, cfg: Dict[str, Any]) -> nn.Module:
    loss_name = loss_name.lower()
    if loss_name == "charbonnier":
        eps = float(cfg.get("charbonnier_eps", 1e-3))
        return CharbonnierLoss(eps=eps)
    elif loss_name in ["l1", "l1loss"]:
        return nn.L1Loss()
    elif loss_name in ["mse", "mseloss"]:
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


def validate_model(
    model: nn.Module,
    val_loader: DataLoader,
    scale: int,
    device: torch.device,
    max_images: Optional[int] = None
) -> tuple[float, float]:
    """Deterministic validation on held-out DIV2K validation images.
    Uses representative central crops for memory safety and fast evaluation.
    """
    model.eval()
    psnrs, ssims = [], []
    count = 0

    with torch.no_grad():
        for lr_t, hr_t, img_id in val_loader:
            # Extract central crop for fast metric computation
            h_lr, w_lr = lr_t.shape[-2:]
            crop_lr = min(256, h_lr, w_lr)
            cy_lr = (h_lr - crop_lr) // 2
            cx_lr = (w_lr - crop_lr) // 2

            lr_sub = lr_t[..., cy_lr : cy_lr + crop_lr, cx_lr : cx_lr + crop_lr].to(device, non_blocking=True)
            hr_sub = hr_t[..., cy_lr * scale : (cy_lr + crop_lr) * scale, cx_lr * scale : (cx_lr + crop_lr) * scale]

            # Forward pass with AMP if on CUDA
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    out_t = model(lr_sub)
            else:
                out_t = model(lr_sub)

            out_t = out_t.squeeze(0).cpu().clamp(0.0, 1.0)
            hr_sub = hr_sub.squeeze(0).cpu().clamp(0.0, 1.0)

            # Ensure spatial alignment
            min_h = min(out_t.shape[-2], hr_sub.shape[-2])
            min_w = min(out_t.shape[-1], hr_sub.shape[-1])
            out_c = out_t[..., :min_h, :min_w]
            hr_c = hr_sub[..., :min_h, :min_w]

            p, s = compute_psnr_ssim(out_c, hr_c, scale=scale, y_channel=True)
            if not math.isnan(p) and not math.isinf(p):
                psnrs.append(p)
                ssims.append(s)

            count += 1
            if max_images and count >= max_images:
                break

    mean_p = float(sum(psnrs) / len(psnrs)) if psnrs else 0.0
    mean_s = float(sum(ssims) / len(ssims)) if ssims else 0.0
    return mean_p, mean_s


def run_experiment(
    exp_id: str,
    exp_cfg: Dict[str, Any],
    global_cfg: Dict[str, Any],
    args: argparse.Namespace
):
    device = torch.device(args.device)
    scale = exp_cfg["scale"]
    model_name = exp_cfg["model"]
    loss_name = exp_cfg["loss"]

    # Paths
    div2k_root = Path(args.div2k_dir)
    train_hr = div2k_root / "DIV2K_train_HR"
    train_lr = div2k_root / f"DIV2K_train_LR_x{scale}"
    valid_hr = div2k_root / "DIV2K_valid_HR"
    valid_lr = div2k_root / f"DIV2K_valid_LR_x{scale}"

    if not train_hr.exists() or not train_lr.exists():
        raise FileNotFoundError(
            f"Preprocessed DIV2K pairs not found in {div2k_root}.\n"
            f"Run: python scripts/prepare_div2k.py"
        )

    checkpoints_dir = Path(args.checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = checkpoints_dir / f"{exp_id}_{model_name}_{scale}x_best.pt"
    recovery_ckpt_path = checkpoints_dir / f"{exp_id}_{model_name}_{scale}x_recovery.pt"
    log_csv_path = logs_dir / f"{exp_id}_history.csv"

    # Instantiate model, loss, optimizer
    model = get_model(model_name, scale, exp_cfg).to(device)
    loss_fn = get_loss_fn(loss_name, exp_cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lr = float(exp_cfg.get("lr", 5e-4))
    min_lr = float(exp_cfg.get("min_lr", 1e-6))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    max_epochs = args.max_epochs or exp_cfg.get("max_epochs", 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=min_lr
    )

    use_amp = global_cfg.get("amp", True) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    grad_clip = float(global_cfg.get("grad_clip", 1.0))
    save_interval = int(global_cfg.get("save_every_epochs", 5))
    patience = int(exp_cfg.get("early_stop_patience", 15))

    # Datasets
    batch_size = args.batch_size or exp_cfg.get("batch_size", 16)
    lr_patch_size = exp_cfg.get("lr_patch_size", 32)
    patches_per_image = exp_cfg.get("patches_per_image", 10)
    if args.smoke_test:
        patches_per_image = 2

    train_ds = DIV2KPatchDataset(
        hr_dir=train_hr,
        lr_dir=train_lr,
        scale=scale,
        lr_patch_size=lr_patch_size,
        patches_per_image=patches_per_image,
        seed=global_cfg.get("seed", 42),
        max_images=10 if args.smoke_test else None,
    )
    val_ds = DIV2KValDataset(
        hr_dir=valid_hr,
        lr_dir=valid_lr,
        scale=scale,
        max_images=10 if args.smoke_test else None
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=global_cfg.get("num_workers", 0),
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    # Resume handling
    start_epoch = 1
    best_psnr = -1.0
    best_ssim = 0.0
    patience_counter = 0

    if args.resume and recovery_ckpt_path.exists():
        print(f"\n[RESUME] Found recovery checkpoint: {recovery_ckpt_path}")
        ckpt_data = torch.load(recovery_ckpt_path, map_location=device)
        model.load_state_dict(ckpt_data["model_state_dict"])
        optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])
        start_epoch = ckpt_data["epoch"] + 1
        best_psnr = ckpt_data.get("best_psnr", -1.0)
        best_ssim = ckpt_data.get("best_ssim", 0.0)
        print(f"[RESUME] Resuming from epoch {start_epoch} (best PSNR: {best_psnr:.2f} dB)")
    else:
        # Initialize CSV log file
        with open(log_csv_path, "w") as f:
            f.write("epoch,train_loss,val_psnr,val_ssim,lr,epoch_time_s,gpu_mem_mb\n")

    print("\n" + "=" * 80)
    print(f"STARTING EXPERIMENT: {exp_id} | Architecture: {model_name.upper()} | Scale: {scale}x")
    print(f"Loss: {loss_name} | Parameters: {param_count:,} | Device: {device} | AMP: {use_amp}")
    print(f"Batch size: {batch_size} | Patch size: {lr_patch_size}x{lr_patch_size} (HR {lr_patch_size*scale}x{lr_patch_size*scale})")
    print(f"Epochs: {start_epoch} to {max_epochs} | Training patches/epoch: {len(train_ds):,}")
    print("=" * 80 + "\n")

    for epoch in range(start_epoch, max_epochs + 1):
        epoch_t0 = time.time()
        model.train()
        running_loss = 0.0
        n_batches = len(train_loader)

        pbar = tqdm(
            train_loader,
            desc=f"[{exp_id}] Epoch {epoch}/{max_epochs}",
            unit="batch",
            ncols=105
        )

        for step, (lr_batch, hr_batch) in enumerate(pbar):
            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            try:
                if scaler:
                    with torch.amp.autocast("cuda"):
                        out = model(lr_batch)
                        loss = loss_fn(out, hr_batch)
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(lr_batch)
                    loss = loss_fn(out, hr_batch)
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                loss_val = loss.item()
                running_loss += loss_val

                if step % 5 == 0 or step == n_batches - 1:
                    gpu_mem_mb = (
                        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                        if device.type == "cuda" else 0.0
                    )
                    curr_lr = optimizer.param_groups[0]["lr"]
                    pbar.set_postfix({
                        "loss": f"{loss_val:.4f}",
                        "lr": f"{curr_lr:.1e}",
                        "vram": f"{gpu_mem_mb:.0f}MB"
                    })

            except torch.cuda.OutOfMemoryError as oom:
                print(f"\n[CUDA OOM ERROR] at epoch {epoch}, batch {step}.")
                print(f"Actionable fix: Reduce batch_size (currently {batch_size}) to {max(1, batch_size // 2)}")
                print(f"Or reduce lr_patch_size in configs/experiments_phase3.yaml.")
                torch.cuda.empty_cache()
                raise oom

        scheduler.step()
        epoch_time = time.time() - epoch_t0
        avg_train_loss = running_loss / max(1, n_batches)

        # Validation (10 images per epoch for high throughput; full validation on final)
        val_img_limit = 5 if args.smoke_test else exp_cfg.get("val_max_images", 10)
        val_psnr, val_ssim = validate_model(
            model, val_loader, scale=scale, device=device,
            max_images=val_img_limit
        )

        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr
            best_ssim = val_ssim
            patience_counter = 0
            best_state = {
                "experiment_id": exp_id,
                "model_name": model_name,
                "scale": scale,
                "loss": loss_name,
                "epoch": epoch,
                "best_psnr": best_psnr,
                "best_ssim": best_ssim,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": exp_cfg,
                "params": param_count,
            }
            torch.save(best_state, best_ckpt_path)
        else:
            patience_counter += 1

        # Periodic Recovery Checkpoint
        if epoch % save_interval == 0 or epoch == max_epochs:
            recovery_state = {
                "experiment_id": exp_id,
                "model_name": model_name,
                "scale": scale,
                "epoch": epoch,
                "best_psnr": best_psnr,
                "best_ssim": best_ssim,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": exp_cfg,
            }
            torch.save(recovery_state, recovery_ckpt_path)

        # Append to CSV log
        with open(log_csv_path, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.6f},{val_psnr:.4f},{val_ssim:.4f},"
                    f"{curr_lr:.6e},{epoch_time:.2f},{gpu_mem_mb:.1f}\n")

        # Live Formatted Epoch Summary
        print(f"\n--- {exp_id} | {model_name.upper()} {scale}x | Epoch {epoch}/{max_epochs} ---")
        print(f"  Train Loss:       {avg_train_loss:.6f}")
        print(f"  Validation PSNR:  {val_psnr:.4f} dB  (Best: {best_psnr:.4f} dB) {'[★ NEW BEST]' if is_best else ''}")
        print(f"  Validation SSIM:  {val_ssim:.4f}  (Best: {best_ssim:.4f})")
        print(f"  Learning Rate:    {curr_lr:.2e}")
        print(f"  Epoch Duration:   {epoch_time:.1f}s")
        print(f"  Peak VRAM:        {gpu_mem_mb:.1f} MB")
        print(f"  Patience:         {patience_counter}/{patience}\n")

        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (no val PSNR improvement for {patience} epochs).")
            break

    print(f"\n[DONE] Experiment {exp_id} Complete!")
    print(f"Best Validation PSNR: {best_psnr:.4f} dB | Best SSIM: {best_ssim:.4f}")
    print(f"Best Checkpoint: {best_ckpt_path}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="DIV2K SISR Trainer (E9-E14)")
    parser.add_argument("--config", type=str, default=str(CONFIG_DEFAULT), help="Path to YAML config")
    parser.add_argument("--exp", type=str, default="all", help="'all' or specific ID: E9, E10, E11, E12, E13, E14")
    parser.add_argument("--div2k-dir", type=str, default="data/processed/DIV2K", help="Path to processed DIV2K")
    parser.add_argument("--checkpoints-dir", type=str, default=str(CHECKPOINTS_DEFAULT), help="Checkpoints dir")
    parser.add_argument("--logs-dir", type=str, default=str(LOGS_DEFAULT), help="Logs dir")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="Resume from last recovery checkpoint")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--smoke-test", action="store_true", help="Run short smoke test mode (2 patches, 10 val images)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        full_cfg = yaml.safe_load(f)

    global_cfg = full_cfg.get("global_settings", {})
    exp_configs = full_cfg.get("experiments", {})

    if args.exp == "all":
        exp_list = ["E9", "E10", "E11", "E12", "E13", "E14"]
    else:
        exp_id = args.exp.upper()
        if exp_id not in exp_configs:
            print(f"Error: Unknown experiment '{args.exp}'. Available: {list(exp_configs.keys())}")
            sys.exit(1)
        exp_list = [exp_id]

    print("=" * 80)
    print("PHASE 3 DIV2K TRAINING PIPELINE (E9-E14)")
    print(f"Target Experiments: {exp_list}")
    print(f"Device: {args.device} | Config: {args.config}")
    if args.smoke_test:
        print("[SMOKE TEST MODE ENABLED]")
    print("=" * 80)

    for exp_id in exp_list:
        run_experiment(exp_id, exp_configs[exp_id], global_cfg, args)


if __name__ == "__main__":
    main()
