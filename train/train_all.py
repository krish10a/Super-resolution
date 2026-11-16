import argparse
import os
import sys
import random
import yaml
import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path when run as a script (python train/train_all.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cpu_affinity import set_cpu_threads
from models.srcnn import SRCNN
from models.espcn import ESPCN
from losses.mse_loss import MSELoss
from losses.perceptual_loss import HybridLoss
from train.trainer import Trainer
from data.dataset import PatchDataset, ManifestPatchDataset, FullImageValDataset
from metrics.psnr_ssim import compute_psnr_ssim
from PIL import Image
import torchvision.transforms.functional as TTF
import numpy as np


def patch_size_for_scale(scale):
    return 32 if scale == 2 else 48


def bicubic_upsample(lr_img, scale):
    """Upsample a PIL LR image by bicubic to HR size (SRCNN-style input)."""
    w, h = lr_img.size
    return lr_img.resize((w * scale, h * scale), Image.BICUBIC)


@torch.no_grad()
def bicubic_baseline(lr_dir, hr_dir, scale, device):
    """PSNR/SSIM of plain bicubic upsampling on every LR/HR pair."""
    import os as _os
    files = [f for f in _os.listdir(lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        return float('nan'), float('nan')
    psnrs, ssims = [], []
    for f in files:
        name = _os.path.splitext(f)[0]
        lr_path = _os.path.join(lr_dir, f)
        hr_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            cand = _os.path.join(hr_dir, name + ext)
            if _os.path.exists(cand):
                hr_path = cand
                break
        if hr_path is None:
            continue
        lr = Image.open(lr_path).convert('RGB')
        hr = Image.open(hr_path).convert('RGB')
        # bicubic upsample LR to exactly HR size
        lr_up = lr.resize(hr.size, Image.BICUBIC)
        lr_t = TTF.to_tensor(lr_up).unsqueeze(0).to(device)
        hr_t = TTF.to_tensor(hr).unsqueeze(0).to(device)
        p, s = compute_psnr_ssim(lr_t.squeeze(0), hr_t.squeeze(0), scale=scale)
        psnrs.append(p)
        ssims.append(s)
    return (sum(psnrs) / len(psnrs) if psnrs else float('nan'),
            sum(ssims) / len(ssims) if ssims else float('nan'))


def split_train_val(hr_dir, val_split, seed=42):
    """Split HR files in hr_dir into train/val basename sets."""
    all_files = sorted([f for f in os.listdir(hr_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    random.seed(seed)
    random.shuffle(all_files)
    n_val = max(1, int(len(all_files) * val_split))
    val_names = {os.path.splitext(f)[0] for f in all_files[:n_val]}
    train_names = {os.path.splitext(f)[0] for f in all_files[n_val:]}
    return train_names, val_names


def main():
    parser = argparse.ArgumentParser(description='Train SISR experiment (E1-E8)')
    parser.add_argument('--config', required=True, help='Path to experiments YAML')
    parser.add_argument('--exp', required=True, help='Experiment id, e.g. E4')
    parser.add_argument('--data-dir', default='data/pairs',
                        help='Root directory containing dataset pairs (default: data/pairs)')
    parser.add_argument('--legacy-data', action='store_true',
                        help='Use legacy directory pairs instead of canonical manifest')
    parser.add_argument('--val-split', type=float, default=0.0,
                        help='Fraction of train_set HR images to hold out as validation (0.0 disables)')
    parser.add_argument('--log-file', default=None,
                        help='Append per-epoch logs to this file (default: none)')
    parser.add_argument('--perc-warmup-epochs', type=int, default=0,
                        help='For perceptual loss: MSE-only epochs before enabling hybrid (default: 0)')
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='Override max_epochs from config (default: use config value)')
    parser.add_argument('--patience', type=int, default=None,
                        help='Override early_stop_patience from config (default: use config value)')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='Disable early stopping and run the full max_epochs regardless of val PSNR')
    parser.add_argument('--ckpt-tag', default=None,
                        help='Optional tag appended to checkpoint filename (e.g. "verified") to avoid overwriting existing checkpoints')
    parser.add_argument('--degrade', action='store_true',
                        help='Generate LR from HR on the fly via degradation pipeline')
    parser.add_argument('--blur-ksize', type=int, default=5,
                        help='Gaussian blur kernel size for degradation')
    parser.add_argument('--blur-sigma', type=float, default=1.0,
                        help='Gaussian blur sigma for degradation')
    parser.add_argument('--noise-sigma', type=float, default=0.0,
                        help='Additive Gaussian noise sigma for degradation')
    parser.add_argument('--jpeg-quality', type=int, default=None,
                        help='JPEG quality for degradation (None disables)')
    args = parser.parse_args()

    set_cpu_threads()
    cfg = yaml.safe_load(open(args.config))['experiments'][args.exp]
    hp = yaml.safe_load(open(args.config)).get('hyperparams', {})

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    scale = cfg['scale']
    model_name = cfg['model']
    loss_name = cfg['loss']
    train_set = cfg['train_set']
    test_set = cfg['test_set']

    lr = hp.get('lr', 1e-4)
    batch_size = hp.get('batch_size_local_mse', 4)
    grad_accum = hp.get('grad_accum_steps', 4)
    max_epochs = args.max_epochs if args.max_epochs is not None else hp.get('max_epochs', 200)
    patience = args.patience if args.patience is not None else hp.get('early_stop_patience', 15)
    patch_size = hp.get(f'patch_lr_{scale}x', patch_size_for_scale(scale))
    perceptual_lambda = hp.get('perceptual_lambda', 0.07)

    degradation_kwargs = {
        'blur_ksize': args.blur_ksize,
        'blur_sigma': args.blur_sigma,
        'noise_sigma': args.noise_sigma,
        'jpeg_quality': args.jpeg_quality,
    }
    if args.degrade:
        print(f'Degradation ON: blur_ksize={args.blur_ksize} '
              f'blur_sigma={args.blur_sigma} noise_sigma={args.noise_sigma} '
              f'jpeg_quality={args.jpeg_quality}')

    # --- Model ---
    if model_name == 'srcnn':
        model = SRCNN()
        upsample_lr = True  # SRCNN expects bicubic-upsampled LR input
    else:
        model = ESPCN(scale=scale)
        upsample_lr = False  # ESPCN PixelShuffle upscales internally
    model = model.to(device)

    # --- Loss ---
    if loss_name in ('mse', 'mse_loss'):
        loss_fn = MSELoss()
    elif loss_name in ('mse_perceptual', 'mse+perceptual', 'hybrid'):
        # Start with lambda_perc=0 for warmup (MSE-only), enable after warmup epochs
        loss_fn = HybridLoss(lambda_perc=0.0 if args.perc_warmup_epochs > 0 else perceptual_lambda)
    else:
        raise ValueError(f'Unknown loss: {loss_name}')

    loss_fn = loss_fn.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Data ---
    manifest_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'dataset_manifest.json')
    if not args.legacy_data and os.path.exists(manifest_file):
        print(f"Loading canonical manifest-based data (Zero test-leakage mode)")
        train_ds = ManifestPatchDataset(manifest_path=manifest_file, dataset=train_set,
                                        split='train', scale=scale, patch_size=patch_size,
                                        upsample_lr=upsample_lr)
        val_ds = FullImageValDataset(manifest_path=manifest_file, dataset=train_set,
                                     split='val', scale=scale, upsample_lr=upsample_lr)
        print(f"Clean training split: {len(train_ds.entries)} train images, {len(val_ds.entries)} full val images (from {train_set})")
        val_lr, val_hr = None, None
    else:
        base = args.data_dir
        train_hr = os.path.join(base, f'{train_set}_HR')
        val_lr = os.path.join(base, f'{test_set}_LR_x{scale}')
        val_hr = os.path.join(base, f'{test_set}_HR')

        if args.degrade:
            # On-the-fly degradation: generate LR from HR during training.
            if not os.path.isdir(train_hr):
                raise FileNotFoundError(
                    f'Data dir missing: {train_hr}\n'
                    f'Run: python data/download_sets.py && python data/prepare_pairs.py'
                )
            if args.val_split > 0:
                train_names, val_names = split_train_val(train_hr, args.val_split)
                train_ds = PatchDataset(lr_dir=None, hr_dir=train_hr, scale=scale,
                                        patch_size=patch_size, upsample_lr=upsample_lr,
                                        degrade_on_the_fly=True,
                                        degradation_kwargs=degradation_kwargs,
                                        file_names=train_names)
                val_ds = PatchDataset(lr_dir=None, hr_dir=train_hr, scale=scale,
                                      patch_size=patch_size, upsample_lr=upsample_lr,
                                      degrade_on_the_fly=True,
                                      degradation_kwargs=degradation_kwargs,
                                      file_names=val_names)
                print(f'Train/val split: {len(train_names)} train / {len(val_names)} val images '
                      f'(from {train_set})')
            else:
                train_ds = PatchDataset(lr_dir=None, hr_dir=train_hr, scale=scale,
                                        patch_size=patch_size, upsample_lr=upsample_lr,
                                        degrade_on_the_fly=True,
                                        degradation_kwargs=degradation_kwargs)
                val_ds = PatchDataset(val_lr, val_hr, scale=scale, patch_size=patch_size,
                                      upsample_lr=upsample_lr)
                print(f'Training with on-the-fly degradation from HR dir: {train_hr}')
        else:
            train_lr = os.path.join(base, f'{train_set}_LR_x{scale}')
            for d in [train_lr, train_hr, val_lr, val_hr]:
                if not os.path.isdir(d):
                    raise FileNotFoundError(
                        f'Data dir missing: {d}\n'
                        f'Run: python data/download_sets.py && python data/prepare_pairs.py'
                    )
            if args.val_split > 0:
                train_names, val_names = split_train_val(train_hr, args.val_split)
                train_ds = PatchDataset(train_lr, train_hr, scale=scale,
                                        patch_size=patch_size, upsample_lr=upsample_lr,
                                        file_names=train_names)
                # Validation on held-out train_set images (LR/HR from train dirs)
                val_ds = PatchDataset(train_lr, train_hr, scale=scale,
                                      patch_size=patch_size, upsample_lr=upsample_lr,
                                      file_names=val_names)
                print(f'Train/val split: {len(train_names)} train / {len(val_names)} val images '
                      f'(from {train_set})')
            else:
                train_ds = PatchDataset(train_lr, train_hr, scale=scale,
                                        patch_size=patch_size, upsample_lr=upsample_lr)
                val_ds = PatchDataset(val_lr, val_hr, scale=scale, patch_size=patch_size,
                                      upsample_lr=upsample_lr)
                print(f'Cross-dataset validation on {test_set} ({len(os.listdir(val_hr))} images)')

    # Windows: num_workers=0, no pin_memory on CPU
    num_workers = 0
    pin_memory = (device == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    scaler = torch.amp.GradScaler('cuda') if (device == 'cuda') else None
    trainer = Trainer(model, loss_fn, optimizer, device, scaler=scaler)

    # --- Bicubic baseline ---
    if val_lr and val_hr:
        base_psnr, base_ssim = bicubic_baseline(val_lr, val_hr, scale, device)
        print(f'Bicubic baseline on {test_set} x{scale}: PSNR {base_psnr:.2f} SSIM {base_ssim:.4f}')
    else:
        # Compute baseline on held-out validation split
        val_psnrs, val_ssims = [], []
        for lr_t, hr_t, _ in val_ds:
            up_t = torch.nn.functional.interpolate(lr_t.unsqueeze(0), scale_factor=scale, mode='bicubic', align_corners=False).squeeze(0)
            p, s = compute_psnr_ssim(up_t, hr_t, scale=scale, y_channel=True)
            val_psnrs.append(p); val_ssims.append(s)
        base_psnr = float(np.mean(val_psnrs))
        base_ssim = float(np.mean(val_ssims))
        print(f'Bicubic baseline on {train_set} val split x{scale}: PSNR {base_psnr:.2f} SSIM {base_ssim:.4f}')

    # --- Epoch log file ---
    log_file = args.log_file
    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        with open(log_file, 'w') as f:
            f.write('epoch,train_loss,val_psnr,val_ssim,best_psnr\n')

    # --- Training loop ---
    best_psnr = float('-inf')
    ckpt_name = f'{model_name}_{scale}x_{loss_name}'
    if args.ckpt_tag:
        ckpt_name += f'_{args.ckpt_tag}'
    best_path = os.path.join('outputs', 'checkpoints', f'{ckpt_name}.pt')
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    patience_counter = 0
    for epoch in range(1, max_epochs + 1):
        # Curriculum: enable perceptual loss after warmup epochs
        if (isinstance(loss_fn, HybridLoss) and args.perc_warmup_epochs > 0
                and epoch == args.perc_warmup_epochs + 1):
            loss_fn.lambda_perc = perceptual_lambda
            print(f'  -> Enabling perceptual loss (lambda={perceptual_lambda}) after warmup')

        train_loss = trainer.train_one_epoch(train_loader, grad_accum=grad_accum)
        val_psnr, val_ssim = trainer.validate(val_loader, compute_psnr_ssim, scale=scale)

        print(f'Epoch {epoch:3d}/{max_epochs} '
              f'train_loss {train_loss:.6f} '
              f'val_PSNR {val_psnr:.2f} val_SSIM {val_ssim:.4f}')

        if log_file:
            with open(log_file, 'a') as f:
                f.write(f'{epoch},{train_loss:.6f},{val_psnr:.4f},{val_ssim:.4f},{best_psnr:.4f}\n')

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            trainer.save(best_path)
            patience_counter = 0
            print(f'  -> new best PSNR {best_psnr:.2f}, saved to {best_path}')
        else:
            patience_counter += 1

        if not args.no_early_stop and patience_counter >= patience:
            print(f'Early stopping at epoch {epoch} (no val PSNR improvement for {patience} epochs)')
            break

    print(f'\nExperiment {args.exp} complete: {model_name} x{scale} {loss_name}')
    print(f'Best val PSNR: {best_psnr:.2f} (bicubic baseline: {base_psnr:.2f})')
    print(f'Checkpoint: {best_path}')


if __name__ == '__main__':
    main()
