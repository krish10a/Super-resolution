import argparse
import os
import sys
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
from data.dataset import PatchDataset
from metrics.psnr_ssim import compute_psnr_ssim, rgb_to_y
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
        # bicubic upsample LR to HR size
        lr_up = bicubic_upsample(lr, scale)
        lr_t = TTF.to_tensor(lr_up).unsqueeze(0).to(device)
        hr_t = TTF.to_tensor(hr).unsqueeze(0).to(device)
        p, s = compute_psnr_ssim(lr_t.squeeze(0), hr_t.squeeze(0), scale=scale)
        psnrs.append(p)
        ssims.append(s)
    return (sum(psnrs) / len(psnrs) if psnrs else float('nan'),
            sum(ssims) / len(ssims) if ssims else float('nan'))


def main():
    parser = argparse.ArgumentParser(description='Train SISR experiment (E1-E8)')
    parser.add_argument('--config', required=True, help='Path to experiments YAML')
    parser.add_argument('--exp', required=True, help='Experiment id, e.g. E4')
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
    max_epochs = hp.get('max_epochs', 200)
    patience = hp.get('early_stop_patience', 15)
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
        loss_fn = HybridLoss(lambda_perc=perceptual_lambda)
    else:
        raise ValueError(f'Unknown loss: {loss_name}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Data ---
    base = 'data/pairs'
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
        train_ds = PatchDataset(lr_dir=None, hr_dir=train_hr, scale=scale,
                                patch_size=patch_size, upsample_lr=upsample_lr,
                                degrade_on_the_fly=True,
                                degradation_kwargs=degradation_kwargs)
        print(f'Training with on-the-fly degradation from HR dir: {train_hr}')
    else:
        train_lr = os.path.join(base, f'{train_set}_LR_x{scale}')
        for d in [train_lr, train_hr, val_lr, val_hr]:
            if not os.path.isdir(d):
                raise FileNotFoundError(
                    f'Data dir missing: {d}\n'
                    f'Run: python data/download_sets.py && python data/prepare_pairs.py'
                )
        train_ds = PatchDataset(train_lr, train_hr, scale=scale,
                                patch_size=patch_size, upsample_lr=upsample_lr)

    val_ds = PatchDataset(val_lr, val_hr, scale=scale, patch_size=patch_size,
                          upsample_lr=upsample_lr)

    # Windows: num_workers=0, no pin_memory on CPU
    num_workers = 0
    pin_memory = (device == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    scaler = torch.cuda.amp.GradScaler() if (device == 'cuda') else None
    trainer = Trainer(model, loss_fn, optimizer, device, scaler=scaler)

    # --- Bicubic baseline ---
    base_psnr, base_ssim = bicubic_baseline(val_lr, val_hr, scale, device)
    print(f'Bicubic baseline on {test_set} x{scale}: PSNR {base_psnr:.2f} SSIM {base_ssim:.4f}')

    # --- Training loop ---
    best_psnr = float('-inf')
    best_path = os.path.join('outputs', 'checkpoints', f'{model_name}_{scale}x_{loss_name}.pt')
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    patience_counter = 0
    for epoch in range(1, max_epochs + 1):
        train_loss = trainer.train_one_epoch(train_loader, grad_accum=grad_accum)
        val_psnr, val_ssim = trainer.validate(val_loader, compute_psnr_ssim, scale=scale)

        print(f'Epoch {epoch:3d}/{max_epochs} '
              f'train_loss {train_loss:.6f} '
              f'val_PSNR {val_psnr:.2f} val_SSIM {val_ssim:.4f}')

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            trainer.save(best_path)
            patience_counter = 0
            print(f'  -> new best PSNR {best_psnr:.2f}, saved to {best_path}')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch} (no val PSNR improvement for {patience} epochs)')
            break

    print(f'\nExperiment {args.exp} complete: {model_name} x{scale} {loss_name}')
    print(f'Best val PSNR: {best_psnr:.2f} (bicubic baseline: {base_psnr:.2f})')
    print(f'Checkpoint: {best_path}')


if __name__ == '__main__':
    main()
