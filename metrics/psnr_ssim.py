import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ski_ssim


def rgb_to_y_cbcr(img):
    """
    Standard ITU-R BT.601 conversion.
    Input img is torch.Tensor in [0, 1] range of shape (C, H, W) or (B, C, H, W).
    Returns Y channel in [0, 1] range (or 16/255 to 235/255 standard scaling).
    Official SR benchmark formula:
    Y = 16.0 / 255.0 + (65.481 * R + 128.553 * G + 24.966 * B) / 255.0
    """
    if img.dim() == 4:
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        y = 16.0 / 255.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
        return y.squeeze(1)
    elif img.dim() == 3:
        r, g, b = img[0:1], img[1:2], img[2:3]
        y = 16.0 / 255.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
        return y.squeeze(0)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {img.shape}")


def psnr(pred, target, max_val=1.0):
    """Compute PSNR between two tensors in [0, 1] range."""
    mse = F.mse_loss(pred, target)
    if mse < 1e-12:
        return 100.0
    return 20.0 * np.log10(max_val) - 10.0 * torch.log10(mse).item()


def compute_psnr_ssim(pred, target, scale=1, y_channel=True):
    """
    Standard SISR evaluation.
    - Shaves border by `scale` pixels unconditionally.
    - Clamps output to [0, 1].
    - Converts RGB to ITU-R BT.601 Y channel.
    - Calculates PSNR and SSIM.
    """
    pred = pred.detach().cpu().float().clamp(0.0, 1.0)
    target = target.detach().cpu().float().clamp(0.0, 1.0)

    # Shave borders unconditionally
    if scale > 0:
        if pred.dim() == 3:
            pred = pred[:, scale:-scale, scale:-scale]
            target = target[:, scale:-scale, scale:-scale]
        elif pred.dim() == 4:
            pred = pred[:, :, scale:-scale, scale:-scale]
            target = target[:, :, scale:-scale, scale:-scale]

    if y_channel and pred.shape[0 if pred.dim() == 3 else 1] == 3:
        y_pred = rgb_to_y_cbcr(pred)
        y_tar = rgb_to_y_cbcr(target)
    else:
        y_pred = pred
        y_tar = target

    # PSNR
    p = psnr(y_pred, y_tar, max_val=1.0)

    # SSIM via skimage
    y_pred_np = y_pred.squeeze().numpy()
    y_tar_np = y_tar.squeeze().numpy()

    if y_pred_np.ndim == 2:
        s = ski_ssim(y_pred_np, y_tar_np, data_range=1.0)
    else:
        # Multi-sample average
        s = float(np.mean([
            ski_ssim(yp, yt, data_range=1.0)
            for yp, yt in zip(y_pred_np, y_tar_np)
        ]))

    return float(p), float(s)
