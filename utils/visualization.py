"""Visualization utilities for SISR training and results.

- save_training_curves: plot loss / PSNR / SSIM over epochs and save to disk.
- save_sr_grid: side-by-side LR / bicubic / SR / HR comparison saved as one image.
- save_image: save a single [0,1] CHW tensor as a PNG.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_image(tensor, path):
    """Save a (3,H,W) float tensor in [0,1] as a PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = TF.to_pil_image(tensor.clamp(0, 1).cpu())
    img.save(path)


def save_grid(*images, path='grid.png', titles=None, scale=1):
    """Save a side-by-side grid of images.

    Each image may be a PIL Image, a (3,H,W) tensor in [0,1], or a numpy HWC
    uint8 array. Optional `titles` labels each column.
    """
    pil_images = []
    for img in images:
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img.clamp(0, 1).cpu())
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        pil_images.append(img)

    # Ensure matching heights (resize all to the tallest).
    max_h = max(im.height for im in pil_images)
    pil_images = [
        im.resize((int(im.width * max_h / im.height), max_h), Image.BICUBIC)
        for im in pil_images
    ]

    total_w = sum(im.width for im in pil_images)
    grid = Image.new('RGB', (total_w, max_h), color='black')
    x = 0
    for im in pil_images:
        grid.paste(im, (x, 0))
        x += im.width

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)
    return grid


def save_training_curves(history, path='outputs/training_curves.png'):
    """Plot loss, PSNR, and SSIM over epochs from a history dict.

    history: dict with optional keys 'loss', 'psnr', 'ssim' -> lists of per-epoch values.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [('loss', 'Train Loss'), ('psnr', 'Val PSNR (dB)'), ('ssim', 'Val SSIM')]
    for ax, (key, label) in zip(axes, metrics):
        values = history.get(key)
        if not values:
            ax.set_visible(False)
            continue
        ax.plot(values, marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_sr_comparison(lr, bicubic, sr, hr, path='outputs/sr_comparison.png'):
    """Save an LR / bicubic / SR / HR comparison grid."""
    return save_grid(lr, bicubic, sr, hr, path=path,
                     titles=['LR', 'Bicubic', 'SR', 'HR'])
