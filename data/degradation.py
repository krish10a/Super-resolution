"""On-the-fly LR degradation simulation for training SISR models.

Provides:
- downscale_bicubic: standard bicubic downsampling (HR -> LR)
- gaussian_blur: isotropic Gaussian blur
- gaussian_noise: additive Gaussian noise
- jpeg_compress: JPEG-quality compression artifact simulation
- degrade: compose blur + noise + downscale (the standard LR generation pipeline)

All functions operate on numpy HWC uint8 arrays in [0, 255] and return uint8
HWC arrays, so they can be chained and applied to PIL/OpenCV images alike.
"""

import cv2
import numpy as np


def downscale_bicubic(img, scale):
    """Downsample HR image to LR by factor `scale` using bicubic interpolation."""
    h, w = img.shape[:2]
    lr = cv2.resize(img, (max(1, w // scale), max(1, h // scale)),
                    interpolation=cv2.INTER_CUBIC)
    return lr


def gaussian_blur(img, ksize=5, sigma=1.0):
    """Apply isotropic Gaussian blur."""
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def gaussian_noise(img, sigma=5.0):
    """Add zero-mean Gaussian noise with std `sigma` (clipped to [0, 255])."""
    noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def jpeg_compress(img, quality=75):
    """Simulate JPEG compression artifacts at the given quality (1-100)."""
    quality = max(1, min(100, int(quality)))
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def degrade(img, scale, blur_ksize=5, blur_sigma=1.0, noise_sigma=0.0,
            jpeg_quality=None):
    """Compose the standard LR generation pipeline: blur -> downscale -> noise -> jpeg.

    Parameters
    ----------
    img : np.ndarray (HWC, uint8, RGB)
        HR source image.
    scale : int
        Downsampling factor.
    blur_ksize, blur_sigma : Gaussian blur applied before downscaling
        (anti-aliasing, mimics real camera blur).
    noise_sigma : float
        Std of additive Gaussian noise (0 disables).
    jpeg_quality : int or None
        JPEG quality after downscaling (None disables).
    """
    out = img
    if blur_ksize and blur_ksize > 1:
        out = gaussian_blur(out, ksize=blur_ksize, sigma=blur_sigma)
    out = downscale_bicubic(out, scale)
    if noise_sigma and noise_sigma > 0:
        out = gaussian_noise(out, sigma=noise_sigma)
    if jpeg_quality is not None:
        out = jpeg_compress(out, quality=jpeg_quality)
    return out
