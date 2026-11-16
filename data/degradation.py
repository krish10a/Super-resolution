"""On-the-fly LR degradation simulation for training SISR models.

Provides:
- downscale_bicubic: standard bicubic downsampling (HR -> LR)
- gaussian_blur: isotropic Gaussian blur
- anisotropic_gaussian_blur: directional anisotropic Gaussian blur
- motion_blur: linear camera motion blur simulation
- gaussian_noise: additive zero-mean Gaussian noise (deterministic with seed)
- poisson_noise: shot noise / low-light Poisson noise simulation
- jpeg_compress: JPEG-quality compression artifact simulation
- degrade: composed modular degradation pipeline (blur -> downscale -> noise -> jpeg)

All functions operate on numpy HWC uint8 arrays in [0, 255] and return uint8
HWC arrays. Fully reproducible when random seed is provided.
"""

from typing import Optional, Union, Dict, Any
import cv2
import numpy as np


def downscale_bicubic(img: np.ndarray, scale: int) -> np.ndarray:
    """Downsample HR image to LR by factor `scale` using bicubic interpolation."""
    h, w = img.shape[:2]
    lr = cv2.resize(img, (max(1, w // scale), max(1, h // scale)),
                    interpolation=cv2.INTER_CUBIC)
    return lr


def gaussian_blur(img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Apply isotropic Gaussian blur."""
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def anisotropic_gaussian_blur(img: np.ndarray, ksize: int = 5,
                              sigma_x: float = 1.5, sigma_y: float = 0.5,
                              angle: float = 0.0) -> np.ndarray:
    """Apply directional anisotropic Gaussian blur."""
    if ksize % 2 == 0:
        ksize += 1
    rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    S = np.array([[sigma_x ** 2, 0], [0, sigma_y ** 2]])
    cov = R @ S @ R.T

    inv_cov = np.linalg.pinv(cov)
    half = ksize // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    coords = np.stack([x, y], axis=-1)
    kernel = np.exp(-0.5 * np.einsum('ijk,kl,ijl->ij', coords, inv_cov, coords))
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel.astype(np.float32))


def motion_blur(img: np.ndarray, size: int = 5, angle: float = 0.0) -> np.ndarray:
    """Apply linear motion blur simulating camera shake or fast target motion."""
    if size <= 1:
        return img
    k = np.zeros((size, size), dtype=np.float32)
    k[(size - 1) // 2, :] = np.ones(size, dtype=np.float32)
    M = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, M, (size, size))
    k_sum = k.sum()
    if k_sum > 0:
        k /= k_sum
    return cv2.filter2D(img, -1, k)


def gaussian_noise(img: np.ndarray, sigma: float = 5.0, seed: Optional[int] = None) -> np.ndarray:
    """Add zero-mean Gaussian noise with std `sigma` (clipped to [0, 255])."""
    if sigma <= 0:
        return img
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def poisson_noise(img: np.ndarray, scale: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
    """Add Poisson shot noise simulating low-light surveillance sensor conditions."""
    if scale <= 0:
        return img
    rng = np.random.default_rng(seed)
    # Scale image into photon count space, sample poisson, scale back
    vals = img.astype(np.float32) / 255.0
    vals = np.clip(vals, 1e-6, 1.0)
    peak = 255.0 / scale
    noisy = rng.poisson(vals * peak) / peak
    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)


def jpeg_compress(img: np.ndarray, quality: int = 75) -> np.ndarray:
    """Simulate JPEG compression artifacts at the given quality (1-100)."""
    quality = max(1, min(100, int(quality)))
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def degrade(img: np.ndarray, scale: int, blur_ksize: int = 5, blur_sigma: float = 1.0,
            noise_sigma: float = 0.0, jpeg_quality: Optional[int] = None,
            blur_type: str = "gaussian", motion_angle: float = 0.0,
            poisson_scale: float = 0.0, seed: Optional[int] = None) -> np.ndarray:
    """Compose the modular LR surveillance generation pipeline.

    Order: blur -> downscale -> noise -> jpeg compression.

    Parameters
    ----------
    img : np.ndarray (HWC, uint8, RGB)
        HR source image.
    scale : int
        Downsampling factor.
    blur_ksize, blur_sigma : Gaussian blur parameters applied before downscaling.
    noise_sigma : float
        Std of additive Gaussian noise (0 disables).
    jpeg_quality : int or None
        JPEG quality after downscaling (None disables).
    blur_type : str
        "gaussian", "motion", or "anisotropic".
    motion_angle : float
        Angle in degrees for motion or anisotropic blur.
    poisson_scale : float
        Scale for Poisson low-light noise (0 disables).
    seed : int or None
        Random seed for 100% reproducible degradation.
    """
    out = img

    # 1. Blur
    if blur_ksize and blur_ksize > 1:
        if blur_type == "motion":
            out = motion_blur(out, size=blur_ksize, angle=motion_angle)
        elif blur_type == "anisotropic":
            out = anisotropic_gaussian_blur(out, ksize=blur_ksize, sigma_x=blur_sigma,
                                            sigma_y=max(0.2, blur_sigma * 0.4), angle=motion_angle)
        else:
            out = gaussian_blur(out, ksize=blur_ksize, sigma=blur_sigma)

    # 2. Downscale
    out = downscale_bicubic(out, scale)

    # 3. Noise
    if noise_sigma and noise_sigma > 0:
        out = gaussian_noise(out, sigma=noise_sigma, seed=seed)
    if poisson_scale and poisson_scale > 0:
        out = poisson_noise(out, scale=poisson_scale, seed=seed)

    # 4. Compression
    if jpeg_quality is not None:
        out = jpeg_compress(out, quality=jpeg_quality)

    return out
