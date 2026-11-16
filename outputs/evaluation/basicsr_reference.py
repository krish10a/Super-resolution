"""
Vendored BasicSR reference implementation for numerical parity testing.

Source: https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/matlab_functions.py

This is an independent reference implementation of:
- imresize (MATLAB-compatible bicubic)
- calculate_psnr / calculate_ssim
- rgb2ycbcr / bgr2ycbcr

Used ONLY for parity comparison with our custom implementation.
"""

import math
import numpy as np
import cv2
import torch


# ============================================================================
# BasicSR color_util.py - rgb2ycbcr / bgr2ycbcr
# ============================================================================

def rgb2ycbcr(img, y_only=False):
    """Convert RGB image to YCbCr (BasicSR reference).

    Input: np.uint8 [0,255] or np.float32 [0,1], HWC, RGB order.
    Output: same dtype/range as input.
    """
    in_img_type = img.dtype
    img = img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.0

    if y_only:
        out = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
    else:
        out = np.matmul(
            img,
            [[65.481, -37.797, 112.0],
             [128.553, -74.203, -93.786],
             [24.966, 112.0, -18.214]]
        ) + [16, 128, 128]

    out = out / 255.0
    out = out.astype(in_img_type)
    return out


def bgr2ycbcr(img, y_only=False):
    """Convert BGR image to YCbCr (BasicSR reference)."""
    in_img_type = img.dtype
    img = img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.0

    if y_only:
        out = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out = np.matmul(
            img,
            [[24.966, 112.0, -18.214],
             [128.553, -74.203, -93.786],
             [65.481, -37.797, 112.0]]
        ) + [16, 128, 128]

    out = out / 255.0
    out = out.astype(in_img_type)
    return out


# ============================================================================
# BasicSR matlab_functions.py - cubic, calculate_weights_indices, imresize
# ============================================================================

def _basicSR_cubic(x):
    """BasicSR's cubic convolution kernel."""
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3

    w = np.zeros_like(absx, dtype=np.float64)
    mask1 = absx <= 1.0
    w[mask1] = 1.5 * absx3[mask1] - 2.5 * absx2[mask1] + 1.0
    mask2 = (absx > 1.0) & (absx <= 2.0)
    w[mask2] = -0.5 * absx3[mask2] + 2.5 * absx2[mask2] - 4.0 * absx[mask2] + 2.0
    return w


def _basicSR_calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    """BasicSR's calculate_weights_indices port."""
    if scale < 1 and antialiasing:
        kernel_width = kernel_width / scale

    # BasicSR: x runs from 1 to out_length (1-based)
    # u = x / scale + 0.5 * (1 - 1/scale)
    # 0-based: center_j = (j+1)/scale + 0.5*(1-1/scale) - 1 = (j+0.5)/scale - 0.5
    def center_fn(j):
        return (j + 0.5) / scale - 0.5

    indices = []
    weights = []

    for j in range(out_length):
        center = center_fn(j)
        left = int(math.floor(center - kernel_width / 2))
        right = int(math.ceil(center + kernel_width / 2))

        idx = np.arange(left, right + 1, dtype=np.int64)
        dist = (idx.astype(np.float64) - center)

        if scale < 1 and antialiasing:
            w = kernel(dist * scale) * scale
        else:
            w = kernel(dist)

        w_sum = np.sum(w)
        if w_sum > 0:
            w = w / w_sum

        indices.append(idx)
        weights.append(w)

    return indices, weights


def basicSR_imresize(img, scale, antialiasing=True):
    """BasicSR's imresize port.

    Parameters
    ----------
    img : np.ndarray
        HWC array in [0,1] (float) or [0,255] (uint8), RGB order.
    scale : float
        Scale factor.
    antialiasing : bool
        Apply antialiasing when downsampling.

    Returns
    -------
    np.ndarray
        Resized HWC array with same dtype/range as input.
    """
    is_uint8 = img.dtype == np.uint8
    img_work = img.astype(np.float64)
    if is_uint8:
        img_work = img_work / 255.0

    h, w, c = img_work.shape

    # BasicSR output dimensions: ceil(in * scale)
    out_h = int(math.ceil(h * scale))
    out_w = int(math.ceil(w * scale))
    out_h = max(out_h, 1)
    out_w = max(out_w, 1)

    kernel_width = 4.0

    out_w_indices, out_w_weights = _basicSR_calculate_weights_indices(
        w, out_w, scale, _basicSR_cubic, kernel_width, antialiasing)
    out_h_indices, out_h_weights = _basicSR_calculate_weights_indices(
        h, out_h, scale, _basicSR_cubic, kernel_width, antialiasing)

    # Horizontal padding
    pad_w_left = max(0, -int(np.min([idx.min() for idx in out_w_indices])))
    pad_w_right = max(0, int(np.max([idx.max() for idx in out_w_indices])) - (w - 1))

    if pad_w_left > 0 or pad_w_right > 0:
        img_padded_w = np.pad(
            img_work,
            ((0, 0), (pad_w_left, pad_w_right), (0, 0)),
            mode='edge',
        )
    else:
        img_padded_w = img_work

    out_w_indices_p = [idx + pad_w_left for idx in out_w_indices]

    temp = np.zeros((h, out_w, c), dtype=np.float64)
    for j in range(out_w):
        idx = out_w_indices_p[j]
        wt = out_w_weights[j]
        temp[:, j, :] = np.sum(img_padded_w[:, idx, :] * wt[None, :, None], axis=1)

    # Vertical padding
    pad_h_left = max(0, -int(np.min([idx.min() for idx in out_h_indices])))
    pad_h_right = max(0, int(np.max([idx.max() for idx in out_h_indices])) - (h - 1))

    if pad_h_left > 0 or pad_h_right > 0:
        temp_padded_h = np.pad(
            temp,
            ((pad_h_left, pad_h_right), (0, 0), (0, 0)),
            mode='edge',
        )
    else:
        temp_padded_h = temp

    out_h_indices_p = [idx + pad_h_left for idx in out_h_indices]

    out = np.zeros((out_h, out_w, c), dtype=np.float64)
    for i in range(out_h):
        idx = out_h_indices_p[i]
        wt = out_h_weights[i]
        out[i, :, :] = np.sum(temp_padded_h[idx, :, :] * wt[:, None, None], axis=0)

    if is_uint8:
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    else:
        out = np.clip(out, 0.0, 1.0)

    return out


# ============================================================================
# BasicSR metrics/psnr_ssim.py - calculate_psnr, calculate_ssim
# ============================================================================

def _basicSR_reorder_image(img, input_order='HWC'):
    """Reorder image to HWC format."""
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported are 'HWC' and 'CHW'")
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def _basicSR_to_y_channel(img):
    """Convert image to Y channel using BT.601 (BasicSR)."""
    # img is HWC float [0,1] or uint8 [0,255], RGB order
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type != np.uint8:
        img = img * 255.0

    y = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
    y = y / 255.0
    y = y.astype(img_type)
    return y


def basicSR_calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False):
    """BasicSR's calculate_psnr (NumPy version)."""
    assert img.shape == img2.shape, f"Shape mismatch: {img.shape} vs {img2.shape}"

    img = _basicSR_reorder_image(img, input_order)
    img2 = _basicSR_reorder_image(img2, input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = _basicSR_to_y_channel(img)
        img2 = _basicSR_to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img - img2) ** 2)
    if mse == 0:
        return float('inf')
    return float(10.0 * np.log10(255.0 * 255.0 / mse))


def basicSR_calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False):
    """BasicSR's calculate_ssim (NumPy version)."""
    assert img.shape == img2.shape, f"Shape mismatch: {img.shape} vs {img2.shape}"

    img = _basicSR_reorder_image(img, input_order)
    img2 = _basicSR_reorder_image(img2, input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = _basicSR_to_y_channel(img)
        img2 = _basicSR_to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssims = []
    for i in range(img.shape[2] if img.ndim == 3 else 1):
        if img.ndim == 3:
            ssims.append(_basicSR_ssim(img[:, :, i], img2[:, :, i]))
        else:
            ssims.append(_basicSR_ssim(img, img2))
    return float(np.mean(ssims))


def _basicSR_ssim(img, img2):
    """BasicSR's _ssim helper (single channel)."""
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    # 11x11 Gaussian window, sigma=1.5
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
        ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    return float(np.mean(ssim_map))


# ============================================================================
# PyTorch versions (matching BasicSR calculate_psnr_pt / calculate_ssim_pt)
# ============================================================================

def rgb2ycbcr_pt(img, y_only=False):
    """BasicSR's rgb2ycbcr_pt for PyTorch tensors (n, 3, h, w) in [0,1], RGB."""
    # img: (n, 3, h, w) RGB [0,1]
    # Permute to (n, h, w, 3), apply weights, permute back
    img = img.permute(0, 2, 3, 1)  # (n, h, w, 3)
    if y_only:
        out = torch.matmul(img, torch.tensor([65.481, 128.553, 24.966], device=img.device, dtype=img.dtype)) + 16.0
    else:
        weights = torch.tensor(
            [[65.481, -37.797, 112.0],
             [128.553, -74.203, -93.786],
             [24.966, 112.0, -18.214]],
            device=img.device, dtype=img.dtype
        )
        out = torch.matmul(img, weights) + torch.tensor([16, 128, 128], device=img.device, dtype=img.dtype)
    out = out / 255.0
    out = out.permute(0, 3, 1, 2)  # (n, 3/c, h, w)
    return out


def basicSR_calculate_psnr_pt(img, img2, crop_border, test_y_channel=False):
    """BasicSR's calculate_psnr_pt for PyTorch tensors (n, 3/1, h, w) in [0,1]."""
    assert img.shape == img2.shape

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2) ** 2, dim=[1, 2, 3])
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
    return psnr


def basicSR_calculate_ssim_pt(img, img2, crop_border, test_y_channel=False):
    """BasicSR's calculate_ssim_pt for PyTorch tensors."""
    assert img.shape == img2.shape

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    return _basicSR_ssim_pth(img * 255.0, img2 * 255.0)


def _basicSR_ssim_pth(img, img2):
    """BasicSR's _ssim_pth for PyTorch tensors (n, c, h, w) in [0,255]."""
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    n, c, h, w = img.shape

    # 11x11 Gaussian window, sigma=1.5
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = torch.from_numpy(np.outer(kernel, kernel.transpose())).to(img.device, img.dtype).unsqueeze(0).unsqueeze(0)
    window = window.expand(c, 1, 11, 11)

    mu1 = torch.nn.functional.conv2d(img, window, padding=0, groups=c)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=0, groups=c)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(img ** 2, window, padding=0, groups=c) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 ** 2, window, padding=0, groups=c) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img * img2, window, padding=0, groups=c) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
        ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    return torch.mean(ssim_map, dim=[1, 2, 3])
