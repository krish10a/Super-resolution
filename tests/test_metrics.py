"""
Tests for metrics/psnr_ssim.py — project PSNR/SSIM implementation.

Covers: BT.601 Y-channel conversion, PSNR for known MSE values,
SSIM for identical images (1.0), border shaving, value ranges,
and clamped inputs.
"""

import numpy as np
import pytest
import torch

from metrics.psnr_ssim import (
    rgb_to_y_cbcr,
    psnr,
    compute_psnr_ssim,
)


class TestRgbToYCbCr:
    """Tests for ITU-R BT.601 RGB->Y conversion."""

    def test_3d_tensor(self):
        img = torch.tensor([[[1.0, 0.0], [0.0, 1.0]],
                            [[0.0, 1.0], [1.0, 0.0]],
                            [[1.0, 1.0], [0.0, 0.0]]])  # (3, H, W)
        y = rgb_to_y_cbcr(img)
        assert y.shape == (2, 2)
        # Y should be in [0, 1]
        assert y.min() >= 0.0 and y.max() <= 1.0

    def test_4d_tensor(self):
        img = torch.rand(2, 3, 8, 8)
        y = rgb_to_y_cbcr(img)
        assert y.shape == (2, 8, 8)

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            rgb_to_y_cbcr(torch.rand(3))  # 1D

    def test_black_gives_min_y(self):
        img = torch.zeros(3, 2, 2)
        y = rgb_to_y_cbcr(img)
        # Y = 16/255 + 0 = 0.0627...
        expected = 16.0 / 255.0
        assert y[0, 0] == pytest.approx(expected, abs=1e-6)

    def test_white_gives_max_y(self):
        img = torch.ones(3, 2, 2)
        y = rgb_to_y_cbcr(img)
        # Y = 16/255 + (65.481+128.553+24.966)/255 = 219/255 = 0.8588...
        expected = (16.0 + 65.481 + 128.553 + 24.966) / 255.0
        assert y[0, 0] == pytest.approx(expected, abs=1e-6)


class TestPsnr:
    """Tests for PSNR calculation."""

    def test_identical_images_high_psnr(self):
        x = torch.rand(3, 32, 32)
        p = psnr(x, x, max_val=1.0)
        assert p == 100.0  # MSE < 1e-12 returns 100

    def test_known_mse(self):
        # MSE = 0.01 -> PSNR = 10*log10(1/0.01) = 20 dB
        x = torch.zeros(1, 4, 4)
        y = torch.full((1, 4, 4), 0.1)
        p = psnr(x, y, max_val=1.0)
        expected = 10 * np.log10(1.0 / 0.01)
        assert p == pytest.approx(expected, abs=1e-6)

    def test_different_images_finite_psnr(self):
        x = torch.rand(3, 32, 32)
        y = torch.rand(3, 32, 32)
        p = psnr(x, y, max_val=1.0)
        assert np.isfinite(p)
        assert p > 0


class TestComputePsnrSsim:
    """Tests for the combined compute_psnr_ssim."""

    def test_identical_images(self):
        img = torch.rand(3, 64, 64)
        p, s = compute_psnr_ssim(img, img, scale=2)
        assert p == 100.0  # identical -> infinite PSNR -> capped at 100
        assert s == pytest.approx(1.0, abs=1e-6)  # identical -> SSIM 1.0

    def test_border_shaving(self):
        # Shaving by scale removes border pixels.
        pred = torch.rand(3, 68, 68)
        target = torch.rand(3, 68, 68)
        p, s = compute_psnr_ssim(pred, target, scale=2)
        # Should run without error on shaved (64x64) tensors
        assert np.isfinite(p)
        assert 0.0 <= s <= 1.0

    def test_scale_zero_no_shave(self):
        pred = torch.rand(3, 32, 32)
        target = torch.rand(3, 32, 32)
        p, s = compute_psnr_ssim(pred, target, scale=0)
        assert np.isfinite(p)
        assert -1.0 <= s <= 1.0

    def test_psnr_range(self):
        pred = torch.rand(3, 64, 64)
        target = torch.rand(3, 64, 64)
        p, s = compute_psnr_ssim(pred, target, scale=2)
        assert np.isfinite(p)
        assert -1.0 <= s <= 1.0

    def test_clamped_output(self):
        # Inputs outside [0,1] should be clamped.
        pred = torch.full((3, 64, 64), 1.5)
        target = torch.full((3, 64, 64), 0.5)
        p, s = compute_psnr_ssim(pred, target, scale=2)
        assert np.isfinite(p)
        assert 0.0 <= s <= 1.0

    def test_4d_tensor(self):
        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)
        p, s = compute_psnr_ssim(pred, target, scale=2)
        assert np.isfinite(p)
        assert 0.0 <= s <= 1.0

    def test_grayscale_input_no_y_conversion(self):
        # Single-channel input should skip Y conversion.
        pred = torch.rand(1, 64, 64)
        target = torch.rand(1, 64, 64)
        p, s = compute_psnr_ssim(pred, target, scale=2, y_channel=True)
        assert np.isfinite(p)
        # SSIM can be slightly negative for random noise; just check finite.
        assert np.isfinite(s)
