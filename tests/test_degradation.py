"""
Tests for data/degradation.py — on-the-fly LR degradation pipeline.

Covers: downscale_bicubic, gaussian_blur, gaussian_noise, jpeg_compress,
and the composed degrade() function.
"""

import cv2
import numpy as np
import pytest

from data.degradation import (
    downscale_bicubic,
    gaussian_blur,
    gaussian_noise,
    jpeg_compress,
    degrade,
)


class TestDownscaleBicubic:
    """Tests for bicubic downsampling."""

    def test_reduces_dimensions(self):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        lr = downscale_bicubic(img, scale=2)
        assert lr.shape == (32, 32, 3)

    def test_scale4(self):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        lr = downscale_bicubic(img, scale=4)
        assert lr.shape == (16, 16, 3)

    def test_preserves_channels(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        lr = downscale_bicubic(img, scale=2)
        assert lr.shape[2] == 3

    def test_returns_uint8(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        lr = downscale_bicubic(img, scale=2)
        assert lr.dtype == np.uint8

    def test_single_pixel_minimum(self):
        img = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
        lr = downscale_bicubic(img, scale=8)  # 4//8 = 0 -> max(1, ...) = 1
        assert lr.shape[0] >= 1 and lr.shape[1] >= 1


class TestGaussianBlur:
    """Tests for Gaussian blur."""

    def test_preserves_shape(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        out = gaussian_blur(img, ksize=5, sigma=1.0)
        assert out.shape == img.shape

    def test_preserves_dtype(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        out = gaussian_blur(img, ksize=5, sigma=1.0)
        assert out.dtype == np.uint8

    def test_even_ksize_becomes_odd(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        out = gaussian_blur(img, ksize=4, sigma=1.0)  # should become 5
        assert out.shape == img.shape


class TestGaussianNoise:
    """Tests for additive Gaussian noise."""

    def test_preserves_shape(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        out = gaussian_noise(img, sigma=5.0)
        assert out.shape == img.shape

    def test_preserves_dtype(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        out = gaussian_noise(img, sigma=5.0)
        assert out.dtype == np.uint8

    def test_range_clipped(self):
        img = np.full((8, 8, 3), 250, dtype=np.uint8)
        out = gaussian_noise(img, sigma=50.0)
        assert out.min() >= 0 and out.max() <= 255

    def test_zero_sigma_no_change(self):
        img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        out = gaussian_noise(img, sigma=0.0)
        np.testing.assert_array_equal(out, img)


class TestJpegCompress:
    """Tests for JPEG compression simulation."""

    def test_preserves_shape(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        out = jpeg_compress(img, quality=75)
        assert out.shape == img.shape

    def test_preserves_dtype(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        out = jpeg_compress(img, quality=75)
        assert out.dtype == np.uint8

    def test_quality_bounds(self):
        img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        out_low = jpeg_compress(img, quality=1)
        out_high = jpeg_compress(img, quality=100)
        assert out_low.shape == img.shape
        assert out_high.shape == img.shape


class TestDegrade:
    """Tests for the composed degrade pipeline."""

    def test_preserves_shape(self):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        out = degrade(img, scale=2)
        assert out.shape == (32, 32, 3)

    def test_returns_uint8(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        out = degrade(img, scale=2)
        assert out.dtype == np.uint8

    def test_no_blur_no_noise_no_jpeg(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        out = degrade(img, scale=2, blur_ksize=1, noise_sigma=0.0,
                      jpeg_quality=None)
        assert out.shape == (16, 16, 3)

    def test_with_all_steps(self):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        out = degrade(img, scale=2, blur_ksize=5, blur_sigma=1.0,
                      noise_sigma=5.0, jpeg_quality=75)
        assert out.shape == (32, 32, 3)
        assert out.dtype == np.uint8

    def test_deterministic_without_noise(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        out1 = degrade(img, scale=2, blur_ksize=5, blur_sigma=1.0,
                       noise_sigma=0.0, jpeg_quality=None)
        out2 = degrade(img, scale=2, blur_ksize=5, blur_sigma=1.0,
                       noise_sigma=0.0, jpeg_quality=None)
        np.testing.assert_array_equal(out1, out2)
