"""
Tests for data/imresize_matlab.py — MATLAB-compatible bicubic imresize.

Covers: output dimensions, antialiasing on downsample, no antialiasing on
upsample, dtype/range preservation, edge padding (no NaN/Inf), identity at
scale=1.0, and roundtrip consistency.
"""

import math
import numpy as np
import pytest

from data.imresize_matlab import (
    cubic,
    calculate_weights_indices,
    imresize_matlab,
    imresize_np,
    imresize_to_size,
)


class TestCubicKernel:
    """Tests for the cubic convolution kernel."""

    def test_kernel_at_zero_is_one(self):
        assert cubic(np.array([0.0]))[0] == pytest.approx(1.0, abs=1e-12)

    def test_kernel_is_even(self):
        x = np.array([0.5, 1.0, 1.5, 2.0])
        np.testing.assert_allclose(cubic(x), cubic(-x), atol=1e-12)

    def test_kernel_zero_beyond_support(self):
        x = np.array([2.1, 3.0, 5.0])
        np.testing.assert_allclose(cubic(x), 0.0, atol=1e-12)

    def test_kernel_scalar(self):
        assert cubic(0.5) == pytest.approx(1.5 * 0.125 - 2.5 * 0.25 + 1.0, abs=1e-12)

    def test_kernel_negative_values(self):
        # For |x| in (1, 2] the cubic kernel takes negative values.
        assert cubic(1.5) < 0


class TestCalculateWeightsIndices:
    """Tests for weight/index computation."""

    def test_upsample_weights_sum_to_one(self):
        indices, weights = calculate_weights_indices(
            in_length=10, out_length=20, scale=2.0,
            kernel=cubic, kernel_width=4.0, antialiasing=False,
        )
        for w in weights:
            assert np.isclose(np.sum(w), 1.0, atol=1e-9), \
                f"Weights do not sum to 1: {np.sum(w)}"

    def test_downsample_weights_sum_to_one(self):
        indices, weights = calculate_weights_indices(
            in_length=20, out_length=10, scale=0.5,
            kernel=cubic, kernel_width=4.0, antialiasing=True,
        )
        for w in weights:
            assert np.isclose(np.sum(w), 1.0, atol=1e-9), \
                f"Weights do not sum to 1: {np.sum(w)}"

    def test_indices_within_range_upsample(self):
        # Indices may extend beyond [0, in_length-1] at boundaries; the
        # imresize_np padding step clamps them. Just check they are within
        # the kernel's expected reach (+/- kernel_width).
        indices, _ = calculate_weights_indices(
            in_length=10, out_length=20, scale=2.0,
            kernel=cubic, kernel_width=4.0, antialiasing=False,
        )
        for idx in indices:
            assert idx.min() >= -4, f"Index too far below 0: {idx.min()}"
            assert idx.max() <= 13, f"Index too far above range: {idx.max()}"

    def test_output_length_matches_ceil(self):
        """Out length should equal ceil(in * scale) per BasicSR convention."""
        for in_len in [1, 2, 7, 33, 100]:
            for scale in [0.25, 0.5, 1.0, 2.0, 4.0]:
                indices, _ = calculate_weights_indices(
                    in_length=in_len, out_length=math.ceil(in_len * scale),
                    scale=scale, kernel=cubic, kernel_width=4.0,
                    antialiasing=(scale < 1),
                )
                assert len(indices) == math.ceil(in_len * scale), \
                    f"Expected {math.ceil(in_len * scale)} output indices, got {len(indices)}"


class TestImresizeDimensions:
    """Tests for output dimensions of imresize."""

    def test_upsample_x2(self, gradient_image):
        out = imresize_np(gradient_image, 2.0, antialiasing=False)
        assert out.shape[0] == math.ceil(64 * 2.0)
        assert out.shape[1] == math.ceil(64 * 2.0)
        assert out.shape[2] == 3

    def test_upsample_x4(self, gradient_image):
        out = imresize_np(gradient_image, 4.0, antialiasing=False)
        assert out.shape[0] == math.ceil(64 * 4.0)
        assert out.shape[1] == math.ceil(64 * 4.0)

    def test_downsample_x2_antialiasing(self, gradient_image):
        out = imresize_np(gradient_image, 0.5, antialiasing=True)
        assert out.shape[0] == math.ceil(64 * 0.5)
        assert out.shape[1] == math.ceil(64 * 0.5)

    def test_downsample_x4_antialiasing(self, gradient_image):
        out = imresize_np(gradient_image, 0.25, antialiasing=True)
        assert out.shape[0] == math.ceil(64 * 0.25)
        assert out.shape[1] == math.ceil(64 * 0.25)

    def test_scale_one_is_identity_shape(self, gradient_image):
        out = imresize_np(gradient_image, 1.0, antialiasing=False)
        assert out.shape == gradient_image.shape

    def test_odd_dimensions(self, odd_dim_image):
        out = imresize_np(odd_dim_image, 2.0, antialiasing=False)
        assert out.shape[0] == math.ceil(99 * 2.0)
        assert out.shape[1] == math.ceil(97 * 2.0)

    def test_single_pixel(self):
        img = np.array([[[128, 128, 128]]], dtype=np.uint8)
        out = imresize_np(img, 2.0, antialiasing=False)
        assert out.shape == (2, 2, 3)  # ceil(1 * 2) = 2


class TestImresizeDtypeAndRange:
    """Tests for dtype and value-range preservation."""

    def test_uint8_stays_uint8(self, gradient_image):
        out = imresize_np(gradient_image, 2.0, antialiasing=False)
        assert out.dtype == np.uint8

    def test_uint8_range_preserved(self, gradient_image):
        out = imresize_np(gradient_image, 2.0, antialiasing=False)
        assert out.min() >= 0 and out.max() <= 255

    def test_float_stays_float(self, gradient_image):
        img_f = gradient_image.astype(np.float64) / 255.0
        out = imresize_np(img_f, 2.0, antialiasing=False)
        assert np.issubdtype(out.dtype, np.floating)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_no_nan_inf(self, odd_dim_image):
        for scale in [0.25, 0.5, 1.0, 2.0, 4.0]:
            out = imresize_np(odd_dim_image, scale, antialiasing=(scale < 1))
            assert not np.isnan(out).any(), f"NaN at scale {scale}"
            assert not np.isinf(out).any(), f"Inf at scale {scale}"

    def test_constant_image_stays_constant(self, constant_red_image):
        # Resizing a constant image should keep each channel constant
        # (bicubic with normalized weights preserves constant channels).
        out = imresize_np(constant_red_image, 2.0, antialiasing=False)
        assert out[:, :, 0].min() >= 250 and out[:, :, 0].max() <= 255, \
            f"Red channel changed: min={out[:, :, 0].min()} max={out[:, :, 0].max()}"
        assert out[:, :, 1].min() == 0 and out[:, :, 1].max() == 0, \
            f"Green channel not zero: min={out[:, :, 1].min()} max={out[:, :, 1].max()}"
        assert out[:, :, 2].min() == 0 and out[:, :, 2].max() == 0, \
            f"Blue channel not zero: min={out[:, :, 2].min()} max={out[:, :, 2].max()}"

    def test_black_stays_black(self, constant_black_image):
        out = imresize_np(constant_black_image, 2.0, antialiasing=False)
        assert np.all(out == 0), "Black image should stay black"

    def test_white_stays_white(self, constant_white_image):
        out = imresize_np(constant_white_image, 2.0, antialiasing=False)
        assert np.all(out == 255), "White image should stay white"


class TestImresizeAntialiasing:
    """Tests that antialiasing affects downsampling but not upsampling."""

    def test_antialiasing_changes_downsample(self, gradient_image):
        out_aa = imresize_np(gradient_image, 0.5, antialiasing=True)
        out_no_aa = imresize_np(gradient_image, 0.5, antialiasing=False)
        # They should differ (antialiasing blurs)
        assert not np.array_equal(out_aa, out_no_aa), \
            "Antialiasing should change downsampled output"

    def test_antialiasing_irrelevant_on_upsample(self, gradient_image):
        # For upsampling antialiasing flag should not change behavior
        # (kernel not widened when scale >= 1).
        out_aa = imresize_np(gradient_image, 2.0, antialiasing=True)
        out_no_aa = imresize_np(gradient_image, 2.0, antialiasing=False)
        np.testing.assert_array_equal(out_aa, out_no_aa)


class TestImresizeRoundtrip:
    """Tests roundtrip downsample->upsample consistency."""

    def test_roundtrip_shape(self, gradient_image):
        lr = imresize_np(gradient_image, 0.5, antialiasing=True)
        hr_up = imresize_np(lr, 2.0, antialiasing=False)
        assert hr_up.shape == gradient_image.shape


class TestImresizeToSize:
    """Tests for imresize_to_size (exact output size)."""

    def test_to_size_upscale(self, gradient_image):
        out = imresize_to_size(gradient_image, (128, 128), antialiasing=False)
        assert out.shape == (128, 128, 3)

    def test_to_size_downscale(self, gradient_image):
        out = imresize_to_size(gradient_image, (16, 16), antialiasing=True)
        assert out.shape == (16, 16, 3)

    def test_to_size_same_returns_copy(self, gradient_image):
        out = imresize_to_size(gradient_image, (64, 64), antialiasing=False)
        assert out.shape == (64, 64, 3)
        np.testing.assert_array_equal(out, gradient_image)

    def test_to_size_no_nan_inf(self, odd_dim_image):
        out = imresize_to_size(odd_dim_image, (50, 50), antialiasing=True)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_to_size_dtype(self, gradient_image):
        out = imresize_to_size(gradient_image, (32, 32), antialiasing=False)
        assert out.dtype == np.uint8


class TestImresizeMatlabWrapper:
    """Tests for the imresize_matlab convenience wrapper."""

    def test_wrapper_matches_imresize_np(self, gradient_image):
        a = imresize_matlab(gradient_image, 2.0, antialiasing=False)
        b = imresize_np(gradient_image, 2.0, antialiasing=False)
        np.testing.assert_array_equal(a, b)
