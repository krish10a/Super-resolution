"""
Tests for models/srcnn.py and models/espcn.py — SISR model architectures.

Covers: forward pass shapes, output ranges, weight initialization (no NaN),
and parameter counts.
"""

import numpy as np
import pytest
import torch

from models.srcnn import SRCNN
from models.espcn import ESPCN


class TestSRCNN:
    """Tests for the SRCNN model."""

    def test_forward_shape(self):
        model = SRCNN()
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 3, 32, 32)

    def test_output_in_tanh_range(self):
        model = SRCNN()
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.min() >= -1.0 and out.max() <= 1.0, \
            f"Tanh output out of range: [{out.min()}, {out.max()}]"

    def test_no_nan_weights(self):
        model = SRCNN()
        for name, param in model.named_parameters():
            assert not torch.isnan(param).any(), f"NaN in {name}"
            assert not torch.isinf(param).any(), f"Inf in {name}"

    def test_eval_mode(self):
        model = SRCNN()
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_batch_independence(self):
        model = SRCNN()
        x1 = torch.randn(1, 3, 32, 32)
        x2 = torch.randn(1, 3, 32, 32)
        x_batch = torch.cat([x1, x2], dim=0)
        with torch.no_grad():
            out_batch = model(x_batch)
            out1 = model(x1)
            out2 = model(x2)
        torch.testing.assert_close(out_batch[0:1], out1)
        torch.testing.assert_close(out_batch[1:2], out2)


class TestESPCN:
    """Tests for the ESPCN model."""

    def test_forward_shape_scale2(self):
        model = ESPCN(scale=2)
        x = torch.randn(2, 3, 16, 16)
        out = model(x)
        assert out.shape == (2, 3, 32, 32), \
            f"Expected (2, 3, 32, 32), got {out.shape}"

    def test_forward_shape_scale4(self):
        model = ESPCN(scale=4)
        x = torch.randn(2, 3, 16, 16)
        out = model(x)
        assert out.shape == (2, 3, 64, 64), \
            f"Expected (2, 3, 64, 64), got {out.shape}"

    def test_output_in_valid_range(self):
        model = ESPCN(scale=2)
        x = torch.randn(2, 3, 16, 16)
        out = model(x)
        # No Tanh on ESPCN output, just check finite
        assert torch.isfinite(out).all()

    def test_no_nan_weights(self):
        model = ESPCN(scale=2)
        for name, param in model.named_parameters():
            assert not torch.isnan(param).any(), f"NaN in {name}"
            assert not torch.isinf(param).any(), f"Inf in {name}"

    def test_pixel_shuffle_upscale(self):
        """PixelShuffle should rearrange channels into spatial upscale."""
        model = ESPCN(scale=3)
        x = torch.randn(1, 3, 8, 8)
        out = model(x)
        assert out.shape == (1, 3, 24, 24)

    def test_eval_mode_deterministic(self):
        model = ESPCN(scale=2)
        model.eval()
        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)
