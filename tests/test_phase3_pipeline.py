"""
Phase 3 Pipeline Test Suite: DIV2K Dataset Discovery, Model Architectures,
Loss Functions, AMP Optimization, and Checkpoint Compatibility.
"""

import os
import json
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from models.espcn import ESPCN
from models.fsrcnn import FSRCNN
from models.edsr import EDSR
from losses.charbonnier_loss import CharbonnierLoss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIV2K_RAW = PROJECT_ROOT / "datasets" / "Div2k dataset"


# ---------------------------------------------------------------------------
# 1. DIV2K Dataset Discovery Tests
# ---------------------------------------------------------------------------
class TestDIV2KDiscovery:
    def test_raw_dataset_directories_exist(self):
        train_dir = DIV2K_RAW / "DIV2K_train_HR"
        valid_dir = DIV2K_RAW / "DIV2K_valid_HR"
        assert train_dir.exists(), f"DIV2K_train_HR missing in {DIV2K_RAW}"
        assert valid_dir.exists(), f"DIV2K_valid_HR missing in {DIV2K_RAW}"

    def test_exact_image_counts(self):
        train_dir = DIV2K_RAW / "DIV2K_train_HR"
        valid_dir = DIV2K_RAW / "DIV2K_valid_HR"
        train_files = list(train_dir.glob("*.png"))
        valid_files = list(valid_dir.glob("*.png"))
        assert len(train_files) == 800, f"Expected 800 train images, got {len(train_files)}"
        assert len(valid_files) == 100, f"Expected 100 valid images, got {len(valid_files)}"

    def test_zero_filename_overlap(self):
        train_dir = DIV2K_RAW / "DIV2K_train_HR"
        valid_dir = DIV2K_RAW / "DIV2K_valid_HR"
        train_names = {f.name for f in train_dir.glob("*.png")}
        valid_names = {f.name for f in valid_dir.glob("*.png")}
        assert train_names.isdisjoint(valid_names), "Train and Valid sets must have zero filename overlap"


# ---------------------------------------------------------------------------
# 2. Model Architecture Forward Tests (E9 - E14)
# ---------------------------------------------------------------------------
class TestArchitectures:
    @pytest.mark.parametrize("scale", [2, 4])
    def test_espcn_forward_shapes(self, scale):
        m = ESPCN(scale=scale, mid_channels=32)
        x = torch.randn(2, 3, 32, 32)
        out = m(x)
        assert out.shape == (2, 3, 32 * scale, 32 * scale)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize("scale", [2, 4])
    def test_fsrcnn_forward_shapes(self, scale):
        m = FSRCNN(scale=scale)
        x = torch.randn(2, 3, 32, 32)
        out = m(x)
        assert out.shape == (2, 3, 32 * scale, 32 * scale)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize("scale", [2, 4])
    def test_edsr_forward_shapes(self, scale):
        m = EDSR(scale=scale, n_feats=64, n_resblocks=16)
        x = torch.randn(2, 3, 32, 32)
        out = m(x)
        assert out.shape == (2, 3, 32 * scale, 32 * scale)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# 3. Loss Functions & Gradient Tests
# ---------------------------------------------------------------------------
class TestLossFunctions:
    def test_charbonnier_loss_zero_diff(self):
        loss_fn = CharbonnierLoss(eps=1e-3)
        pred = torch.ones(2, 3, 16, 16)
        target = torch.ones(2, 3, 16, 16)
        loss = loss_fn(pred, target)
        assert pytest.approx(loss.item(), rel=1e-4) == 1e-3

    def test_charbonnier_loss_gradient(self):
        loss_fn = CharbonnierLoss(eps=1e-3)
        pred = torch.randn(2, 3, 16, 16, requires_grad=True)
        target = torch.randn(2, 3, 16, 16)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()


# ---------------------------------------------------------------------------
# 4. Optimization & AMP Mixed Precision Smoke Tests
# ---------------------------------------------------------------------------
class TestOptimizationAndAMP:
    @pytest.mark.parametrize("arch,model_cls", [
        ("espcn", lambda: ESPCN(scale=2, mid_channels=32)),
        ("fsrcnn", lambda: FSRCNN(scale=2)),
        ("edsr", lambda: EDSR(scale=2, n_feats=64, n_resblocks=4)),  # smaller for fast test
    ])
    def test_backprop_one_step(self, arch, model_cls):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_cls().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = CharbonnierLoss()

        lr = torch.randn(2, 3, 24, 24, device=device)
        hr = torch.randn(2, 3, 48, 48, device=device)

        optimizer.zero_grad()
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = model(lr)
                loss = loss_fn(out, hr)
            scaler = torch.amp.GradScaler("cuda")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(lr)
            loss = loss_fn(out, hr)
            loss.backward()
            optimizer.step()

        assert not torch.isnan(loss).any()
