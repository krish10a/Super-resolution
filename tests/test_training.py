"""
Tests for the training pipeline (train/train_all.py, train/trainer.py).

These are lightweight, CPU-friendly tests: a single training step on synthetic
data, checkpoint save/load, and loss computation. Full multi-epoch training
is intentionally not covered here (it is slow and covered by the experiment
scripts themselves).
"""

import numpy as np
import pytest
import torch

from models.srcnn import SRCNN
from models.espcn import ESPCN
from losses.mse_loss import MSELoss
from losses.perceptual_loss import HybridLoss
from train.trainer import Trainer


class TestMSELoss:
    """Tests for MSELoss."""

    def test_identical_inputs_zero_loss(self):
        loss_fn = MSELoss()
        pred = torch.rand(2, 3, 16, 16)
        l = loss_fn(pred, pred)
        assert l.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_inputs_positive_loss(self):
        loss_fn = MSELoss()
        pred = torch.zeros(2, 3, 16, 16)
        target = torch.ones(2, 3, 16, 16)
        l = loss_fn(pred, target)
        assert l.item() > 0


class TestHybridLoss:
    """Tests for HybridLoss (MSE + perceptual)."""

    def test_mse_only_when_lambda_zero(self):
        loss_fn = HybridLoss(lambda_perc=0.0)
        pred = torch.rand(2, 3, 16, 16)
        target = torch.rand(2, 3, 16, 16)
        l = loss_fn(pred, target)
        # With lambda_perc=0, loss is pure MSE
        expected = torch.nn.functional.mse_loss(pred, target)
        assert l.item() == pytest.approx(expected.item(), abs=1e-6)

    def test_hybrid_loss_positive(self):
        loss_fn = HybridLoss(lambda_perc=0.07)
        pred = torch.rand(2, 3, 16, 16)
        target = torch.rand(2, 3, 16, 16)
        l = loss_fn(pred, target)
        assert l.item() > 0


class TestTrainerSingleStep:
    """Tests for Trainer.train_one_epoch on a tiny synthetic loader."""

    def test_train_one_epoch_srcnn(self):
        model = SRCNN()
        loss_fn = MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        trainer = Trainer(model, loss_fn, optimizer, device='cpu')

        # Synthetic loader: 2 batches of (LR, HR) patches
        lr = torch.rand(2, 3, 16, 16)
        hr = torch.rand(2, 3, 16, 16)
        dataset = torch.utils.data.TensorDataset(lr, hr)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        train_loss = trainer.train_one_epoch(loader, grad_accum=1)
        assert np.isfinite(train_loss)
        assert train_loss >= 0

    def test_train_one_epoch_espcn(self):
        model = ESPCN(scale=2)
        loss_fn = MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        trainer = Trainer(model, loss_fn, optimizer, device='cpu')

        lr = torch.rand(2, 3, 8, 8)    # LR input
        hr = torch.rand(2, 3, 16, 16)  # HR target (2x upscale)
        dataset = torch.utils.data.TensorDataset(lr, hr)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        train_loss = trainer.train_one_epoch(loader, grad_accum=1)
        assert np.isfinite(train_loss)
        assert train_loss >= 0

    def test_validate_returns_finite(self):
        model = SRCNN()
        loss_fn = MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        trainer = Trainer(model, loss_fn, optimizer, device='cpu')

        from metrics.psnr_ssim import compute_psnr_ssim

        lr = torch.rand(2, 3, 16, 16)
        hr = torch.rand(2, 3, 16, 16)
        dataset = torch.utils.data.TensorDataset(lr, hr)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        val_psnr, val_ssim = trainer.validate(loader, compute_psnr_ssim,
                                              scale=2)
        assert np.isfinite(val_psnr)
        assert -1.0 <= val_ssim <= 1.0


class TestCheckpointSaveLoad:
    """Tests for checkpoint save and load roundtrip."""

    def test_save_and_load_state_dict(self, tmp_path):
        model = SRCNN()
        ckpt_path = str(tmp_path / "test_model.pt")
        torch.save(model.state_dict(), ckpt_path)

        model2 = SRCNN()
        state = torch.load(ckpt_path, map_location='cpu')
        model2.load_state_dict(state)

        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        torch.testing.assert_close(out1, out2)
