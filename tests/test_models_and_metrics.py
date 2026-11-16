import pytest
import torch
from models.srcnn import SRCNN
from models.espcn import ESPCN
from metrics.psnr_ssim import compute_psnr_ssim, rgb_to_y_cbcr, psnr
from losses.perceptual_loss import HybridLoss

def test_srcnn_forward():
    model = SRCNN()
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 3, 64, 64)
    # Check tanh output range [-1, 1]
    assert out.min() >= -1.0 - 1e-5
    assert out.max() <= 1.0 + 1e-5

def test_espcn_forward():
    model_2x = ESPCN(scale=2)
    x = torch.randn(2, 3, 32, 32)
    out_2x = model_2x(x)
    assert out_2x.shape == (2, 3, 64, 64)

    model_4x = ESPCN(scale=4)
    out_4x = model_4x(x)
    assert out_4x.shape == (2, 3, 128, 128)

def test_metrics_identical_images():
    img1 = torch.ones(3, 64, 64)
    img2 = torch.ones(3, 64, 64)
    p, s = compute_psnr_ssim(img1, img2, scale=2, y_channel=True)
    assert p >= 80.0  # PSNR near infinite/100 dB
    assert pytest.approx(s, rel=1e-3) == 1.0

def test_hybrid_loss_reduction():
    loss_fn = HybridLoss(lambda_perc=0.05)
    pred = torch.rand(2, 3, 64, 64, requires_grad=True)
    target = torch.rand(2, 3, 64, 64)
    loss = loss_fn(pred, target)
    assert loss.item() > 0.0
    loss.backward()
    assert pred.grad is not None
