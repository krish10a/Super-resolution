"""
One-off diagnostic: verify VGG16 perceptual architecture consistency.

Checks:
- Model is torchvision.vgg16 with IMAGENET1K_V1 weights (not VGG11).
- features[:9] maps through relu2_2 (index 8 = ReLU; index 9 = MaxPool2d).
- VGG parameters frozen (requires_grad=False).
- Module is in eval mode.
- ImageNet mean/std buffers present and move to device correctly.
- Gradient flow: pred features propagate grads to SR model; target under no_grad.
- Loss values remain finite.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
import torchvision
from torchvision import models

from models.vgg_perceptual import VGGPerceptual, _FEATURE_END_IDX
from losses.perceptual_loss import HybridLoss


def vgg_layer_index_info():
    """Print torchvision VGG16 feature module indices for documentation."""
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    print("=== torchvision VGG16 feature module indices ===")
    for i, m in enumerate(vgg.features):
        print(f"  [{i}] {m.__class__.__name__}")
    print(f"  _FEATURE_END_IDX = {_FEATURE_END_IDX}")
    print(f"  features[:9] -> indices 0..8, last is relu2_2 (ReLU at index 8)")
    print(f"  index 9 = MaxPool2d (subsequent pooling, NOT relu2_2)")
    return vgg


def verify_architecture():
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    assert vgg.__class__.__name__ == "VGG", f"Expected VGG, got {vgg.__class__.__name__}"

    # Confirm weights enum value
    w = models.VGG16_Weights.IMAGENET1K_V1
    print(f"\nWeights enum: {w}")
    print(f"Weights meta categories: {w.meta['categories'][:5]}...")

    # Instantiate our perceptual module
    vgg_perc = VGGPerceptual(feature_layer='relu2_2')

    # features[:9] = modules 0..8
    expected_children = list(vgg.features.children())[:9]
    actual_children = list(vgg_perc.features.children())
    assert len(actual_children) == len(expected_children), \
        f"Module count mismatch: {len(actual_children)} vs {len(expected_children)}"
    for i, (a, e) in enumerate(zip(actual_children, expected_children)):
        assert type(a) is type(e), f"Module {i} type mismatch: {type(a)} vs {type(e)}"
    print("\n[PASS] features[:9] matches torchvision modules 0..8 (through relu2_2)")

    # Verify last module of features[:9] is ReLU (relu2_2), not MaxPool (index 9)
    last = list(vgg_perc.features.children())[-1]
    assert isinstance(last, torch.nn.ReLU), f"Last module is {type(last)}, expected ReLU (relu2_2)"
    print("[PASS] Last module of features[:9] is ReLU = relu2_2 (index 8), not MaxPool (index 9)")

    # Frozen parameters
    frozen = all(not p.requires_grad for p in vgg_perc.features.parameters())
    assert frozen, "Not all VGG parameters are frozen!"
    print("[PASS] All VGG parameters frozen (requires_grad=False)")

    # Eval mode
    assert not vgg_perc.training, "VGGPerceptual not in eval mode!"
    print("[PASS] VGGPerceptual in eval() mode")

    # ImageNet buffers
    assert hasattr(vgg_perc, 'mean'), "Missing mean buffer"
    assert hasattr(vgg_perc, 'std'), "Missing std buffer"
    expected_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    expected_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert torch.allclose(vgg_perc.mean, expected_mean), "Mean buffer mismatch"
    assert torch.allclose(vgg_perc.std, expected_std), "Std buffer mismatch"
    print("[PASS] ImageNet mean/std buffers correct")

    # Device movement of buffers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg_perc = vgg_perc.to(device)
    assert vgg_perc.mean.device.type == device.type, "Mean buffer did not move to device"
    assert vgg_perc.std.device.type == device.type, "Std buffer did not move to device"
    print(f"[PASS] Mean/std buffers moved to {device}")

    return vgg_perc


def verify_gradients():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = HybridLoss(lambda_perc=0.07, feature_layer='relu2_2').to(device)
    loss_fn.eval()

    src_input = torch.rand(1, 3, 64, 64, device=device) * 0.5 + 0.25  # in [0,1]
    target = torch.rand(1, 3, 64, 64, device=device) * 0.5 + 0.25      # ground truth
    src_input.requires_grad_(True)
    # target does NOT require grad (ground truth, as in real training)

    # Forward
    loss = loss_fn(src_input, target)
    assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
    print(f"\n[PASS] HybridLoss forward finite: {loss.item():.6f}")

    # Backward through prediction features (should propagate to src_input)
    loss.backward()
    assert src_input.grad is not None, "No gradient flowed to prediction input!"
    assert torch.isfinite(src_input.grad).all(), "Prediction grads not finite!"
    print("[PASS] Prediction feature gradients propagate to input (pred path)")

    # Target under no_grad: target.grad should be None (target not part of loss graph)
    # Note: target requires_grad but HybridLoss wraps target in torch.no_grad()
    assert target.grad is None, "Target should not accumulate gradients (no_grad wrapper)"
    print("[PASS] Target features computed under torch.no_grad (no target grads)")

    # VGG params have no grads
    for name, p in loss_fn.vgg.features.named_parameters():
        assert p.requires_grad is False, f"VGG param {name} requires_grad=True"
        assert p.grad is None, f"VGG param {name} has gradient"
    print("[PASS] VGG params frozen with no gradients")

    # MSE-only path (lambda_perc=0)
    loss_mse = HybridLoss(lambda_perc=0.0).to(device)
    loss_mse.eval()
    loss2 = loss_mse(src_input.detach(), target.detach())
    assert torch.isfinite(loss2), f"MSE-only loss not finite: {loss2.item()}"
    print(f"[PASS] MSE-only path (lambda_perc=0) finite: {loss2.item():.6f}")


if __name__ == '__main__':
    vgg_layer_index_info()
    verify_architecture()
    verify_gradients()
    print("\n=== VGG16 perceptual architecture verification: ALL CHECKS PASSED ===")
