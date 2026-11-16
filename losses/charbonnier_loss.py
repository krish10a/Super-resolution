import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (smooth L1 / robust loss function).

    L(x, y) = sqrt((x - y)^2 + eps^2)
    Frequently used in SISR (LapSRN, RCAN, NTIRE challenges) as a differentiable,
    outlier-robust alternative to L1 and MSE.
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)
