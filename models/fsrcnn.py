import math
import torch
import torch.nn as nn

class FSRCNN(nn.Module):
    """Fast Super-Resolution Convolutional Neural Network (FSRCNN).

    Reference:
    Chao Dong, Chen Change Loy, Xiaoou Tang.
    "Accelerating the Super-Resolution Convolutional Neural Network", ECCV 2016.

    Architecture (9-1-5-1-9 / 5-1-3-1-9):
    1. Feature extraction: Conv(3, d, 5) + PReLU
    2. Shrinking: Conv(d, s, 1) + PReLU
    3. Mapping: m x (Conv(s, s, 3) + PReLU)
    4. Expanding: Conv(s, d, 1) + PReLU
    5. Deconvolution: ConvTranspose2d(d, 3, 9, stride=scale)
    """
    def __init__(self, scale: int = 2, in_channels: int = 3, d: int = 56, s: int = 12, m: int = 4):
        super().__init__()
        self.scale = scale

        # 1. Feature extraction
        layers = [
            nn.Conv2d(in_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d)
        ]

        # 2. Shrinking
        layers += [
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        ]

        # 3. Non-linear mapping (m layers)
        for _ in range(m):
            layers += [
                nn.Conv2d(s, s, kernel_size=3, padding=1),
                nn.PReLU(s)
            ]

        # 4. Expanding
        layers += [
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d)
        ]

        self.features = nn.Sequential(*layers)

        # 5. Deconvolution / Upsampling
        self.deconv = nn.ConvTranspose2d(
            d, in_channels, kernel_size=9, stride=scale, padding=4, output_padding=scale - 1
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.deconv.weight, mode='fan_in', nonlinearity='linear')
        if self.deconv.bias is not None:
            nn.init.zeros_(self.deconv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        out = self.deconv(feat)
        return out
