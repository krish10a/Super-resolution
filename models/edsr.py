import math
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """Residual block without Batch Normalization (Lim et al., 2017)."""
    def __init__(self, n_feats: int, kernel_size: int = 3, res_scale: float = 1.0):
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x).mul(self.res_scale)
        return x + res


class Upsampler(nn.Sequential):
    """Sub-pixel convolution upsampler (PixelShuffle)."""
    def __init__(self, scale: int, n_feats: int):
        m = []
        if (scale & (scale - 1)) == 0:  # Is power of 2 (2x, 4x, 8x)
            for _ in range(int(math.log2(scale))):
                m.append(nn.Conv2d(n_feats, 4 * n_feats, kernel_size=3, padding=1))
                m.append(nn.PixelShuffle(2))
                m.append(nn.ReLU(inplace=True))
        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 9 * n_feats, kernel_size=3, padding=1))
            m.append(nn.PixelShuffle(3))
            m.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError(f"Scale {scale} is not supported.")
        super().__init__(*m)


class EDSR(nn.Module):
    """Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR-Baseline).

    Reference:
    Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, Kyoung Mu Lee.
    "Enhanced Deep Residual Networks for Single Image Super-Resolution", CVPRW 2017.

    Baseline Configuration:
    - n_resblocks: 16
    - n_feats: 64
    - res_scale: 1.0
    - rgb_range: 1.0 (input and target normalized in [0, 1])
    - Total parameters: ~1.37M (2x), ~1.52M (4x)
    """
    def __init__(self, scale: int = 2, in_channels: int = 3, out_channels: int = 3,
                 n_feats: int = 64, n_resblocks: int = 16, res_scale: float = 1.0):
        super().__init__()
        self.scale = scale

        # Head
        self.head = nn.Conv2d(in_channels, n_feats, kernel_size=3, padding=1)

        # Body: residual trunk + conv
        body = [ResBlock(n_feats, kernel_size=3, res_scale=res_scale) for _ in range(n_resblocks)]
        body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)

        # Tail: upsampler + reconstruction conv
        self.tail = nn.Sequential(
            Upsampler(scale, n_feats),
            nn.Conv2d(n_feats, out_channels, kernel_size=3, padding=1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        res = self.body(feat)
        res = res + feat
        out = self.tail(res)
        return out
