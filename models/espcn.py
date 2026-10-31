import torch.nn as nn
import torch

class ESPCN(nn.Module):
    def __init__(self, scale=2, channels=64):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3 * (scale ** 2), kernel_size=3, padding=1),
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.net(x)
        x = self.pixel_shuffle(x)
        return x
