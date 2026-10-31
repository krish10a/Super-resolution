import torch
import torch.nn as nn
from torchvision import models

# VGG11 feature indices (children of .features):
# 0 conv1_1, 1 relu, 2 conv1_2, 3 relu, 4 pool,
# 5 conv2_1, 6 relu, 7 conv2_2, 8 relu, 9 pool,
# 10 conv3_1, 11 relu, 12 conv3_2, 13 relu, 14 conv3_3, 15 relu, 16 pool, ...
_FEATURE_END_IDX = {'relu2_2': 9, 'relu3_3': 16}


class VGGPerceptual(nn.Module):
    def __init__(self, feature_layer='relu2_2'):
        super().__init__()
        if feature_layer not in _FEATURE_END_IDX:
            raise ValueError(
                f"Unsupported feature_layer '{feature_layer}'. "
                f"Choose from {list(_FEATURE_END_IDX)}."
            )
        end_idx = _FEATURE_END_IDX[feature_layer]
        vgg = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1).features
        self.features = nn.Sequential(*list(vgg.children())[:end_idx])
        self.feature_layer = feature_layer
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

    def forward(self, x):
        return self.features(x)
