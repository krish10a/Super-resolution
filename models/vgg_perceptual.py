import torch
import torch.nn as nn
from torchvision import models

# VGG16 layer indexing in torchvision:
# 0: Conv2d(3, 64, 3, 1, 1), 1: ReLU
# 2: Conv2d(64, 64, 3, 1, 1), 3: ReLU (relu1_2)
# 4: MaxPool2d
# 5: Conv2d(64, 128, 3, 1, 1), 6: ReLU (relu2_1)
# 7: Conv2d(128, 128, 3, 1, 1), 8: ReLU (relu2_2) -> features[:9]
# 9: MaxPool2d
# 10: Conv2d(128, 256, 3, 1, 1), 11: ReLU (relu3_1)
# 12: Conv2d(256, 256, 3, 1, 1), 13: ReLU (relu3_2)
# 14: Conv2d(256, 256, 3, 1, 1), 15: ReLU (relu3_3) -> features[:16]

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
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.features = nn.Sequential(*list(vgg.children())[:end_idx])
        self.feature_layer = feature_layer

        # ImageNet normalization constants
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        for p in self.features.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        # Expect x in range [0, 1]
        x_norm = (x - self.mean) / self.std
        return self.features(x_norm)
