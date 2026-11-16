import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg_perceptual import VGGPerceptual


class HybridLoss(nn.Module):
    def __init__(self, lambda_perc=0.07, feature_layer='relu2_2'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.vgg = VGGPerceptual(feature_layer=feature_layer)
        self.lambda_perc = lambda_perc

    def forward(self, pred, target):
        # pred and target are in [0, 1] range
        l_mse = self.mse(pred, target)

        # VGG features - skip during MSE-only warmup (lambda_perc == 0)
        if self.lambda_perc > 0:
            feat_pred = self.vgg(pred)
            with torch.no_grad():
                feat_target = self.vgg(target)
            l_perc = F.mse_loss(feat_pred, feat_target)
            return l_mse + self.lambda_perc * l_perc
        return l_mse
