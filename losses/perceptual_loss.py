import torch.nn as nn
from models.vgg_perceptual import VGGPerceptual

class HybridLoss(nn.Module):
    def __init__(self, lambda_perc=0.07):
        super().__init__()
        self.mse = nn.MSELoss()
        self.vgg = VGGPerceptual()
        self.lambda_perc = lambda_perc

    def forward(self, pred, target):
        mse = self.mse(pred, target)
        feat_pred = self.vgg(pred)
        feat_target = self.vgg(target)
        perc = nn.functional.mse_loss(feat_pred, feat_target)
        return mse + self.lambda_perc * perc
