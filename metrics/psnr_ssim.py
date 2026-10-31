import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

def rgb_to_y(img):
    # img in [0,1] CxHxW
    r, g, b = img[0], img[1], img[2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y

def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse + 1e-10)

def compute_psnr_ssim(pred, target, scale=1):
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    if pred.shape[-2] != target.shape[-2]:
        # shave borders
        border = scale
        pred = pred[..., border:-border, border:-border]
        target = target[..., border:-border, border:-border]
    y_pred = rgb_to_y(pred)
    y_tar = rgb_to_y(target)
    p = psnr(y_pred, y_tar).item()
    s = ssim(y_pred.numpy(), y_tar.numpy(), data_range=1.0)
    return p, s
