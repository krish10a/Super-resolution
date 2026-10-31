import torch
from data.dataset import PatchDataset
from models.espcn import ESPCN
from metrics.psnr_ssim import compute_psnr_ssim

from config.paths import pairs_dir, checkpoints_dir
lr_dir = pairs_dir('Set5_LR_x4')
hr_dir = pairs_dir('Set5_HR')
ds = PatchDataset(lr_dir, hr_dir, scale=4, patch_size=32, upsample_lr=False)
model = ESPCN(scale=4)
ckpt = str(checkpoints_dir('espcn_best.pt'))
model.load_state_dict(torch.load(ckpt, map_location='cpu'))
model.eval()
psnrs, ssims = [], []
for i in range(10):
    lr, hr = ds[i]
    with torch.no_grad():
        pred = model(lr.unsqueeze(0)).squeeze(0)
    p, s = compute_psnr_ssim(pred, hr)
    psnrs.append(p); ssims.append(s)
print('ESPCN best avg PSNR', sum(psnrs)/len(psnrs), 'SSIM', sum(ssims)/len(ssims))
