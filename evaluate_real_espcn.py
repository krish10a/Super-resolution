import torch, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.paths import pairs_real_dir, checkpoints_dir
from models.espcn import ESPCN
from metrics.psnr_ssim import compute_psnr_ssim
from PIL import Image
import torchvision.transforms.functional as TF

device='cpu'
model=ESPCN(scale=4)
ckpt=str(checkpoints_dir('espcn_real_best.pt'))
model.load_state_dict(torch.load(ckpt, map_location='cpu'))
model.eval()

lr_dir=pairs_real_dir('Set14_real_LR_x4')
hr_dir=pairs_real_dir('Set14_real_HR')

files=[f for f in os.listdir(lr_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
psnrs=[]; ssims=[]
for f in files:
    lr_path=os.path.join(lr_dir,f)
    name=os.path.splitext(f)[0]
    hr_path=None
    for ext in ['.png','.jpg','.jpeg']:
        cand=os.path.join(hr_dir,name+ext)
        if os.path.exists(cand):
            hr_path=cand; break
    if hr_path is None: continue
    lr_img=Image.open(lr_path).convert('RGB')
    hr_img=Image.open(hr_path).convert('RGB')
    lr_t=TF.to_tensor(lr_img).unsqueeze(0)
    with torch.no_grad():
        out=model(lr_t)
    # ensure size match
    if out.shape[-2:] != hr_img.size[::-1]:
        # crop to min
        h = min(out.shape[-2], hr_img.size[1])
        w = min(out.shape[-1], hr_img.size[0])
        out_c = out[..., :h, :w]
        hr_t_c = TF.to_tensor(hr_img.crop((0,0,w,h)))
    else:
        out_c = out
        hr_t_c = TF.to_tensor(hr_img)
    p,s=compute_psnr_ssim(out_c.squeeze(0), hr_t_c, scale=4)
    psnrs.append(p); ssims.append(s)
    print(f'{name}: PSNR {p:.2f} SSIM {s:.4f}')
print('Mean PSNR', sum(psnrs)/len(psnrs))
print('Mean SSIM', sum(ssims)/len(ssims))
