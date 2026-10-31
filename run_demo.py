import os, torch
from utils.cpu_affinity import set_cpu_threads
from data.dataset import PatchDataset
from torch.utils.data import DataLoader
from models.srcnn import SRCNN
from losses.mse_loss import MSELoss
from train.trainer import Trainer
from metrics.psnr_ssim import compute_psnr_ssim

set_cpu_threads()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

# Create synthetic data via existing scripts
os.system('python data/download_sets.py')
os.system('python data/prepare_pairs.py')

from config.paths import pairs_dir, checkpoints_dir
lr_dir = pairs_dir('Set5_LR_x2')
hr_dir = pairs_dir('Set5_HR')
dataset = PatchDataset(lr_dir, hr_dir, scale=2, patch_size=32)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

model = SRCNN()
loss_fn = MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
trainer = Trainer(model, loss_fn, opt, device, scaler=torch.cuda.amp.GradScaler() if device=='cuda' else None)

for epoch in range(2):
    loss = trainer.train_one_epoch(loader, grad_accum=1)
    print(f'Epoch {epoch+1} loss {loss:.6f}')

# Quick eval on one sample
model.eval()
lr, hr = dataset[0]
with torch.no_grad():
    pred = model(lr.unsqueeze(0).to(device)).squeeze(0)
    p, s = compute_psnr_ssim(pred.cpu(), hr)
    print(f'Eval PSNR {p:.2f} SSIM {s:.4f}')

trainer.save(str(checkpoints_dir('srcnn_demo.pt')))
print('Demo complete')
