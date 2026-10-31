import torch, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.paths import pairs_real_dir, checkpoints_dir
from data.dataset import PatchDataset
from torch.utils.data import DataLoader
from models.espcn import ESPCN
import torch.nn as nn

device='cpu'
print('Device', device)

lr_dir = pairs_real_dir('Set5_real_LR_x4')
hr_dir = pairs_real_dir('Set5_real_HR')
val_lr_dir = pairs_real_dir('Set14_real_LR_x4')
val_hr_dir = pairs_real_dir('Set14_real_HR')

train_ds = PatchDataset(lr_dir, hr_dir, scale=4, patch_size=24, upsample_lr=False)
val_ds = PatchDataset(val_lr_dir, val_hr_dir, scale=4, patch_size=24, upsample_lr=False)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

model = ESPCN(scale=4)
model.to(device)
criterion = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

best = float('inf')
best_path = str(checkpoints_dir('espcn_real_best.pt'))
os.makedirs(os.path.dirname(best_path), exist_ok=True)

for epoch in range(100):
    model.train()
    tot=0
    for lr,hr in train_loader:
        lr,hr = lr.to(device), hr.to(device)
        opt.zero_grad()
        out = model(lr)
        loss = criterion(out, hr)
        loss.backward()
        opt.step()
        tot+=loss.item()
    train_loss = tot/len(train_loader)
    sched.step()
    model.eval()
    vloss=0
    with torch.no_grad():
        for lr,hr in val_loader:
            lr,hr = lr.to(device), hr.to(device)
            vloss += criterion(model(lr), hr).item()
    vloss/=len(val_loader)
    print(f'Epoch {epoch+1} train {train_loss:.6f} val {vloss:.6f} lr {sched.get_last_lr()[0]:.2e}')
    if vloss < best:
        best=vloss
        torch.save(model.state_dict(), best_path)
        print('Saved best')
    if epoch>15 and vloss<0.008:
        print('Early stop')
        break
print('Done best', best)
