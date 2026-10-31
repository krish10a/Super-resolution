import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.paths import pairs_real_dir, checkpoints_dir

from data.dataset import PatchDataset
from torch.utils.data import DataLoader
from models.srcnn import SRCNN
import torch.nn as nn

device = 'cpu'
print('Device:', device)

lr_dir = pairs_real_dir('Set5_real_LR_x2')
hr_dir = pairs_real_dir('Set5_real_HR')
val_lr_dir = pairs_real_dir('Set14_real_LR_x2')
val_hr_dir = pairs_real_dir('Set14_real_HR')

train_dataset = PatchDataset(lr_dir, hr_dir, scale=2, patch_size=32, upsample_lr=True)
val_dataset = PatchDataset(val_lr_dir, val_hr_dir, scale=2, patch_size=32, upsample_lr=True)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

model = SRCNN()
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

best_loss = float('inf')
best_path = str(checkpoints_dir('srcnn_real_best.pt'))
os.makedirs(os.path.dirname(best_path), exist_ok=True)

for epoch in range(100):
    model.train()
    total = 0
    for lr, hr in train_loader:
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        out = model(lr)
        loss = criterion(out, hr)
        loss.backward()
        optimizer.step()
        total += loss.item()
    train_loss = total / len(train_loader)
    scheduler.step()
    # validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            out = model(lr)
            loss = criterion(out, hr)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f'Epoch {epoch+1} train {train_loss:.6f} val {val_loss:.6f} lr {scheduler.get_last_lr()[0]:.2e}')
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), best_path)
        print('Saved best')
    if epoch > 15 and val_loss < 0.005:
        print('Early stop')
        break

print('Done best loss', best_loss)
