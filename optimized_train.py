import os, torch
from utils.cpu_affinity import set_cpu_threads
from data.dataset import PatchDataset
from torch.utils.data import DataLoader
from models.srcnn import SRCNN
from losses.mse_loss import MSELoss
from train.trainer import Trainer

set_cpu_threads()
device = 'cpu'
print('Device:', device)

os.system('python data/download_sets.py')
os.system('python data/prepare_pairs.py')

from config.paths import pairs_dir, checkpoints_dir
lr_dir = pairs_dir('Set5_LR_x2')
hr_dir = pairs_dir('Set5_HR')
dataset = PatchDataset(lr_dir, hr_dir, scale=2, patch_size=48)
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

model = SRCNN()
loss_fn = MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

trainer = Trainer(model, loss_fn, opt, device)
best_loss = float('inf')
best_path = str(checkpoints_dir('srcnn_best.pt'))

for epoch in range(50):
    loss = trainer.train_one_epoch(loader, grad_accum=1)
    scheduler.step()
    print(f'Epoch {epoch+1}/50 loss {loss:.6f} lr {scheduler.get_last_lr()[0]:.2e}')
    if loss < best_loss:
        best_loss = loss
        trainer.save(best_path)
        print('  -> new best, saved')
    # early stop if plateau
    if epoch > 10 and loss < 0.005:
        print('Early stop: loss plateau')
        break

print('Training complete, best loss', best_loss)
