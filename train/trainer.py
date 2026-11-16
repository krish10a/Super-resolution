import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, model, loss_fn, optimizer, device, scaler=None):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler

    def train_one_epoch(self, loader, grad_accum=1):
        self.model.train()
        total = 0
        for i, (lr, hr) in enumerate(tqdm(loader)):
            lr, hr = lr.to(self.device), hr.to(self.device)
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    out = self.model(lr)
                    loss = self.loss_fn(out, hr) / grad_accum
                self.scaler.scale(loss).backward()
                if (i+1) % grad_accum == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                out = self.model(lr)
                loss = self.loss_fn(out, hr) / grad_accum
                loss.backward()
                if (i+1) % grad_accum == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            total += loss.item() * grad_accum
        return total / len(loader)

    @torch.no_grad()
    def validate(self, loader, metric_fn, scale=1):
        self.model.eval()
        psnrs, ssims = [], []
        for lr, hr in loader:
            lr, hr = lr.to(self.device), hr.to(self.device)
            out = self.model(lr).squeeze(0)     # (1,3,H,W) -> (3,H,W)
            hr = hr.squeeze(0)                  # (1,3,H,W) -> (3,H,W)
            p, s = metric_fn(out, hr, scale=scale)
            psnrs.append(p)
            ssims.append(s)
        return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
