import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

from data.degradation import degrade


class PatchDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, scale, patch_size, upsample_lr=True,
                 degrade_on_the_fly=False, degradation_kwargs=None):
        self.scale = scale
        self.patch_size = patch_size
        self.upsample_lr = upsample_lr
        self.degrade_on_the_fly = degrade_on_the_fly
        self.degradation_kwargs = degradation_kwargs or {}

        if degrade_on_the_fly:
            # Generate LR from HR on the fly; hr_dir required, lr_dir ignored.
            if hr_dir is None:
                raise ValueError('hr_dir is required when degrade_on_the_fly=True')
            files = [f for f in os.listdir(hr_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.hr_files = [os.path.join(hr_dir, f) for f in files]
            self.lr_dir = None
            self.hr_dir = hr_dir
        else:
            # Standard paired LR/HR mode (existing behavior).
            if lr_dir is None or hr_dir is None:
                raise ValueError('lr_dir and hr_dir are required when '
                                 'degrade_on_the_fly=False')
            files = [f for f in os.listdir(lr_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.lr_files = [os.path.join(lr_dir, f) for f in files]
            self.hr_dir = hr_dir

    def __len__(self):
        base = self.hr_files if self.degrade_on_the_fly else self.lr_files
        return len(base) * 50

    def _find_hr(self, name):
        for ext in ['.png', '.jpg', '.jpeg']:
            cand = os.path.join(self.hr_dir, name + ext)
            if os.path.exists(cand):
                return cand
        return None

    def __getitem__(self, idx):
        if self.degrade_on_the_fly:
            hr_path = self.hr_files[idx % len(self.hr_files)]
            name = os.path.splitext(os.path.basename(hr_path))[0]
            hr = Image.open(hr_path).convert('RGB')
            w, h = hr.size
            if w < self.patch_size * self.scale or h < self.patch_size * self.scale:
                hr = hr.resize((max(w, self.patch_size * self.scale),
                                max(h, self.patch_size * self.scale)),
                               Image.BICUBIC)
                w, h = hr.size
            left = random.randint(0, w - self.patch_size * self.scale)
            top = random.randint(0, h - self.patch_size * self.scale)
            hr_c = hr.crop((left, top,
                            left + self.patch_size * self.scale,
                            top + self.patch_size * self.scale))
            # Generate LR from the HR patch via the degradation pipeline.
            hr_arr = np.array(hr_c)
            lr_arr = degrade(hr_arr, self.scale, **self.degradation_kwargs)
            lr_c = Image.fromarray(lr_arr)
            if self.upsample_lr:
                lr_c = lr_c.resize(hr_c.size, Image.BICUBIC)
            if random.random() > 0.5:
                lr_c = TF.hflip(lr_c)
                hr_c = TF.hflip(hr_c)
            lr_t = TF.to_tensor(lr_c)
            hr_t = TF.to_tensor(hr_c)
            return lr_t, hr_t
        else:
            f = self.lr_files[idx % len(self.lr_files)]
            lr = Image.open(f).convert('RGB')
            name = os.path.splitext(os.path.basename(f))[0]
            hr_path = self._find_hr(name)
            if hr_path is None:
                raise FileNotFoundError(f'HR image not found for {name}')
            hr = Image.open(hr_path).convert('RGB')
            w, h = lr.size
            if w < self.patch_size or h < self.patch_size:
                lr = lr.resize((max(w, self.patch_size), max(h, self.patch_size)),
                               Image.BICUBIC)
                hr = hr.resize((w * self.scale, h * self.scale), Image.BICUBIC)
                w, h = lr.size
            left = random.randint(0, w - self.patch_size)
            top = random.randint(0, h - self.patch_size)
            lr_c = lr.crop((left, top, left + self.patch_size, top + self.patch_size))
            hr_left = left * self.scale
            hr_top = top * self.scale
            hr_c = hr.crop((hr_left, hr_top,
                            hr_left + self.patch_size * self.scale,
                            hr_top + self.patch_size * self.scale))
            if random.random() > 0.5:
                lr_c = TF.hflip(lr_c)
                hr_c = TF.hflip(hr_c)
            if self.upsample_lr:
                lr_c = lr_c.resize(hr_c.size, Image.BICUBIC)
            lr_t = TF.to_tensor(lr_c)
            hr_t = TF.to_tensor(hr_c)
            return lr_t, hr_t
