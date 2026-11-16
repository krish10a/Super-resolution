import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

from data.degradation import degrade


class PatchDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, scale, patch_size, upsample_lr=True,
                 degrade_on_the_fly=False, degradation_kwargs=None, patches_per_image=10,
                 file_names=None):
        self.scale = scale
        self.patch_size = patch_size
        self.upsample_lr = upsample_lr
        self.degrade_on_the_fly = degrade_on_the_fly
        self.degradation_kwargs = degradation_kwargs or {}
        self.patches_per_image = patches_per_image

        self.cached_images = []

        if degrade_on_the_fly:
            if hr_dir is None:
                raise ValueError('hr_dir is required when degrade_on_the_fly=True')
            all_files = sorted([f for f in os.listdir(hr_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            files = all_files if file_names is None else \
                [f for f in all_files if os.path.splitext(f)[0] in file_names]
            self.hr_files = [os.path.join(hr_dir, f) for f in files]
            # Preload HR images into RAM
            for f in self.hr_files:
                img = Image.open(f).convert('RGB')
                self.cached_images.append((None, img))
        else:
            if lr_dir is None or hr_dir is None:
                raise ValueError('lr_dir and hr_dir are required when degrade_on_the_fly=False')
            all_files = sorted([f for f in os.listdir(lr_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            files = all_files if file_names is None else \
                [f for f in all_files if os.path.splitext(f)[0] in file_names]
            self.lr_files = [os.path.join(lr_dir, f) for f in files]
            self.hr_dir = hr_dir
            for f in self.lr_files:
                name = os.path.splitext(os.path.basename(f))[0]
                hr_path = self._find_hr(name)
                if hr_path is None:
                    continue
                lr_img = Image.open(f).convert('RGB')
                hr_img = Image.open(hr_path).convert('RGB')
                self.cached_images.append((lr_img, hr_img))

    def _find_hr(self, name):
        for ext in ['.png', '.jpg', '.jpeg']:
            cand = os.path.join(self.hr_dir, name + ext)
            if os.path.exists(cand):
                return cand
        return None

    def __len__(self):
        return len(self.cached_images) * self.patches_per_image

    def __getitem__(self, idx):
        pair_idx = idx % len(self.cached_images)
        lr_cached, hr_cached = self.cached_images[pair_idx]

        if self.degrade_on_the_fly:
            hr = hr_cached
            w, h = hr.size
            req_size = self.patch_size * self.scale
            if w < req_size or h < req_size:
                hr = hr.resize((max(w, req_size), max(h, req_size)), Image.BICUBIC)
                w, h = hr.size
            left = random.randint(0, w - req_size)
            top = random.randint(0, h - req_size)
            hr_c = hr.crop((left, top, left + req_size, top + req_size))

            hr_arr = np.array(hr_c)
            lr_arr = degrade(hr_arr, self.scale, **self.degradation_kwargs)
            lr_c = Image.fromarray(lr_arr)
            if self.upsample_lr:
                lr_c = lr_c.resize(hr_c.size, Image.BICUBIC)
            if random.random() > 0.5:
                lr_c = TF.hflip(lr_c)
                hr_c = TF.hflip(hr_c)
            return TF.to_tensor(lr_c), TF.to_tensor(hr_c)
        else:
            lr = lr_cached
            hr = hr_cached
            w, h = lr.size
            if w < self.patch_size or h < self.patch_size:
                lr = lr.resize((max(w, self.patch_size), max(h, self.patch_size)), Image.BICUBIC)
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
            return TF.to_tensor(lr_c), TF.to_tensor(hr_c)


# ---------------------------------------------------------------------------
# Canonical Manifest-Driven Datasets (Phase 2 Clean Train/Val Separation)
# ---------------------------------------------------------------------------
import json
from pathlib import Path

class ManifestPatchDataset(Dataset):
    """Clean training dataset driven by canonical dataset manifest.

    Extracts patches strictly from training images (e.g. 80 T91 train images).
    Guarantees zero overlap with held-out validation or official test benchmarks.
    """
    def __init__(self, manifest_path=None, dataset='T91', split='train', scale=2,
                 patch_size=32, upsample_lr=False, patches_per_image=10, seed=42):
        self.scale = scale
        self.patch_size = patch_size
        self.upsample_lr = upsample_lr
        self.patches_per_image = patches_per_image
        self.rng = random.Random(seed)

        if manifest_path is None:
            manifest_path = Path(__file__).resolve().parents[1] / "data" / "dataset_manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        project_root = Path(manifest_path).resolve().parents[1]
        self.entries = [
            e for e in manifest["entries"]
            if e["dataset"] == dataset and e["split"] == split
        ]
        if not self.entries:
            raise ValueError(f"No entries found for dataset='{dataset}', split='{split}' in {manifest_path}")

        self.cached_images = []
        for e in self.entries:
            hr_p = project_root / e["hr_path"]
            lr_p = project_root / e[f"lr_x{scale}_path"]
            hr_img = Image.open(hr_p).convert("RGB")
            lr_img = Image.open(lr_p).convert("RGB")
            self.cached_images.append((lr_img, hr_img, e["image_id"]))

    def __len__(self):
        return len(self.cached_images) * self.patches_per_image

    def __getitem__(self, idx):
        pair_idx = idx % len(self.cached_images)
        lr_img, hr_img, _ = self.cached_images[pair_idx]

        w, h = lr_img.size
        if w < self.patch_size or h < self.patch_size:
            lr_img = lr_img.resize((max(w, self.patch_size), max(h, self.patch_size)), Image.BICUBIC)
            hr_img = hr_img.resize((w * self.scale, h * self.scale), Image.BICUBIC)
            w, h = lr_img.size

        left = self.rng.randint(0, w - self.patch_size)
        top = self.rng.randint(0, h - self.patch_size)

        lr_c = lr_img.crop((left, top, left + self.patch_size, top + self.patch_size))
        hr_left, hr_top = left * self.scale, top * self.scale
        hr_c = hr_img.crop((hr_left, hr_top, hr_left + self.patch_size * self.scale, hr_top + self.patch_size * self.scale))

        # Data augmentations: horizontal flip, vertical flip, 90-degree rotations
        if self.rng.random() > 0.5:
            lr_c = TF.hflip(lr_c)
            hr_c = TF.hflip(hr_c)
        if self.rng.random() > 0.5:
            lr_c = TF.vflip(lr_c)
            hr_c = TF.vflip(hr_c)

        rot_choice = self.rng.choice([0, 90, 180, 270])
        if rot_choice > 0:
            lr_c = TF.rotate(lr_c, rot_choice)
            hr_c = TF.rotate(hr_c, rot_choice)

        if self.upsample_lr:
            lr_c = lr_c.resize(hr_c.size, Image.BICUBIC)

        return TF.to_tensor(lr_c), TF.to_tensor(hr_c)


class FullImageValDataset(Dataset):
    """Deterministic full-image validation dataset.

    Evaluates on held-out original images (e.g. 11 T91 validation images) without
    random patch sampling or flipping.
    """
    def __init__(self, manifest_path=None, dataset='T91', split='val', scale=2, upsample_lr=False):
        self.scale = scale
        self.upsample_lr = upsample_lr

        if manifest_path is None:
            manifest_path = Path(__file__).resolve().parents[1] / "data" / "dataset_manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        project_root = Path(manifest_path).resolve().parents[1]
        self.entries = [
            e for e in manifest["entries"]
            if e["dataset"] == dataset and e["split"] == split
        ]
        if not self.entries:
            raise ValueError(f"No validation entries found for dataset='{dataset}', split='{split}'")

        self.cached_images = []
        for e in self.entries:
            hr_p = project_root / e["hr_path"]
            lr_p = project_root / e[f"lr_x{scale}_path"]
            hr_img = Image.open(hr_p).convert("RGB")
            lr_img = Image.open(lr_p).convert("RGB")
            self.cached_images.append((lr_img, hr_img, e["image_id"]))

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, idx):
        lr_img, hr_img, image_id = self.cached_images[idx]
        if self.upsample_lr:
            lr_img = lr_img.resize(hr_img.size, Image.BICUBIC)
        return TF.to_tensor(lr_img), TF.to_tensor(hr_img), image_id
