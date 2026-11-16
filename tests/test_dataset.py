"""
Tests for data/dataset.py — PatchDataset.

Covers: length, item shapes, degrade_on_the_fly, file_names filtering,
HR/LR pairing, and patch extraction.
"""

import os
import numpy as np
import pytest
from PIL import Image

from data.dataset import PatchDataset


class TestPatchDatasetInit:
    """Tests for PatchDataset initialization."""

    def test_requires_lr_and_hr_when_not_degrading(self, tmp_path):
        hr_dir = tmp_path / "HR"
        hr_dir.mkdir()
        with pytest.raises(ValueError, match="lr_dir and hr_dir are required"):
            PatchDataset(lr_dir=None, hr_dir=str(hr_dir), scale=2,
                         patch_size=32)

    def test_requires_hr_when_degrading(self, tmp_path):
        lr_dir = tmp_path / "LR"
        lr_dir.mkdir()
        with pytest.raises(ValueError, match="hr_dir is required"):
            PatchDataset(lr_dir=str(lr_dir), hr_dir=None, scale=2,
                         patch_size=32, degrade_on_the_fly=True)

    def test_loads_pairs(self, hr_lr_pair_paths):
        hr_dir, lr_dir = hr_lr_pair_paths
        ds = PatchDataset(lr_dir=lr_dir, hr_dir=hr_dir, scale=2,
                          patch_size=16, patches_per_image=5)
        assert len(ds) == 1 * 5  # 1 image * 5 patches

    def test_loads_multiple_pairs(self, multiple_hr_lr_pairs):
        hr_dir, lr_dir, names = multiple_hr_lr_pairs
        ds = PatchDataset(lr_dir=lr_dir, hr_dir=hr_dir, scale=4,
                          patch_size=16, patches_per_image=3)
        assert len(ds) == 3 * 3  # 3 images * 3 patches


class TestPatchDatasetGetItem:
    """Tests for PatchDataset __getitem__."""

    def test_returns_tensors(self, hr_lr_pair_paths):
        hr_dir, lr_dir = hr_lr_pair_paths
        ds = PatchDataset(lr_dir=lr_dir, hr_dir=hr_dir, scale=2,
                          patch_size=16, patches_per_image=1,
                          upsample_lr=True)
        lr, hr = ds[0]
        # upsample_lr=True -> LR patch is upsampled to HR patch size
        assert lr.shape == (3, 32, 32)
        assert hr.shape == (3, 32, 32)  # patch_size * scale

    def test_values_in_range(self, hr_lr_pair_paths):
        hr_dir, lr_dir = hr_lr_pair_paths
        ds = PatchDataset(lr_dir=lr_dir, hr_dir=hr_dir, scale=2,
                          patch_size=16, patches_per_image=1)
        lr, hr = ds[0]
        assert lr.min() >= 0.0 and lr.max() <= 1.0
        assert hr.min() >= 0.0 and hr.max() <= 1.0

    def test_different_patches(self, multiple_hr_lr_pairs):
        hr_dir, lr_dir, names = multiple_hr_lr_pairs
        ds = PatchDataset(lr_dir=lr_dir, hr_dir=hr_dir, scale=4,
                          patch_size=16, patches_per_image=20)
        # With enough patches, at least two should differ
        patches = [ds[i][0] for i in range(min(20, len(ds)))]
        assert any(not torch_equal(patches[0], p) for p in patches[1:]), \
            "Expected varied patches"


def torch_equal(a, b):
    return torch_allclose(a, b)


def torch_allclose(a, b, atol=1e-6):
    import torch
    return torch.allclose(a, b, atol=atol)


class TestPatchDatasetDegradeOnTheFly:
    """Tests for on-the-fly degradation mode."""

    def test_degrade_on_the_fly_returns_tensors(self, tmp_path):
        hr_dir = tmp_path / "HR"
        hr_dir.mkdir()
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(img).save(hr_dir / "img_01.png")

        ds = PatchDataset(lr_dir=None, hr_dir=str(hr_dir), scale=2,
                          patch_size=16, patches_per_image=1,
                          degrade_on_the_fly=True)
        lr, hr = ds[0]
        assert lr.shape == (3, 32, 32)   # upsampled to HR patch size
        assert hr.shape == (3, 32, 32)
        assert lr.min() >= 0.0 and lr.max() <= 1.0
        assert hr.min() >= 0.0 and hr.max() <= 1.0

    def test_degrade_on_the_fly_file_names_filter(self, tmp_path):
        hr_dir = tmp_path / "HR"
        hr_dir.mkdir()
        for name in ["img_01", "img_02", "img_03"]:
            img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(img).save(hr_dir / f"{name}.png")

        ds = PatchDataset(lr_dir=None, hr_dir=str(hr_dir), scale=2,
                          patch_size=16, patches_per_image=1,
                          degrade_on_the_fly=True,
                          file_names={"img_01", "img_03"})
        assert len(ds) == 2 * 1  # only 2 images


class TestFindHR:
    """Tests for _find_hr helper."""

    def test_finds_hr_by_basename(self, tmp_path):
        hr_dir = tmp_path / "HR"
        hr_dir.mkdir()
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(img).save(hr_dir / "foo.png")

        ds = PatchDataset.__new__(PatchDataset)
        ds.hr_dir = str(hr_dir)
        result = ds._find_hr("foo")
        assert result is not None
        assert result.endswith("foo.png")

    def test_returns_none_when_missing(self, tmp_path):
        hr_dir = tmp_path / "HR"
        hr_dir.mkdir()

        ds = PatchDataset.__new__(PatchDataset)
        ds.hr_dir = str(hr_dir)
        result = ds._find_hr("nonexistent")
        assert result is None
