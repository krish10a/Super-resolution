"""Shared pytest fixtures for the SISR test suite."""

import os
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tmp_image_path(tmp_path):
    """Create a small RGB test image and return its path."""
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    p = tmp_path / "test.png"
    Image.fromarray(img).save(p)
    return str(p)


@pytest.fixture
def constant_red_image():
    """A solid red 32x32 image."""
    return np.full((32, 32, 3), [255, 0, 0], dtype=np.uint8)


@pytest.fixture
def constant_white_image():
    """A solid white 32x32 image."""
    return np.full((32, 32, 3), [255, 255, 255], dtype=np.uint8)


@pytest.fixture
def constant_black_image():
    """A solid black 32x32 image."""
    return np.zeros((32, 32, 3), dtype=np.uint8)


@pytest.fixture
def gradient_image():
    """A smooth gradient image useful for resize testing."""
    h, w = 64, 64
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([x / w, y / h, (x + y) / (w + h)], axis=-1)
    return (img * 255).astype(np.uint8)


@pytest.fixture
def odd_dim_image():
    """An odd-dimension image (99x97) for edge-case resize testing."""
    return np.random.randint(0, 256, (99, 97, 3), dtype=np.uint8)


@pytest.fixture
def hr_lr_pair_paths(tmp_path):
    """Create a small HR/LR pair directory structure for dataset tests."""
    hr_dir = tmp_path / "HR"
    lr_dir = tmp_path / "LR_x2"
    hr_dir.mkdir()
    lr_dir.mkdir()

    img_hr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    img_lr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

    Image.fromarray(img_hr).save(hr_dir / "img_01.png")
    Image.fromarray(img_lr).save(lr_dir / "img_01.png")

    return str(hr_dir), str(lr_dir)


@pytest.fixture
def multiple_hr_lr_pairs(tmp_path):
    """Create multiple HR/LR pairs for dataset tests."""
    hr_dir = tmp_path / "HR"
    lr_dir = tmp_path / "LR_x4"
    hr_dir.mkdir()
    lr_dir.mkdir()

    names = ["img_01", "img_02", "img_03"]
    for name in names:
        img_hr = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        img_lr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(img_hr).save(hr_dir / f"{name}.png")
        Image.fromarray(img_lr).save(lr_dir / f"{name}.png")

    return str(hr_dir), str(lr_dir), names
