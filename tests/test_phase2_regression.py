"""
Phase 2 Regression Test Suite: Research Integrity, Metric Invariants, Dataset Manifest,
and Degradation Determinism.
"""

import os
import json
import numpy as np
import pytest
import torch
from pathlib import Path
from PIL import Image

from metrics.psnr_ssim import compute_psnr_ssim, rgb_to_y_cbcr, psnr
from data.degradation import (
    downscale_bicubic, gaussian_blur, anisotropic_gaussian_blur,
    motion_blur, gaussian_noise, poisson_noise, jpeg_compress, degrade
)
from data.dataset import ManifestPatchDataset, FullImageValDataset
from models.srcnn import SRCNN
from models.espcn import ESPCN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "dataset_manifest.json"


# ---------------------------------------------------------------------------
# 1. Dataset Manifest & Integrity Tests
# ---------------------------------------------------------------------------
class TestDatasetManifestIntegrity:
    def test_manifest_file_exists(self):
        assert MANIFEST_PATH.exists(), "Manifest file data/dataset_manifest.json missing."

    def test_manifest_dataset_counts(self):
        with open(MANIFEST_PATH, "r") as f:
            m = json.load(f)
        summary = m["summary"]
        assert summary["Set5"]["count"] == 5, f"Set5 must have exactly 5 images, got {summary['Set5']['count']}"
        assert summary["Set14"]["count"] == 14, f"Set14 must have exactly 14 images, got {summary['Set14']['count']}"
        assert summary["BSD100"]["count"] == 100, f"BSD100 must have exactly 100 images, got {summary['BSD100']['count']}"
        assert summary["T91"]["count"] == 91, f"T91 must have exactly 91 images, got {summary['T91']['count']}"

    def test_t91_split_isolation(self):
        with open(MANIFEST_PATH, "r") as f:
            m = json.load(f)
        train_ids = {e["image_id"] for e in m["entries"] if e["dataset"] == "T91" and e["split"] == "train"}
        val_ids = {e["image_id"] for e in m["entries"] if e["dataset"] == "T91" and e["split"] == "val"}
        test_ids = {e["image_id"] for e in m["entries"] if e["dataset"] != "T91"}

        assert len(train_ids) == 80, f"Expected 80 train images, got {len(train_ids)}"
        assert len(val_ids) == 11, f"Expected 11 val images, got {len(val_ids)}"
        assert train_ids.isdisjoint(val_ids), "Train and Val splits must have zero overlap."
        assert train_ids.isdisjoint(test_ids), "Train and Test sets must have zero overlap."
        assert val_ids.isdisjoint(test_ids), "Val and Test sets must have zero overlap."

    def test_hr_lr_dimension_correspondence(self):
        with open(MANIFEST_PATH, "r") as f:
            m = json.load(f)
        for e in m["entries"][:20]:  # Sample check first 20 entries
            hw, hh = e["hr_size"]
            l2w, l2h = e["lr_x2_size"]
            l4w, l4h = e["lr_x4_size"]

            assert hw % 4 == 0 and hh % 4 == 0, f"HR size {hw}x{hh} not divisible by 4"
            assert l2w == hw // 2 and l2h == hh // 2, f"LR_x2 size mismatch: HR={hw}x{hh}, LR2={l2w}x{l2h}"
            assert l4w == hw // 4 and l4h == hh // 4, f"LR_x4 size mismatch: HR={hw}x{hh}, LR4={l4w}x{l4h}"


# ---------------------------------------------------------------------------
# 2. Metric Invariants & Mathematical Parity Tests
# ---------------------------------------------------------------------------
class TestMetricMathematicalInvariants:
    def test_zero_error_gives_max_psnr_and_ssim(self):
        img = torch.rand(3, 64, 64)
        p, s = compute_psnr_ssim(img, img, scale=2, y_channel=True)
        assert p == 100.0, f"Expected 100.0 dB for zero error, got {p}"
        assert pytest.approx(s, rel=1e-4) == 1.0, f"Expected 1.0 SSIM for identical images, got {s}"

    def test_known_constant_error_psnr(self):
        # A known constant luminance difference c has MSE = c^2, PSNR = -10*log10(c^2)
        c = 0.05
        target = torch.ones(3, 64, 64) * 0.5
        pred = target + c
        # Full range Y: delta Y = c
        mse = c ** 2
        expected_psnr = -10.0 * np.log10(mse)
        # Note: rgb_to_y_cbcr scales by 219/255
        y_tar = rgb_to_y_cbcr(target)
        y_pred = rgb_to_y_cbcr(pred)
        actual_mse = torch.mean((y_pred - y_tar) ** 2).item()
        actual_psnr = psnr(y_pred, y_tar, max_val=1.0)

        expected_studio_mse = (c * 219.0 / 255.0) ** 2
        assert pytest.approx(actual_mse, rel=1e-3) == expected_studio_mse
        assert pytest.approx(actual_psnr, rel=1e-3) == -10.0 * np.log10(expected_studio_mse)

    def test_studio_swing_mathematical_shift_ratio(self):
        # Derivation: MSE_studio = (219/255)^2 * MSE_full
        # Shift in PSNR under fixed max_val=1.0 is exactly +1.3219 dB
        scale_ratio = 219.0 / 255.0
        shift_db = 10.0 * np.log10(1.0 / (scale_ratio ** 2))
        assert pytest.approx(shift_db, abs=1e-4) == 1.3219213

    def test_border_shaving_consistency(self):
        img_a = torch.rand(3, 32, 32)
        img_b = img_a.clone()
        # Corrupt borders only
        img_b[:, :2, :] = 0.0
        img_b[:, -2:, :] = 0.0
        img_b[:, :, :2] = 0.0
        img_b[:, :, -2:] = 0.0

        # With border shave = 2, corrupted borders are stripped -> zero error!
        p, s = compute_psnr_ssim(img_b, img_a, scale=2, y_channel=False)
        assert p == 100.0, "Border shave should exclude outer 2 pixels"
        assert pytest.approx(s, rel=1e-4) == 1.0


# ---------------------------------------------------------------------------
# 3. Degradation Reproducibility Tests
# ---------------------------------------------------------------------------
class TestDegradationReproducibility:
    def test_noise_seed_reproducibility(self):
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        noisy1 = gaussian_noise(img, sigma=15.0, seed=1234)
        noisy2 = gaussian_noise(img, sigma=15.0, seed=1234)
        np.testing.assert_array_equal(noisy1, noisy2, "Same seed must yield bitwise identical Gaussian noise")

        noisy3 = gaussian_noise(img, sigma=15.0, seed=5678)
        assert not np.array_equal(noisy1, noisy3), "Different seeds must yield different noise"

    def test_poisson_noise_reproducibility(self):
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        p1 = poisson_noise(img, scale=2.0, seed=42)
        p2 = poisson_noise(img, scale=2.0, seed=42)
        np.testing.assert_array_equal(p1, p2, "Same seed must yield bitwise identical Poisson noise")

    def test_full_pipeline_degrade_determinism(self):
        img = (np.random.RandomState(99).rand(128, 128, 3) * 255).astype(np.uint8)
        out1 = degrade(img, scale=2, blur_ksize=5, blur_sigma=1.2, noise_sigma=5.0,
                       jpeg_quality=70, seed=42)
        out2 = degrade(img, scale=2, blur_ksize=5, blur_sigma=1.2, noise_sigma=5.0,
                       jpeg_quality=70, seed=42)
        np.testing.assert_array_equal(out1, out2, "degrade() must be 100% deterministic with seed")

    def test_motion_and_anisotropic_blur_shapes(self):
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        m_blur = motion_blur(img, size=7, angle=45.0)
        assert m_blur.shape == img.shape

        a_blur = anisotropic_gaussian_blur(img, ksize=7, sigma_x=2.0, sigma_y=0.5, angle=30.0)
        assert a_blur.shape == img.shape


# ---------------------------------------------------------------------------
# 4. Manifest Datasets & Full-Image Validation Tests
# ---------------------------------------------------------------------------
class TestManifestDatasets:
    def test_manifest_patch_dataset(self):
        ds = ManifestPatchDataset(dataset="T91", split="train", scale=2, patch_size=32, patches_per_image=2)
        assert len(ds) == 80 * 2, f"Expected 160 patches, got {len(ds)}"
        lr_t, hr_t = ds[0]
        assert lr_t.shape == (3, 32, 32), f"Expected LR patch (3, 32, 32), got {lr_t.shape}"
        assert hr_t.shape == (3, 64, 64), f"Expected HR patch (3, 64, 64), got {hr_t.shape}"
        assert 0.0 <= lr_t.min() and lr_t.max() <= 1.0

    def test_full_image_val_dataset_determinism(self):
        ds = FullImageValDataset(dataset="T91", split="val", scale=2)
        assert len(ds) == 11, f"Expected 11 validation images, got {len(ds)}"
        lr_t, hr_t, img_id = ds[0]
        assert lr_t.shape[0] == 3 and hr_t.shape[0] == 3
        assert hr_t.shape[1] == lr_t.shape[1] * 2
        assert hr_t.shape[2] == lr_t.shape[2] * 2

        # Second access must return identical tensors (deterministic)
        lr_t2, hr_t2, _ = ds[0]
        torch.testing.assert_close(lr_t, lr_t2)
        torch.testing.assert_close(hr_t, hr_t2)


# ---------------------------------------------------------------------------
# 5. Model Architecture & Checkpoint Compatibility Tests
# ---------------------------------------------------------------------------
class TestModelArchitectureCompatibility:
    def test_espcn_variant_a_loads_e5(self):
        m = ESPCN(scale=2)
        ckpt_path = PROJECT_ROOT / "outputs" / "checkpoints" / "espcn_2x_mse.pt"
        if ckpt_path.exists():
            sd = torch.load(ckpt_path, map_location="cpu")
            m.load_state_dict(sd, strict=True)
            assert sum(p.numel() for p in m.parameters()) == 48716

    def test_espcn_variant_b_loads_real_best(self):
        m = ESPCN(scale=4, mid_channels=32)
        ckpt_path = PROJECT_ROOT / "outputs" / "checkpoints" / "espcn_real_best.pt"
        if ckpt_path.exists():
            sd = torch.load(ckpt_path, map_location="cpu")
            m.load_state_dict(sd, strict=True)
            assert sum(p.numel() for p in m.parameters()) == 37200
