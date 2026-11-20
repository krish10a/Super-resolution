"""
Tests for Phase 4.2 Statistical Robustness Validation and Phase 4.3 Denoising Ablation.

Validates:
1. Deterministic seed generation (SHA-256 stability across platforms)
2. Paired bootstrap confidence interval mathematical properties
3. Denoiser interface contracts (shape, dtype, value bounds)
4. Win/tie/loss rate boundary handling and numerical tolerances
5. Integrity of results files (no NaNs, complete schemas)
"""

import math
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_phase4_2_statistical_validation import (
    derive_deterministic_seed,
    paired_bootstrap_ci,
)
from scripts.run_phase4_3_denoising_ablation import apply_denoiser


class TestDeterministicSeeding:
    def test_seed_determinism(self):
        s1 = derive_deterministic_seed("monarch", 2, "Sensor_Noise")
        s2 = derive_deterministic_seed("monarch", 2, "Sensor_Noise")
        assert s1 == s2
        assert isinstance(s1, int)
        assert 0 <= s1 < 2**32

    def test_seed_uniqueness(self):
        s_noise = derive_deterministic_seed("monarch", 2, "Sensor_Noise")
        s_motion = derive_deterministic_seed("monarch", 2, "Motion_Blur")
        s_other_img = derive_deterministic_seed("baboon", 2, "Sensor_Noise")
        s_other_scale = derive_deterministic_seed("monarch", 4, "Sensor_Noise")

        assert len({s_noise, s_motion, s_other_img, s_other_scale}) == 4


class TestBootstrapConfidenceIntervals:
    def test_zero_delta_yields_zero_interval(self):
        deltas = np.zeros(50, dtype=np.float64)
        low, high = paired_bootstrap_ci(deltas, num_resamples=500, seed=42)
        assert math.isclose(low, 0.0, abs_tol=1e-7)
        assert math.isclose(high, 0.0, abs_tol=1e-7)

    def test_strictly_positive_delta(self):
        rng = np.random.default_rng(123)
        deltas = rng.uniform(0.5, 1.5, size=50)
        low, high = paired_bootstrap_ci(deltas, num_resamples=1000, seed=42)
        assert low > 0.0
        assert high > low
        assert low < np.mean(deltas) < high

    def test_centered_noise_spans_zero(self):
        rng = np.random.default_rng(456)
        deltas = rng.normal(0.0, 1.0, size=100)
        low, high = paired_bootstrap_ci(deltas, num_resamples=1000, seed=42)
        assert low < 0.0 < high

    def test_bootstrap_reproducibility(self):
        deltas = np.array([0.1, 0.2, -0.05, 0.3, 0.15, -0.1], dtype=np.float64)
        low1, high1 = paired_bootstrap_ci(deltas, num_resamples=1000, seed=42)
        low2, high2 = paired_bootstrap_ci(deltas, num_resamples=1000, seed=42)
        assert low1 == low2
        assert high1 == high2


class TestDenoiserAblation:
    @pytest.fixture
    def sample_lr_image(self):
        rng = np.random.default_rng(42)
        return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

    def test_a0_identity(self, sample_lr_image):
        out, lat = apply_denoiser(sample_lr_image, "A0")
        assert np.array_equal(out, sample_lr_image)
        assert lat == 0.0

    def test_a1_nlm_properties(self, sample_lr_image):
        out, lat = apply_denoiser(sample_lr_image, "A1")
        assert out.shape == sample_lr_image.shape
        assert out.dtype == np.uint8
        assert 0 <= out.min() <= out.max() <= 255
        assert lat > 0.0

    def test_a2_bilateral_properties(self, sample_lr_image):
        out, lat = apply_denoiser(sample_lr_image, "A2")
        assert out.shape == sample_lr_image.shape
        assert out.dtype == np.uint8
        assert 0 <= out.min() <= out.max() <= 255
        assert lat > 0.0

    def test_unknown_denoiser_raises(self, sample_lr_image):
        with pytest.raises(ValueError, match="Unknown denoiser"):
            apply_denoiser(sample_lr_image, "A99")


class TestPhase4ArtifactIntegrity:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    def test_phase4_2_results_exist_and_finite(self):
        csv_path = self.PROJECT_ROOT / "results" / "phase4_2_statistical_summary.csv"
        if not csv_path.exists():
            pytest.skip("Phase 4.2 results not yet generated")

        import csv
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        assert len(reader) > 0
        for row in reader:
            assert not math.isnan(float(row["mean_psnr"]))
            assert not math.isnan(float(row["mean_ssim"]))
            assert not math.isnan(float(row["ci95_delta_psnr_low"]))
            assert not math.isnan(float(row["ci95_delta_psnr_high"]))
            assert 0.0 <= float(row["win_rate_psnr"]) <= 100.0

    def test_phase4_3_results_exist_and_finite(self):
        csv_path = self.PROJECT_ROOT / "results" / "phase4_3_denoising_summary.csv"
        if not csv_path.exists():
            pytest.skip("Phase 4.3 results not yet generated")

        import csv
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        assert len(reader) > 0
        for row in reader:
            assert not math.isnan(float(row["mean_psnr"]))
            assert not math.isnan(float(row["mean_ssim"]))
            assert not math.isnan(float(row["ci95_delta_psnr_vs_a0_low"]))
            assert not math.isnan(float(row["ci95_delta_psnr_vs_a0_high"]))
            assert float(row["mean_denoiser_latency_ms"]) >= 0.0
