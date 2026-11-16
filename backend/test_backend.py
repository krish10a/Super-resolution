"""
Backend tests for the SISR Surveillance Super-Resolution Studio.

Covers: health endpoint, model registry, model loading, checkpoint
compatibility, SRCNN / ESPCN inference at x2 / x4, output dimensions,
image validation, invalid model IDs, unsupported files, and
ground-truth evaluation.

Run with:  pytest backend/test_backend.py
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Ensure project root + backend/app are importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_APP_DIR = _PROJECT_ROOT / "backend" / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from app.main import app  # noqa: E402
from app.registry import ModelRegistry  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def registry():
    return ModelRegistry()


# --------------------------------------------------------------------------- helpers
def _make_png_bytes(size: tuple[int, int], color=(40, 90, 160)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _upload(file_bytes: bytes, filename: str = "test.png"):
    return {"file": (filename, io.BytesIO(file_bytes), "image/png")}


# --------------------------------------------------------------------------- health
def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "torch_version" in body
    assert body["device"] in ("cuda", "cpu")
    assert "cuda_available" in body
    assert isinstance(body["models_available"], int)
    assert body["models_available"] == 8


# --------------------------------------------------------------------------- model registry / listing
def test_list_models(client):
    r = client.get("/api/v1/models")
    assert r.status_code == 200
    models = r.json()
    assert len(models) == 8
    ids = {m["model_id"] for m in models}
    expected = {
        "srcnn_2x_mse", "srcnn_2x_mse_perceptual",
        "srcnn_4x_mse", "srcnn_4x_mse_perceptual",
        "espcn_2x_mse", "espcn_2x_mse_perceptual",
        "espcn_4x_mse", "espcn_4x_mse_perceptual",
    }
    assert ids == expected
    for m in models:
        assert m["architecture"] in ("srcnn", "espcn")
        assert m["scale"] in (2, 4)
        assert m["loss"] in ("mse", "mse_perceptual")
        assert m["checkpoint"].endswith(".pt")
        assert m["device"] in ("cuda", "cpu")


def test_model_detail(client):
    r = client.get("/api/v1/models/srcnn_2x_mse")
    assert r.status_code == 200
    m = r.json()
    assert m["model_id"] == "srcnn_2x_mse"
    assert m["architecture"] == "srcnn"
    assert m["scale"] == 2
    assert m["loss"] == "mse"
    assert m["display_name"] == "SRCNN 2x MSE"


def test_model_detail_unknown(client):
    r = client.get("/api/v1/models/does_not_exist")
    assert r.status_code == 404


def test_registry_loads_all_checkpoints(registry):
    """Every registered checkpoint loads strictly."""
    loaded = 0
    for mid in registry.available_model_ids():
        model = registry.load_model(mid)
        assert model is not None
        assert registry.get_model(mid) is model  # cached
        info = registry.get_info(mid)
        assert info.available is True
        assert info.param_count is not None and info.param_count > 0
        assert info.load_error is None
        loaded += 1
    assert loaded == 8


def test_registry_unknown_model(registry):
    with pytest.raises(KeyError):
        registry.load_model("no_such_model")


# --------------------------------------------------------------------------- super-resolve
@pytest.mark.parametrize("model_id,scale", [
    ("srcnn_2x_mse", 2),
    ("srcnn_4x_mse", 4),
    ("espcn_2x_mse", 2),
    ("espcn_4x_mse", 4),
])
def test_super_resolve_dimensions(client, model_id, scale):
    lr = 64
    files = _upload(_make_png_bytes((lr, lr)))
    data = {"model": model_id}
    r = client.post("/api/v1/super-resolve", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_id"] == model_id
    assert body["input_size"] == [lr, lr]
    assert body["output_size"] == [lr * scale, lr * scale]
    assert body["scale"] == scale
    assert body["device"] in ("cuda", "cpu")
    assert body["checkpoint"].endswith(".pt")
    assert body["pipeline"] in ("srcnn_bicubic_upsample", "espcn_pixelshuffle")
    assert body["latency_ms"] >= 0
    assert body["image_base64"]


def test_super_resolve_srcnn_pipeline(client):
    r = client.post(
        "/api/v1/super-resolve",
        files=_upload(_make_png_bytes((32, 32))),
        data={"model": "srcnn_4x_mse_perceptual"},
    )
    assert r.status_code == 200
    assert r.json()["pipeline"] == "srcnn_bicubic_upsample"
    assert r.json()["output_size"] == [128, 128]


def test_super_resolve_espcn_pipeline(client):
    r = client.post(
        "/api/v1/super-resolve",
        files=_upload(_make_png_bytes((32, 32))),
        data={"model": "espcn_4x_mse_perceptual"},
    )
    assert r.status_code == 200
    assert r.json()["pipeline"] == "espcn_pixelshuffle"
    assert r.json()["output_size"] == [128, 128]


def test_super_resolve_unknown_model(client):
    r = client.post(
        "/api/v1/super-resolve",
        files=_upload(_make_png_bytes((32, 32))),
        data={"model": "no_such_model"},
    )
    assert r.status_code == 404


def test_super_resolve_invalid_file_type(client):
    r = client.post(
        "/api/v1/super-resolve",
        files={"file": ("bad.txt", io.BytesIO(b"not an image"), "text/plain")},
        data={"model": "srcnn_2x_mse"},
    )
    assert r.status_code == 400
    assert "Unsupported file type" in r.json()["error"]


# --------------------------------------------------------------------------- compare
def test_compare_same_scale(client):
    files = _upload(_make_png_bytes((48, 48)))
    data = {"model_ids": "srcnn_2x_mse,espcn_2x_mse"}
    r = client.post("/api/v1/compare", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["input_size"] == [48, 48]
    assert body["scale"] == 2
    assert len(body["models"]) == 2
    for m in body["models"]:
        assert m["output_size"] == [96, 96]
        assert m["image_base64"]


def test_compare_mismatched_scale(client):
    files = _upload(_make_png_bytes((48, 48)))
    data = {"model_ids": "srcnn_2x_mse,espcn_4x_mse"}
    r = client.post("/api/v1/compare", files=files, data=data)
    assert r.status_code == 400
    assert "same scale" in r.json()["error"]


def test_compare_single_model(client):
    files = _upload(_make_png_bytes((48, 48)))
    data = {"model_ids": "srcnn_2x_mse"}
    r = client.post("/api/v1/compare", files=files, data=data)
    assert r.status_code == 400


def test_compare_unknown_model(client):
    files = _upload(_make_png_bytes((48, 48)))
    data = {"model_ids": "srcnn_2x_mse,no_such_model"}
    r = client.post("/api/v1/compare", files=files, data=data)
    assert r.status_code == 404


# --------------------------------------------------------------------------- evaluate
def test_evaluate_with_ground_truth(client):
    lr = _make_png_bytes((64, 64))
    hr = _make_png_bytes((256, 256))  # 4x of 64
    files = {
        "lr_file": ("lr.png", io.BytesIO(lr), "image/png"),
        "hr_file": ("hr.png", io.BytesIO(hr), "image/png"),
    }
    data = {"model_id": "espcn_4x_mse"}
    r = client.post("/api/v1/evaluate", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["psnr"], float)
    assert isinstance(body["ssim"], float)
    assert body["y_channel"] is True
    assert body["scale"] == 4
    assert body["image_count"] == 1
    assert "BT.601" in body["metric"]


def test_evaluate_requires_model_id(client):
    lr = _make_png_bytes((64, 64))
    hr = _make_png_bytes((256, 256))
    files = {
        "lr_file": ("lr.png", io.BytesIO(lr), "image/png"),
        "hr_file": ("hr.png", io.BytesIO(hr), "image/png"),
    }
    r = client.post("/api/v1/evaluate", files=files, data={})
    assert r.status_code == 400
    assert "model_id" in r.json()["error"]


def test_evaluate_unknown_model(client):
    lr = _make_png_bytes((64, 64))
    hr = _make_png_bytes((256, 256))
    files = {
        "lr_file": ("lr.png", io.BytesIO(lr), "image/png"),
        "hr_file": ("hr.png", io.BytesIO(hr), "image/png"),
    }
    data = {"model_id": "no_such_model"}
    r = client.post("/api/v1/evaluate", files=files, data=data)
    assert r.status_code == 404

