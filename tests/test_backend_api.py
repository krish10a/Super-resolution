import io
from PIL import Image
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_api_health():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["models_available"] >= 8

def test_api_models():
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    models = res.json()
    assert len(models) >= 8
    model_ids = [m["model_id"] for m in models]
    assert "srcnn_2x_mse" in model_ids
    assert "espcn_4x_mse" in model_ids

def test_api_super_resolve_espcn():
    im = Image.new("RGB", (32, 32), color="blue")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    
    res = client.post(
        "/api/v1/super-resolve",
        data={"model": "espcn_2x_mse"},
        files={"file": ("test.png", buf.getvalue(), "image/png")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["model_id"] == "espcn_2x_mse"
    assert data["input_size"] == [32, 32]
    assert data["output_size"] == [64, 64]
    assert len(data["image_base64"]) > 0

def test_api_super_resolve_srcnn():
    im = Image.new("RGB", (32, 32), color="green")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    
    res = client.post(
        "/api/v1/super-resolve",
        data={"model": "srcnn_2x_mse"},
        files={"file": ("test.png", buf.getvalue(), "image/png")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["model_id"] == "srcnn_2x_mse"
    assert data["output_size"] == [64, 64]

