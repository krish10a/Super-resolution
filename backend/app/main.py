"""
SISR Surveillance Super-Resolution Studio — FastAPI backend.

Endpoints:
  GET    /api/v1/health
  GET    /api/v1/models
  GET    /api/v1/models/{model_id}
  POST   /api/v1/super-resolve
  POST   /api/v1/compare
  POST   /api/v1/evaluate

Real PyTorch inference through existing trained checkpoints (E1-E8).
No fake predictions or metrics. PSNR/SSIM/LPIPS only with ground truth.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import sys
from typing import List, Optional

import torch
import torchvision.transforms.functional as TTF
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image as PILImage

# Ensure project root and backend/app are importable regardless of CWD.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from backend.app.registry import ModelRegistry, DEVICE  # noqa: E402
from backend.app.inference import InferenceService  # noqa: E402
from backend.app.security import (  # noqa: E402
    SecurityError, validate_upload, save_upload_to_temp, remove_file,
    ALLOWED_ORIGINS,
)
from metrics.psnr_ssim import compute_psnr_ssim  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sisr.api")

app = FastAPI(title="SISR Surveillance Super-Resolution Studio", version="1.0.0")

# CORS for the Vite dev frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- singletons
registry = ModelRegistry()
inference = InferenceService(registry)

# --------------------------------------------------------------------------- Pydantic schemas
class ModelItem(BaseModel):
    model_id: str
    architecture: str
    scale: int
    loss: str
    checkpoint: str
    available: bool
    param_count: Optional[int] = None
    device: str
    load_error: Optional[str] = None
    display_name: str


class HealthResponse(BaseModel):
    status: str
    torch_version: str
    device: str
    cuda_available: bool
    models_loaded: int
    models_available: int


class SuperResolveResponse(BaseModel):
    model_id: str
    architecture: str
    scale: int
    input_size: List[int]       # [w, h]
    output_size: List[int]      # [w, h]
    latency_ms: float
    device: str
    checkpoint: str
    pipeline: str
    image_base64: str           # PNG, base64
    lr_base64: Optional[str] = None  # Processed/Degraded LR image
    hr_base64: Optional[str] = None  # Original HR ground-truth (if simulate_lr=True)


class CompareItem(BaseModel):
    model_id: str
    architecture: str
    scale: int
    checkpoint: str
    input_size: List[int]
    output_size: List[int]
    latency_ms: float
    device: str
    pipeline: str
    image_base64: str


class CompareResponse(BaseModel):
    input_size: List[int]
    scale: int
    models: List[CompareItem]


class EvaluateResponse(BaseModel):
    psnr: float
    ssim: float
    y_channel: bool
    scale: int
    image_count: int
    metric: str = "PSNR/SSIM (BT.601 Y, border shave=scale)"


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class BicubicResponse(BaseModel):
    scale: int
    input_size: List[int]       # [w, h]
    output_size: List[int]      # [w, h]
    device: str
    pipeline: str = "bicubic"
    image_base64: str           # PNG, base64


# --------------------------------------------------------------------------- helpers
def _model_item(info) -> ModelItem:
    return ModelItem(
        model_id=info.model_id,
        architecture=info.architecture,
        scale=info.scale,
        loss=info.loss,
        checkpoint=info.checkpoint,
        available=info.available,
        param_count=info.param_count,
        device=info.device,
        load_error=info.load_error,
        display_name=info.display_name,
    )


def _error(message: str, detail: Optional[str] = None, code: int = status.HTTP_400_BAD_REQUEST):
    logger.warning("API error %s: %s (%s)", code, message, detail)
    return JSONResponse(status_code=code, content={"error": message, "detail": detail})


def _read_upload(file: UploadFile) -> bytes:
    data = file.file.read()
    file.file.seek(0)
    return data


# --------------------------------------------------------------------------- endpoints
@app.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
def health():
    loaded = len(registry._models)
    available = sum(1 for m in registry.list_models() if m.checkpoint_path.is_file())
    return HealthResponse(
        status="ok",
        torch_version=torch.__version__,
        device=str(DEVICE),
        cuda_available=torch.cuda.is_available(),
        models_loaded=loaded,
        models_available=available,
    )


@app.get("/api/v1/models", response_model=List[ModelItem], tags=["models"])
def list_models():
    return [_model_item(m) for m in registry.list_models()]


@app.get("/api/v1/models/{model_id}", response_model=ModelItem, tags=["models"])
def get_model(model_id: str):
    info = registry.get_info(model_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown model id: {model_id}")
    return _model_item(info)


@app.post("/api/v1/super-resolve", response_model=SuperResolveResponse, tags=["inference"])
async def super_resolve(
    file: UploadFile = File(..., description="LR image (jpg/jpeg/png/webp)"),
    model: str = Form(..., description="Model id, e.g. srcnn_2x_mse"),
    simulate_lr: bool = Form(False, description="Downscale/degrade HR into realistic CCTV LR before super-resolving"),
):
    try:
        data = await file.read()
        validate_upload(data, file.filename or "upload")
    except SecurityError as exc:
        return _error(str(exc), code=status.HTTP_400_BAD_REQUEST)
    except Exception as exc:  # noqa: BLE001
        return _error("Invalid upload", detail=str(exc), code=status.HTTP_400_BAD_REQUEST)

    try:
        result = inference.infer(model, data, simulate_lr=simulate_lr)
    except ValueError as exc:
        return _error(str(exc), code=status.HTTP_404_NOT_FOUND if "Unknown" in str(exc) else status.HTTP_400_BAD_REQUEST)
    except RuntimeError as exc:
        return _error("Model load failure", detail=str(exc), code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as exc:  # noqa: BLE001
        return _error("Inference failed", detail=str(exc), code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return SuperResolveResponse(
        model_id=result.model_id,
        architecture=result.architecture,
        scale=result.scale,
        input_size=[result.input_size[0], result.input_size[1]],
        output_size=[result.output_size[0], result.output_size[1]],
        latency_ms=result.latency_ms,
        device=result.device,
        checkpoint=result.checkpoint,
        pipeline=result.pipeline,
        image_base64=base64.b64encode(result.image_bytes).decode("ascii"),
        lr_base64=base64.b64encode(result.lr_image_bytes).decode("ascii") if result.lr_image_bytes else None,
        hr_base64=base64.b64encode(result.hr_ground_truth_bytes).decode("ascii") if result.hr_ground_truth_bytes else None,
    )


@app.post("/api/v1/compare", response_model=CompareResponse, tags=["inference"])
async def compare(
    file: UploadFile = File(..., description="LR image (jpg/jpeg/png/webp)"),
    model_ids: str = Form(..., description="Comma-separated model ids (same scale only)"),
):
    try:
        data = await file.read()
        validate_upload(data, file.filename or "upload")
    except SecurityError as exc:
        return _error(str(exc), code=status.HTTP_400_BAD_REQUEST)
    except Exception as exc:  # noqa: BLE001
        return _error("Invalid upload", detail=str(exc), code=status.HTTP_400_BAD_REQUEST)

    ids = [m.strip() for m in model_ids.split(",") if m.strip()]
    if len(ids) < 2:
        return _error("Provide at least two model ids", code=status.HTTP_400_BAD_REQUEST)
    infos = []
    for mid in ids:
        info = registry.get_info(mid)
        if info is None:
            return _error(f"Unknown model id: {mid}", code=status.HTTP_404_NOT_FOUND)
        if not info.checkpoint_path.is_file():
            return _error(f"Model unavailable: {mid}", code=status.HTTP_400_BAD_REQUEST)
        infos.append(info)

    scales = {i.scale for i in infos}
    if len(scales) != 1:
        return _error(
            "Compare requires models of the same scale",
            detail=f"Requested scales: {sorted(scales)}",
            code=status.HTTP_400_BAD_REQUEST,
        )
    scale = infos[0].scale

    try:
        img = PILImage.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        return _error("Unable to decode image", detail=str(exc), code=status.HTTP_400_BAD_REQUEST)
    img = img.convert("RGB")

    items: List[CompareItem] = []
    for info in infos:
        try:
            result = inference.infer(info.model_id, data)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Compare inference failed for %s", info.model_id)
            continue
        items.append(CompareItem(
            model_id=result.model_id,
            architecture=result.architecture,
            scale=result.scale,
            checkpoint=result.checkpoint,
            input_size=[result.input_size[0], result.input_size[1]],
            output_size=[result.output_size[0], result.output_size[1]],
            latency_ms=result.latency_ms,
            device=result.device,
            pipeline=result.pipeline,
            image_base64=base64.b64encode(result.image_bytes).decode("ascii"),
        ))

    if not items:
        return _error("All model inferences failed", code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return CompareResponse(
        input_size=[img.size[0], img.size[1]],
        scale=scale,
        models=items,
    )


@app.post("/api/v1/bicubic", response_model=BicubicResponse, tags=["inference"])
async def bicubic_baseline(
    file: UploadFile = File(..., description="LR image (jpg/jpeg/png/webp)"),
    scale: int = Form(2, description="Upscale factor (2 or 4)")
):
    """Produce a bicubic-upsampled baseline (no neural network) for fair
    visual comparison with neural model outputs."""
    try:
        data = await file.read()
        validate_upload(data, file.filename or "upload")
    except SecurityError as exc:
        return _error(str(exc), code=status.HTTP_400_BAD_REQUEST)
    except Exception as exc:  # noqa: BLE001
        return _error("Invalid upload", detail=str(exc), code=status.HTTP_400_BAD_REQUEST)

    if scale not in (2, 4):
        return _error("Scale must be 2 or 4", code=status.HTTP_400_BAD_REQUEST)

    try:
        img = PILImage.open(io.BytesIO(data))
        img.load()
        input_w, input_h = img.size
    except Exception as exc:
        return _error("Unable to decode image", detail=str(exc), code=status.HTTP_400_BAD_REQUEST)

    try:
        result_bytes = inference.bicubic_baseline(data, scale)
        out_img = PILImage.open(io.BytesIO(result_bytes))
        out_w, out_h = out_img.size
    except Exception as exc:  # noqa: BLE001
        return _error("Bicubic upscale failed", detail=str(exc), code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return BicubicResponse(
        scale=scale,
        input_size=[input_w, input_h],
        output_size=[out_w, out_h],
        device=str(DEVICE),
        image_base64=base64.b64encode(result_bytes).decode("ascii"),
    )


@app.post("/api/v1/evaluate", response_model=EvaluateResponse, tags=["evaluation"])
async def evaluate(
    lr_file: UploadFile = File(..., description="LR input image"),
    hr_file: UploadFile = File(..., description="HR ground-truth image (required for metrics)"),
    model_id: Optional[str] = Form(None, description="Model id to run inference on LR"),
    scale: Optional[int] = Form(None, description="Override scale (default from model)"),
):
    """Compute PSNR/SSIM between an SR output and HR ground truth.

    The SR output is either produced by running `model_id` on `lr_file`, or
    supplied directly via `sr_file`. Metrics require HR ground truth — never
    computed on arbitrary uploads without it.
    """
    try:
        lr_data = await lr_file.read()
        hr_data = await hr_file.read()
        validate_upload(lr_data, lr_file.filename or "lr")
        validate_upload(hr_data, hr_file.filename or "hr")
    except SecurityError as exc:
        return _error(str(exc), code=status.HTTP_400_BAD_REQUEST)
    except Exception as exc:  # noqa: BLE001
        return _error("Invalid upload", detail=str(exc), code=status.HTTP_400_BAD_REQUEST)

    try:
        lr_img = PILImage.open(io.BytesIO(lr_data)).convert("RGB")
        hr_img = PILImage.open(io.BytesIO(hr_data)).convert("RGB")
        lr_img.load(); hr_img.load()
    except Exception as exc:
        return _error("Unable to decode images", detail=str(exc), code=status.HTTP_400_BAD_REQUEST)

    # Determine SR tensor by running the selected model on the LR input.
    eff_scale = scale
    if model_id:
        info = registry.get_info(model_id)
        if info is None:
            return _error(f"Unknown model id: {model_id}", code=status.HTTP_404_NOT_FOUND)
        if not info.checkpoint_path.is_file():
            return _error(f"Model unavailable: {model_id}", code=status.HTTP_400_BAD_REQUEST)
        eff_scale = info.scale
        try:
            result = inference.infer(model_id, lr_data)
            sr_img = PILImage.open(io.BytesIO(result.image_bytes)).convert("RGB")
            sr_t = TTF.to_tensor(sr_img).unsqueeze(0)
        except Exception as exc:  # noqa: BLE001
            return _error("SR inference failed", detail=str(exc), code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    else:
        return _error("model_id is required to generate SR for evaluation",
                      code=status.HTTP_400_BAD_REQUEST)

    if eff_scale is None:
        return _error("Scale could not be determined", code=status.HTTP_400_BAD_REQUEST)

    hr_t = TTF.to_tensor(hr_img).unsqueeze(0)

    # Match spatial extents.
    min_h = min(sr_t.shape[-2], hr_t.shape[-2])
    min_w = min(sr_t.shape[-1], hr_t.shape[-1])
    sr_c = sr_t[..., :min_h, :min_w]
    hr_c = hr_t[..., :min_h, :min_w]

    try:
        p, s = compute_psnr_ssim(sr_c.squeeze(0), hr_c.squeeze(0), scale=eff_scale, y_channel=True)
    except Exception as exc:  # noqa: BLE001
        return _error("Metric computation failed", detail=str(exc), code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return EvaluateResponse(
        psnr=p,
        ssim=s,
        y_channel=True,
        scale=eff_scale,
        image_count=1,
    )


@app.exception_handler(Exception)
async def unhandled(request, exc):  # noqa: ARG001
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# Lazy import for PIL.Image used in compare/evaluate (kept at top via function).
