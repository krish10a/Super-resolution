"""
SISR inference service.

Implements the correct, model-specific pipelines:

* SRCNN:  LR upload -> bicubic pre-upsample (imresize_matlab) -> SRCNN -> SR
          (SRNN output is Tanh [-1,1], denormalized to [0,1] for display)
* ESPCN:  LR upload -> ESPCN -> PixelShuffle -> SR  (NO pre-upsample)

Both produce exactly x2 / x4 output dimensions. Runs under torch.no_grad()
in inference mode, returns PIL/PNG bytes plus metadata (dims, latency,
device, checkpoint).
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
from PIL import Image, ImageOps

# Make project-root `data` and `models` importable regardless of CWD.
import os, sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.degradation import degrade
from backend.app.registry import ModelRegistry, DEVICE


@dataclass
class InferenceResult:
    model_id: str
    architecture: str
    scale: int
    input_size: Tuple[int, int]     # (w, h) of the LR processed
    output_size: Tuple[int, int]    # (w, h) of the SR output
    device: str
    checkpoint: str
    latency_ms: float
    image_bytes: bytes              # PNG-encoded SR image
    pipeline: str                   # "srcnn_bicubic_upsample" or "espcn_pixelshuffle"
    lr_image_bytes: Optional[bytes] = None  # PNG-encoded LR image (especially when simulate_lr=True)
    hr_ground_truth_bytes: Optional[bytes] = None


class InferenceService:
    """Wraps ModelRegistry and runs real PyTorch inference through
    the existing trained checkpoints."""

    def __init__(self, registry: ModelRegistry | None = None):
        self.registry = registry or ModelRegistry()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _pil_to_np_rgb(img: Image.Image) -> np.ndarray:
        """PIL RGB -> HWC uint8 [0,255] numpy."""
        return np.asarray(img.convert("RGB"))

    @staticmethod
    def _np_to_tensor(img_np: np.ndarray) -> torch.Tensor:
        """HWC uint8 [0,255] -> CHW float32 [0,1] tensor (1,3,H,W)."""
        t = torch.from_numpy(img_np.astype(np.float32) / 255.0).permute(2, 0, 1)
        return t.unsqueeze(0)

    @staticmethod
    def _tensor_to_pil(img_t: torch.Tensor) -> Image.Image:
        """(1,3,H,W) or (3,H,W) float [0,1] -> PIL RGB."""
        t = img_t.detach().cpu().clamp(0.0, 1.0).squeeze(0)
        ndarray = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(ndarray, mode="RGB")

    # ------------------------------------------------------------------ pipelines
    def _srcnn_pipeline(self, model: torch.nn.Module, lr_img: Image.Image,
                        scale: int, device: torch.device) -> Image.Image:
        """SRCNN: bicubic pre-upsample then network forward.

        High-speed PIL bicubic used for pre-upsample (under 50ms vs 30s in pure numpy).
        SRCNN convs are same-size, so output dims == upsampled input dims == LR x scale exactly.
        Output is clamped to [0, 1] matching training supervision.
        """
        w, h = lr_img.size
        up_img = lr_img.resize((w * scale, h * scale), Image.BICUBIC)
        up_np = self._pil_to_np_rgb(up_img)
        x = self._np_to_tensor(up_np).to(device)
        with torch.inference_mode():
            out = model(x)
        # Model was trained directly against HR targets in [0, 1] range; clamp to [0, 1].
        out = out.clamp(0.0, 1.0)
        return self._tensor_to_pil(out)

    def _espcn_pipeline(self, model: torch.nn.Module, lr_np: np.ndarray,
                        scale: int, device: torch.device) -> Image.Image:
        """ESPCN: raw LR into network; PixelShuffle upscales internally.

        No bicubic pre-upsample. Output dims == LR x scale exactly.
        """
        x = self._np_to_tensor(lr_np).to(device)
        with torch.inference_mode():
            out = model(x)
        # ESPCN has no output activation -> clamp to [0, 1].
        out = out.clamp(0.0, 1.0)
        return self._tensor_to_pil(out)

    # ------------------------------------------------------------------ public API
    def infer(self, model_id: str, image_bytes: bytes, simulate_lr: bool = False) -> InferenceResult:
        """Run real inference for `model_id` on the uploaded image bytes.

        If `simulate_lr=True`:
          The uploaded image is treated as high-resolution ground truth, downsampled/degraded
          to realistic surveillance camera resolution (mimicking a distance CCTV sensor),
          and then super-resolved. Both the degraded LR and the super-resolved output are returned!
        """
        info = self.registry.get_info(model_id)
        if info is None:
            raise ValueError(f"Unknown model id: {model_id}")
        if not info.checkpoint_path.is_file():
            raise ValueError(f"Model unavailable (checkpoint missing): {model_id}")

        model = self.registry.ensure_loaded(model_id)  # may raise RuntimeError

        # Decode + validate the upload. Apply EXIF transpose for correct smartphone orientation.
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img = ImageOps.exif_transpose(img)
            img.load()  # force decode to catch corrupt files early
        except Exception as exc:
            raise ValueError(f"Unable to decode image: {exc}") from exc

        if img.mode != "RGB":
            img = img.convert("RGB")

        scale = info.scale
        device = self.registry.device
        hr_gt_bytes = None
        lr_bytes = None

        if simulate_lr:
            # Save original as HR ground truth
            hr_buf = io.BytesIO()
            img.save(hr_buf, format="PNG")
            hr_gt_bytes = hr_buf.getvalue()

            # Downsample to realistic CCTV resolution with mild degradation (mimics surveillance sensor)
            orig_w, orig_h = img.size
            # Modulo-crop HR so it divides cleanly by scale
            mod_w, mod_h = (orig_w // scale) * scale, (orig_h // scale) * scale
            if mod_w != orig_w or mod_h != orig_h:
                img = img.crop((0, 0, max(scale, mod_w), max(scale, mod_h)))

            hr_np = self._pil_to_np_rgb(img)
            lr_np = degrade(hr_np, scale=scale, blur_ksize=3, blur_sigma=0.8, noise_sigma=3.0, jpeg_quality=70)
            lr_img = Image.fromarray(lr_np, mode="RGB")
        else:
            # If the user uploaded a very large image (>800px) in direct mode, safely scale to reasonable LR
            # to prevent GPU out-of-memory and maintain responsive sub-second inference.
            max_dim = max(img.size)
            if max_dim > 720:
                scale_down = 720 / max_dim
                new_w = max(16, int(img.size[0] * scale_down))
                new_h = max(16, int(img.size[1] * scale_down))
                img = img.resize((new_w, new_h), Image.BICUBIC)

            lr_img = img
            lr_np = self._pil_to_np_rgb(lr_img)

        # Encode LR image for frontend comparison
        lr_buf = io.BytesIO()
        lr_img.save(lr_buf, format="PNG")
        lr_bytes = lr_buf.getvalue()

        input_w, input_h = lr_img.size

        start = time.perf_counter()
        try:
            if info.architecture == "srcnn":
                sr_img = self._srcnn_pipeline(model, lr_img, scale, device)
                pipeline = "srcnn_bicubic_upsample"
            elif info.architecture in ["espcn", "fsrcnn", "edsr"]:
                sr_img = self._espcn_pipeline(model, lr_np, scale, device)
                pipeline = f"{info.architecture}_direct"
            else:
                raise ValueError(f"Unsupported architecture: {info.architecture}")
        finally:
            # Free inference tensors promptly.
            if device.type == "cuda":
                torch.cuda.empty_cache()

        out_w, out_h = sr_img.size
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Encode SR image as PNG.
        buf = io.BytesIO()
        sr_img.save(buf, format="PNG")
        image_bytes_out = buf.getvalue()

        return InferenceResult(
            model_id=model_id,
            architecture=info.architecture,
            scale=scale,
            input_size=(input_w, input_h),
            output_size=(out_w, out_h),
            device=str(device),
            checkpoint=info.checkpoint,
            latency_ms=latency_ms,
            image_bytes=image_bytes_out,
            pipeline=pipeline,
            lr_image_bytes=lr_bytes,
            hr_ground_truth_bytes=hr_gt_bytes,
        )

    def bicubic_baseline(self, image_bytes: bytes, scale: int) -> bytes:
        """Produce a bicubic-upsampled baseline (no neural network) for
        fair visual comparison. Uses fast PIL bicubic (sub-50ms)."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img = ImageOps.exif_transpose(img)
            img.load()
        except Exception as exc:
            raise ValueError(f"Unable to decode image: {exc}") from exc
        img = img.convert("RGB")
        w, h = img.size
        sr_img = img.resize((w * scale, h * scale), Image.BICUBIC)
        buf = io.BytesIO()
        sr_img.save(buf, format="PNG")
        return buf.getvalue()
