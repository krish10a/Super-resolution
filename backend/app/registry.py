"""
Central model registry for the SISR Surveillance Super-Resolution Studio.

Discovers and loads the trained checkpoints (E1-E8) and exposes model
availability, architecture, scale, and loss metadata to the API layer.

Model ID convention:  {arch}_{scale}x_{loss}
  e.g. "srcnn_2x_mse", "espcn_4x_mse_perceptual"

Checkpoints are raw PyTorch state_dicts (no training metadata), so loss
attribution is config-documented intent (see Step 1 provenance report).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch

from models.srcnn import SRCNN
from models.espcn import ESPCN

# ---------------------------------------------------------------------------
# Project-root-relative paths (this file lives at backend/app/registry.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model definitions (E1-E8). Checkpoint filenames match evaluate_all.py.
# ---------------------------------------------------------------------------
# id -> (architecture, scale, loss, checkpoint filename)
_MODEL_SPECS = {
    # SRCNN
    "srcnn_2x_mse": ("srcnn", 2, "mse", "srcnn_2x_mse.pt"),
    "srcnn_2x_mse_perceptual": ("srcnn", 2, "mse_perceptual", "srcnn_2x_mse_perceptual.pt"),
    "srcnn_4x_mse": ("srcnn", 4, "mse", "srcnn_4x_mse.pt"),
    "srcnn_4x_mse_perceptual": ("srcnn", 4, "mse_perceptual", "srcnn_4x_mse_perceptual.pt"),
    # ESPCN
    "espcn_2x_mse": ("espcn", 2, "mse", "espcn_2x_mse.pt"),
    "espcn_2x_mse_perceptual": ("espcn", 2, "mse_perceptual", "espcn_2x_mse_perceptual.pt"),
    "espcn_4x_mse": ("espcn", 4, "mse", "espcn_4x_mse.pt"),
    "espcn_4x_mse_perceptual": ("espcn", 4, "mse_perceptual", "espcn_4x_mse_perceptual.pt"),
}

# Architecture -> display name
_ARCH_DISPLAY = {
    "srcnn": "SRCNN",
    "espcn": "ESPCN",
}

# Loss -> display name
_LOSS_DISPLAY = {
    "mse": "MSE",
    "mse_perceptual": "MSE + Perceptual",
}


@dataclass
class ModelInfo:
    """Immutable metadata for a registered model variant."""

    model_id: str
    architecture: str
    scale: int
    loss: str
    checkpoint: str          # filename only
    checkpoint_path: Path    # resolved path
    available: bool = False
    param_count: Optional[int] = None
    load_error: Optional[str] = None
    device: str = str(DEVICE)

    @property
    def display_name(self) -> str:
        return f"{_ARCH_DISPLAY.get(self.architecture, self.architecture)} " \
               f"{self.scale}x {_LOSS_DISPLAY.get(self.loss, self.loss)}"


class ModelRegistry:
    """Central registry: discovers checkpoints, loads models strictly,
    and tracks availability. No silent exception swallowing."""

    def __init__(self, checkpoints_dir: Optional[Path] = None):
        self.checkpoints_dir = checkpoints_dir or CHECKPOINTS_DIR
        self.device = DEVICE
        self._models: Dict[str, torch.nn.Module] = {}
        self._infos: Dict[str, ModelInfo] = {}
        self._discover()

    # ------------------------------------------------------------------ discovery
    def _discover(self) -> None:
        """Build ModelInfo entries for every known model id, checking
        checkpoint file existence. Loading is deferred to load_model()."""
        for model_id, (arch, scale, loss, ckpt_name) in _MODEL_SPECS.items():
            ckpt_path = self.checkpoints_dir / ckpt_name
            exists = ckpt_path.is_file()
            self._infos[model_id] = ModelInfo(
                model_id=model_id,
                architecture=arch,
                scale=scale,
                loss=loss,
                checkpoint=ckpt_name,
                checkpoint_path=ckpt_path,
                available=exists,
                param_count=None,
                load_error=None if exists else f"checkpoint not found: {ckpt_path}",
                device=str(self.device),
            )

    # ------------------------------------------------------------------ loading
    def _instantiate(self, arch: str, scale: int) -> torch.nn.Module:
        if arch == "srcnn":
            return SRCNN()
        if arch == "espcn":
            return ESPCN(scale=scale)
        raise ValueError(f"Unknown architecture: {arch}")

    def load_model(self, model_id: str) -> torch.nn.Module:
        """Load a model by id (strict state_dict load). Raises KeyError if
        unknown id, RuntimeError if checkpoint missing or load fails.

        The loaded model is cached and moved to the selected device in eval
        mode. Failures are recorded in ModelInfo.load_error and availability
        is set to False — never silently swallowed."""
        if model_id not in self._infos:
            raise KeyError(f"Unknown model id: {model_id}")

        info = self._infos[model_id]
        if model_id in self._models:
            return self._models[model_id]

        if not info.checkpoint_path.is_file():
            msg = f"Checkpoint not found: {info.checkpoint_path}"
            info.available = False
            info.load_error = msg
            raise RuntimeError(msg)

        try:
            model = self._instantiate(info.architecture, info.scale)
            state = torch.load(info.checkpoint_path, map_location=self.device)
            # Strict load: architecture must match checkpoint exactly.
            model.load_state_dict(state, strict=True)
            model.to(self.device)
            model.eval()
            param_count = sum(p.numel() for p in model.parameters())
            self._models[model_id] = model
            info.available = True
            info.param_count = param_count
            info.load_error = None
            return model
        except Exception as exc:  # noqa: BLE001 - surface load failures explicitly
            msg = f"Failed to load checkpoint {info.checkpoint_path}: {exc}"
            info.available = False
            info.param_count = None
            info.load_error = msg
            raise RuntimeError(msg) from exc

    def get_model(self, model_id: str) -> Optional[torch.nn.Module]:
        """Return a previously loaded model, or None (does not raise)."""
        return self._models.get(model_id)

    def ensure_loaded(self, model_id: str) -> torch.nn.Module:
        """Load (if needed) and return the model. Raises on failure."""
        if model_id not in self._models:
            self.load_model(model_id)
        return self._models[model_id]

    # ------------------------------------------------------------------ listing
    def list_models(self) -> list[ModelInfo]:
        """Return metadata for all known models (loaded or not)."""
        return list(self._infos.values())

    def available_model_ids(self) -> list[str]:
        """IDs of models that are currently loadable (checkpoint exists)."""
        return [
            mid for mid, info in self._infos.items()
            if info.checkpoint_path.is_file()
        ]

    def is_available(self, model_id: str) -> bool:
        info = self._infos.get(model_id)
        return bool(info and info.checkpoint_path.is_file())

    def get_info(self, model_id: str) -> Optional[ModelInfo]:
        return self._infos.get(model_id)
