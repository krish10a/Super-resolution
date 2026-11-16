"""
Security utilities for the SISR API.

- File-type validation (extension + actual image content via PIL verify)
- Upload size limit (default 10 MB)
- Image dimension limit (default 2048 x 2048)
- Safe temp-file handling with cleanup
- CORS origin allow-list (localhost:5173 frontend dev server)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, UnidentifiedImageError
from starlette.datastructures import UploadFile

# --------------------------------------------------------------------------- limits
MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_IMAGE_DIMENSION = 2048             # max width or height in pixels
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Frontend dev origin (Vite default). Allow both localhost and 127.0.0.1
# forms since browsers treat them as distinct origins for CORS.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# --------------------------------------------------------------------------- validation


class SecurityError(ValueError):
    """Raised when an upload fails a security/validation check."""


def validate_extension(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise SecurityError(
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )


def validate_image_content(data: bytes) -> Tuple[int, int]:
    """Verify the bytes are a real, decodable image. Returns (width, height).

    Uses PIL verify() then re-opens to get dimensions — verify() can leave
    the file in an unusable state, so we open a fresh copy for sizing.
    """
    try:
        img = Image.open(__import__("io").BytesIO(data))
        img.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise SecurityError("Upload is not a valid image file.") from exc

    # Re-open for dimension read (verify() may have consumed the stream).
    try:
        img = Image.open(__import__("io").BytesIO(data))
        width, height = img.size
    except Exception as exc:  # noqa: BLE001
        raise SecurityError("Unable to read image dimensions.") from exc

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise SecurityError(
            f"Image too large: {width}x{height}. "
            f"Max dimension is {MAX_IMAGE_DIMENSION}px."
        )
    if width == 0 or height == 0:
        raise SecurityError("Image has zero dimensions.")
    return width, height


def validate_upload(data: bytes, filename: str) -> Tuple[int, int]:
    """Run all upload checks: extension, content, size, dimensions."""
    validate_extension(filename)
    if len(data) > MAX_UPLOAD_BYTES:
        raise SecurityError(
            f"File too large: {len(data) / (1024*1024):.1f} MB. "
            f"Max is {MAX_UPLOAD_BYTES / (1024*1024):.0f} MB."
        )
    return validate_image_content(data)


def save_upload_to_temp(data: bytes, filename: str, cleanup: bool = True) -> str:
    """Write upload bytes to a safe temp file and return its path.

    The temp file is created in the system temp directory with a random name
    (no path traversal risk). Caller is responsible for removing it; pass
    cleanup=True to register deletion on process exit (best-effort).
    """
    suffix = Path(filename).suffix.lower()
    fd, path = tempfile.mkstemp(prefix="sisr_upload_", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except Exception:
        os.unlink(path)
        raise
    if cleanup:
        try:
            import atexit
            atexit.register(lambda p=path: os.unlink(p) if os.path.exists(p) else None)
        except Exception:
            pass
    return path


def remove_file(path: str) -> None:
    """Best-effort removal of a temp file."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


def is_allowed_origin(origin: Optional[str]) -> bool:
    return origin in ALLOWED_ORIGINS
