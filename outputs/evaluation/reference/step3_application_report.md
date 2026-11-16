# Step 3 — SISR Surveillance Super-Resolution Studio: Application Report

**Project:** SISR Surveillance Super-Resolution Studio  
**Repository:** `C:\Users\ASUS\Desktop\cv`  
**Date:** 2026-11-01  
**Scope:** Step 3 only (React + FastAPI full-stack application)

---

## Summary

A complete full-stack super-resolution application was built that performs **real PyTorch inference through existing trained checkpoints** (E1–E8). No fake predictions, no placeholder images, no fabricated metrics. All metrics (PSNR/SSIM) are computed only against a supplied HR ground truth.

## Backend (FastAPI)

**Location:** `backend/app/` — `main.py`, `registry.py`, `inference.py`, `security.py`  
**Tests:** `backend/test_backend.py` — **21/21 passing**

### Model registry (8 variants, E1–E8)

| Model ID | Architecture | Scale | Loss | Checkpoint | Available |
|---|---|---|---|---|---|
| srcnn_2x_mse | SRCNN | 2× | MSE | srcnn_2x_mse.pt | ✅ |
| srcnn_2x_mse_perceptual | SRCNN | 2× | MSE+Perceptual | srcnn_2x_mse_perceptual.pt | ✅ |
| srcnn_4x_mse | SRCNN | 4× | MSE | srcnn_4x_mse.pt | ✅ |
| srcnn_4x_mse_perceptual | SRCNN | 4× | MSE+Perceptual | srcnn_4x_mse_perceptual.pt | ✅ |
| espcn_2x_mse | ESPCN | 2× | MSE | espcn_2x_mse.pt | ✅ |
| espcn_2x_mse_perceptual | ESPCN | 2× | MSE+Perceptual | espcn_2x_mse_perceptual_verified.pt | ✅ |
| espcn_4x_mse | ESPCN | 4× | MSE | espcn_4x_mse_verified.pt | ✅ |
| espcn_4x_mse_perceptual | ESPCN | 4× | MSE+Perceptual | espcn_4x_mse_perceptual_verified.pt | ✅ |

- Strict `load_state_dict(strict=True)` loading; failures mark a model unavailable (never silently swallowed).
- CPU/CUDA auto-detect (`torch.device("cuda" if available else "cpu")`).
- Checkpoints are raw state_dicts — loss attribution is config-documented (see Step 1 provenance report).

### Correct, model-specific inference pipelines

- **SRCNN:** LR upload → bicubic pre-upsample (`imresize_matlab`, MATLAB/BasicSR-compatible) → SRCNN → Tanh output mapped [−1,1] → [0,1]. Pipeline tag: `srcnn_bicubic_upsample`.
- **ESPCN:** LR upload → ESPCN → PixelShuffle → SR output (no pre-upsample). Pipeline tag: `espcn_pixelshuffle`.
- Both produce exactly ×2 / ×4 output dimensions. Runs under `torch.inference_mode()`, PIL/PNG encoding, CUDA cache cleanup.

### Endpoints (all verified live)

| Endpoint | Method | Status |
|---|---|---|
| `GET /api/v1/health` | status, torch version, device, CUDA, model counts | ✅ 200 |
| `GET /api/v1/models` | list of 8 models with availability, param counts | ✅ 200 |
| `GET /api/v1/models/{id}` | detailed model config | ✅ 200 / 404 unknown |
| `POST /api/v1/super-resolve` | file + model → SR image (base64 PNG) + dims + latency | ✅ 200 |
| `POST /api/v1/compare` | file + same-scale model ids → individual outputs | ✅ 200 / 400 mismatched scale |
| `POST /api/v1/evaluate` | LR + HR ground truth + model → PSNR/SSIM | ✅ 200 |

### Security

- Extension allow-list (jpg/jpeg/png/webp) + PIL `verify()` content validation.
- 10 MB upload size limit, 2048×2048 pixel dimension limit.
- Safe temp files (random names in system temp, atexit cleanup).
- CORS restricted to `http://localhost:5173`.
- No uploaded images persisted; no arbitrary path access; no secrets exposed.

### Test results (21/21)

Health, model registry, all checkpoint loads, SRCNN ×2/×4, ESPCN ×2/×4, output dimensions, image validation, invalid model IDs (404), unsupported file types (400), compare same/mismatched scale, ground-truth evaluation.

---

## Frontend (React + Vite)

**Location:** `frontend/src/` — build passes (`vite build`, 177.48 kB bundle)

### Pages
- **Dashboard** (`#/`) — live backend data: health, 8 registered models, device, scales, architectures, quick nav.
- **Super-Resolution Studio** (`#/studio`) — drag-and-drop + file-picker upload (JPG/JPEG/PNG/WEBP, ≤10 MB), preview, model/scale/loss selection derived from backend availability, Enhance button with loading state, real latency, output preview + metadata, download.
- **Model Comparison** (`#/compare`) — same-scale model group selection, run compatible models on one input, side-by-side results with output dims/latency/device/checkpoint, download.
- **Model Information** (`#/info`) — SRCNN/ESPCN architecture explanations, loss functions (MSE/perceptual/hybrid), honest limitations (reconstruction ≠ recovered original; not forensic evidence; no face/license-plate claims), registered checkpoint table.

### Components
- `ImageUpload.jsx` — drag-and-drop + picker, validation, preview, size display, remove.
- `ModelSelector.jsx` — availability-derived options, unavailable combos disabled.
- `Layout.jsx` — shell with navigation drawer (mobile) + top bar.
- `AppRouter.jsx` — dependency-free hash router (`/`, `/studio`, `/compare`, `/info`).
- `api/client.js` — fetch client through Vite `/api` proxy, structured error handling.

### Design
- Tailwind CSS via CDN, dark-mode ready, responsive grid, lucide-react icons.
- No hardcoded model availability — all from `/api/v1/models` and `/api/v1/health`.
- Metrics never fabricated; shown only with ground truth.

---

## End-to-end smoke test

React → FastAPI → PyTorch checkpoint → React output verified via live server on `http://localhost:8000`:

- `srcnn_4x_mse_perceptual` on 64×64 LR → 256×256 SR, CUDA, ~195 ms, `srcnn_bicubic_upsample` pipeline. ✅
- `espcn_4x_mse` on 64×64 LR → 256×256 SR, CUDA, ~15 ms, `espcn_pixelshuffle` pipeline. ✅
- `evaluate` (LR 64×64 vs HR 256×256, espcn_4x_mse) → PSNR 29.94 dB, SSIM 0.8246, BT.601 Y, shave=scale. ✅
- Invalid model id → 404. Invalid file type → 400. Mismatched-scale compare → 400. ✅

---

## Constraints honored

- ✅ No git commit/push.
- ✅ No retraining.
- ✅ No overwriting checkpoints / `evaluation_results.json` / `summary.csv` / training logs.
- ✅ No background training/processes.
- ✅ No new framework (FastAPI + React + Vite only; Tailwind via CDN).
- ✅ No paid APIs/services.
- ✅ Real inference only; metrics only with ground truth.
- ✅ All existing checkpoints and evaluation outputs preserved.

---

## Launch commands

```bash
# Backend (one terminal)
cd backend
python -m uvicorn app.main:app --port 8000

# Frontend (another terminal)
cd frontend
npm run dev        # dev server on :5173, proxies /api to :8000
# or
npm run build      # production build (verified passing)
```

Open `http://localhost:5173` → Studio → upload image → select model → Enhance.
