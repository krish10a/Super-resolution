# STITCH UI TEMPLATE Migration Progress Checklist

## Phase 0: Discovery & Setup
- [x] Read `STITCH UI TEMPLATE/` completely (HTML and screenshots for Overview, Studio, Benchmarks, Models, Method).
- [x] Analyze `DESIGN.md` for Forensic Precision Laboratory design tokens.
- [x] Backup existing UI to `frontend/_backup_old_ui/`.
- [x] Exclude `_backup_old_ui` from `tsconfig.json` and build tools.
- [x] Create `frontend/scripts/sync-data.mjs` to auto-copy `outputs/evaluation/` to `public/data/`.
- [x] Wire `predev` and `prebuild` scripts into `package.json`.

## Phase 1: Foundation & Design Tokens
- [x] Install `@fontsource/space-grotesk`, `@fontsource/inter`, `@fontsource/jetbrains-mono`.
- [x] Remove Google Fonts & Material Symbols CDN links from `index.html`.
- [x] Write consolidated `src/styles/tokens.css` with exact variables matching Stitch `DESIGN.md`.
- [x] Update `src/index.css` with Tailwind theme classes matching Stitch (`surface-container-lowest`, etc.).
- [x] Build global 56px TopBar with animated underline, hardware chip, and API polling dot (no avatar).
- [x] Build 28px fixed StatusBar with live hardware, active checkpoint, and latency telemetry.

## Phase 2: Screen-by-Screen Port with Pixel Fidelity
- [x] **Overview (`/`)**:
  - [x] Port `overview/code.html` markup & pixel layout.
  - [x] Build `HeroCurtain.tsx` with drag splitter, laser divider, viewfinder reticles, local images.
  - [x] Build dynamic Metric Tiles (8 Models, 119 Images, Best PSNR Delta from data).
  - [x] Build 4-Stage Horizontal Topology Strip.
  - [x] Build Closing CTA Banner linking to `/studio` and `/benchmarks`.
- [x] **Studio (`/studio`)**:
  - [x] Port `studio/code.html` 3-column layout `[300px | 1fr | 320px]`.
  - [x] Build Source Dropzone (validation, preview, clear).
  - [x] Build Sample Carousel (Set5, Set14, BSD100, CCTV presets with active ring).
  - [x] Build Model, Scale, Loss controls resolving real checkpoints.
  - [x] Build collapsible degradation simulator (blur, noise, JPEG; enabled for samples).
  - [x] Wire Enhance button to real `POST /api/v1/super-resolve` with loading shimmer & cancel.
  - [x] Build center Viewport (Curtain slider, Side-by-side, 2Hz Flicker, Residual map).
  - [x] Build Loupe Quad shelf (Input nearest, Bicubic, Model, Ground truth) with sync lens & lock.
  - [x] Build Compare 4 Quad view with tile promotion to Single view.
  - [x] Build Metrics Panel (PSNR/SSIM/LPIPS count-up, baseline deltas, telemetry, context rank).
- [x] **Benchmarks (`/benchmarks`)**:
  - [x] Port `benchmarks/code.html` layout.
  - [x] Wire FilterBar chips synced to URL query params (`?dataset=...&scale=...&metric=...`).
  - [x] Compute dynamic headline from real evaluation data (`Closest to bicubic: ...`).
  - [x] Build sortable Heatmap Matrix with delta coloring & row drawer trigger.
  - [x] Build Grouped Bar chart with dashed bicubic baseline reference line and champion glow.
  - [x] Build Tradeoff Scatter chart (PSNR vs LPIPS) with paired connecting vectors.
  - [x] Build Per-Image Detailed Drawer with sortable metrics table.
- [x] **Models (`/models`)**:
  - [x] Port `models/code.html` layout.
  - [x] Build 4×2 Experiment Cards (E1–E8) with SVG sparklines and status tags.
  - [x] Build 560px Drawer with Architecture Diagram (real layer specs) & Training Curve (warmup line).
  - [x] Real training convergence curve with live stats (Initial PSNR, Peak PSNR, Final Loss).
  - [x] Quick actions: Load into Studio, CSV Export, Batch Delta.
- [x] **Method (`/method`)**:
  - [x] Port `method/code.html` layout.
  - [x] Build interactive 4-stage degradation stepper on local frame with real CSS filters.
  - [x] Build Protocol Card (ITU-R BT.601, border shaving by scale).
  - [x] Build Loss Objective Card (hybrid loss, VGG-19 feature perceptual loss, 3-epoch warmup).
  - [x] Build Data Parity Timeline (T91 76 mismatches -> 0 mismatches in `pairs_fixed`).
  - [x] Build Academic Limitations List.

## Phase 3: Zero-Placeholder Audit & Offline Readiness
- [x] Remove all external placeholder URLs (`googleusercontent.com`, `unsplash`, etc.).
- [x] Remove non-functional Stitch elements (avatar icon, dead buttons).
- [x] Ensure all icons are bundled SVGs / inline icons (no CDN).
- [x] Verify 100% offline functionality (self-hosted fonts via `@fontsource`).

## Phase 4: States, Quality & Accessibility
- [x] Skeleton loaders, empty states, and API offline banners.
- [x] Responsive drawer collapse at 1024px and vertical stacking under 768px.
- [x] Full keyboard navigation (Tab, Enter, Arrows, Esc focus traps).
- [x] `prefers-reduced-motion` compliance.

## Phase 5: Verification & Simplified Revision Update
- [x] Run `npm run build` with zero errors (built in 9.67s, clean bundles).
- [x] Run `pytest tests/` with zero regressions (105 passed in 26.87s).
- [x] Zero-placeholder audit: 0 instances of external placeholder URLs, CDN links, or TODO/FIXME markers in `frontend/src`.
- [x] Simplified Revision aligned across all screens:
  - [x] TopBar & StatusBar: Removed redundant telemetry tickers, simplified to `CUDA · {ckpt} · {ms}`.
  - [x] Overview: Cleaned hero text, removed CUDA ticker/extra chips, 3 dynamic metrics, 4-stage pipeline, concise CTA.
  - [x] Benchmarks: Fixed to show one scale at a time (2× vs 4×), merged Model column, deleted telemetry badges and extra chips, clean per-image drawer.
  - [x] Models: Simplified 4×2 grid with clean sparklines, deleted registry footnote / CSV / Batch Delta, streamlined 520px drawer with architecture sequence + SVG training curve.
  - [x] Method: Updated 4-step degradation stepper, concise protocol & loss cards, clean data fix stats, concise academic limitations.
  - [x] Studio: Aligned with simplified `studio/code.html` layout (clean labels, single-line checkpoint, simplified Loupe shelf, FIDELITY/RUN/IN CONTEXT cards with real computation, concise amber anomaly alert).
- [x] All real API wiring and data computations preserved.
