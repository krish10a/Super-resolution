import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ViewerMode, CompareAgainst, UpscaleResponse } from '../../lib/types';
import { generateDiffMap } from '../../lib/diffMap';
import { Chip } from '../ui/Chip';
import { ModeSwitch } from './ModeSwitch';

export interface ComparisonViewerProps {
  result: UpscaleResponse | null;
  baselineSrc: string;
  enhancedSrc: string;
  groundTruthSrc?: string | null;
  inputSrc: string;
  mode: ViewerMode;
  onModeChange: (mode: ViewerMode) => void;
  compareAgainst: CompareAgainst;
  onCompareAgainstChange: (target: CompareAgainst) => void;
  lensState?: { x: number; y: number; locked: boolean; active: boolean };
  onPointerMove?: (e: React.PointerEvent<HTMLDivElement>) => void;
  onPointerLeave?: () => void;
  onClickLens?: () => void;
  containerRef?: React.RefObject<HTMLDivElement | null>;
  className?: string;
}

export const ComparisonViewer: React.FC<ComparisonViewerProps> = ({
  result,
  baselineSrc,
  enhancedSrc,
  groundTruthSrc,
  inputSrc,
  mode,
  onModeChange,
  compareAgainst,
  onCompareAgainstChange,
  lensState,
  onPointerMove,
  onPointerLeave,
  onClickLens,
  containerRef,
  className = '',
}) => {
  const [sliderPos, setSliderPos] = useState(50); // 0..100 %
  const [isDragging, setIsDragging] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [flickerActive, setFlickerActive] = useState(true);
  const [flickerToggle, setFlickerToggle] = useState(false);
  const [diffMapUrl, setDiffMapUrl] = useState<string | null>(null);
  const [diffStats, setDiffStats] = useState<{ maxDiff: number; meanDiff: number } | null>(null);
  const [isEnhanceSweeping, setIsEnhanceSweeping] = useState(false);

  const localRef = useRef<HTMLDivElement | null>(null);
  const activeContainerRef = containerRef || localRef;

  // Resolve left comparison image based on compareAgainst
  const leftImageSrc =
    compareAgainst === 'input'
      ? inputSrc
      : compareAgainst === 'ground-truth' && groundTruthSrc
      ? groundTruthSrc
      : baselineSrc;

  const leftLabel =
    compareAgainst === 'input'
      ? 'RAW INPUT'
      : compareAgainst === 'ground-truth'
      ? 'GROUND TRUTH'
      : 'BICUBIC';

  const rightLabel = result ? `${result.modelId.toUpperCase()}` : 'ENHANCED (SR)';

  // Flicker 2Hz timer
  useEffect(() => {
    if (mode !== 'flicker' || !flickerActive) return;
    const interval = setInterval(() => {
      setFlickerToggle((prev) => !prev);
    }, 500); // 2 Hz = 500ms
    return () => clearInterval(interval);
  }, [mode, flickerActive]);

  // Compute DiffMap on Error Map mode
  useEffect(() => {
    if (mode === 'error-map' && groundTruthSrc && enhancedSrc) {
      let isMounted = true;
      generateDiffMap(enhancedSrc, groundTruthSrc)
        .then((res) => {
          if (isMounted) {
            setDiffMapUrl(res.dataUrl);
            setDiffStats({ maxDiff: res.maxDiff, meanDiff: res.meanDiff });
          }
        })
        .catch(console.error);
      return () => {
        isMounted = false;
      };
    }
  }, [mode, enhancedSrc, groundTruthSrc]);

  // Trigger scan line sweep when result updates
  useEffect(() => {
    if (result) {
      setIsEnhanceSweeping(true);
      const timer = setTimeout(() => setIsEnhanceSweeping(false), 800);
      return () => clearTimeout(timer);
    }
  }, [result?.srUrl]);

  // Pointer drag logic for slider divider
  const handlePointerDown = (e: React.PointerEvent) => {
    if (mode !== 'slider') return;
    e.preventDefault();
    setIsDragging(true);
  };

  const handlePointerMoveInternal = (e: React.PointerEvent<HTMLDivElement>) => {
    onPointerMove?.(e);
    if (!isDragging || !activeContainerRef.current) return;
    const rect = activeContainerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    setSliderPos((x / rect.width) * 100);
  };

  const handlePointerUp = () => setIsDragging(false);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (mode !== 'slider') return;
    if (e.key === 'ArrowLeft') setSliderPos((p) => Math.max(0, p - 2));
    if (e.key === 'ArrowRight') setSliderPos((p) => Math.min(100, p + 2));
  };

  const downloadResult = () => {
    if (!enhancedSrc) return;
    const a = document.createElement('a');
    a.href = enhancedSrc;
    a.download = `sisr_result_${result?.modelId || 'enhanced'}.png`;
    a.click();
  };

  return (
    <div
      className={`relative flex flex-col h-full bg-[var(--bg-base)] overflow-hidden select-none ${className}`}
      onPointerUp={handlePointerUp}
    >
      {/* Top Floating Controls */}
      <div className="absolute top-4 inset-x-4 z-30 flex items-center justify-between pointer-events-none">
        {/* Left Side Label & Dropdown */}
        <div className="flex items-center gap-2 pointer-events-auto">
          <Chip variant="bicubic" size="sm">
            {leftLabel}
          </Chip>
          <select
            value={compareAgainst}
            onChange={(e) => onCompareAgainstChange(e.target.value as CompareAgainst)}
            className="text-[11px] font-mono bg-[var(--surface)]/90 border border-[var(--border)] text-[var(--muted)] rounded px-2 py-0.5 focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
          >
            <option value="bicubic">vs. Bicubic</option>
            <option value="input">vs. Raw Input</option>
            {groundTruthSrc && <option value="ground-truth">vs. Ground Truth</option>}
          </select>
        </div>

        {/* Center Mode Switch */}
        <div className="pointer-events-auto">
          <ModeSwitch
            mode={mode}
            onChange={onModeChange}
            errorMapDisabled={!groundTruthSrc}
          />
        </div>

        {/* Right Side Model Label */}
        <div className="pointer-events-auto">
          <Chip variant="accent" size="sm">
            {rightLabel}
          </Chip>
        </div>
      </div>

      {/* Main Viewport Container */}
      <div
        ref={activeContainerRef as any}
        tabIndex={0}
        onKeyDown={handleKeyDown}
        onPointerMove={handlePointerMoveInternal}
        onPointerLeave={onPointerLeave}
        onClick={onClickLens}
        className="relative flex-1 w-full h-full flex items-center justify-center p-4 cursor-crosshair overflow-hidden outline-none"
      >
        <div
          className="relative max-w-full max-h-full flex items-center justify-center transition-transform duration-150"
          style={{ transform: `scale(${zoomLevel})` }}
        >
          {/* 1. SLIDER MODE */}
          {mode === 'slider' && (
            <div className="relative overflow-hidden rounded-[var(--radius-control)] border border-[var(--border)] shadow-2xl">
              {/* Enhanced Image (Right/Under) */}
              <img
                src={enhancedSrc}
                alt="Enhanced Output"
                className="max-h-[62vh] max-w-[80vw] object-contain block pointer-events-none"
                style={{ imageRendering: 'auto' }}
              />

              {/* Before Image with CSS clipPath */}
              <div
                className="absolute inset-0 overflow-hidden pointer-events-none"
                style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
              >
                <img
                  src={leftImageSrc}
                  alt="Reference Baseline"
                  className="w-full h-full object-contain block"
                  style={{ imageRendering: compareAgainst === 'input' ? 'pixelated' : 'auto' }}
                />
              </div>

              {/* Draggable Divider Line */}
              <div
                className="absolute top-0 bottom-0 w-0.5 bg-[var(--accent)] cursor-ew-resize z-20 shadow-[0_0_8px_rgba(56,225,255,0.8)]"
                style={{ left: `${sliderPos}%` }}
                onPointerDown={handlePointerDown}
              >
                {/* Circular Draggable Handle */}
                <div
                  className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-[var(--surface)] border-2 border-[var(--accent)] flex items-center justify-center shadow-lg cursor-ew-resize hover:scale-110 active:scale-95 transition-transform"
                >
                  <svg className="w-3.5 h-3.5 text-[var(--accent)]" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M8 9l-4 3 4 3m8-6l4 3-4 3" />
                  </svg>
                </div>
              </div>
            </div>
          )}

          {/* 2. SIDE BY SIDE MODE */}
          {mode === 'side-by-side' && (
            <div className="grid grid-cols-2 gap-3 max-h-[62vh] max-w-[85vw]">
              <div className="relative rounded-[var(--radius-control)] border border-[var(--border)] overflow-hidden bg-[var(--surface)]">
                <div className="absolute top-2 left-2 z-10">
                  <Chip variant="bicubic" size="sm">{leftLabel}</Chip>
                </div>
                <img
                  src={leftImageSrc}
                  alt="Reference"
                  className="w-full h-full object-contain"
                  style={{ imageRendering: compareAgainst === 'input' ? 'pixelated' : 'auto' }}
                />
              </div>
              <div className="relative rounded-[var(--radius-control)] border border-[var(--border)] overflow-hidden bg-[var(--surface)]">
                <div className="absolute top-2 left-2 z-10">
                  <Chip variant="accent" size="sm">{rightLabel}</Chip>
                </div>
                <img
                  src={enhancedSrc}
                  alt="Enhanced"
                  className="w-full h-full object-contain"
                />
              </div>
            </div>
          )}

          {/* 3. FLICKER MODE */}
          {mode === 'flicker' && (
            <div className="relative overflow-hidden rounded-[var(--radius-control)] border border-[var(--border)] shadow-2xl">
              <div className="absolute top-3 left-3 z-20 flex items-center gap-2">
                <Chip variant={flickerToggle ? 'accent' : 'bicubic'} size="sm">
                  {flickerToggle ? rightLabel : leftLabel}
                </Chip>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setFlickerActive(!flickerActive);
                  }}
                  className="p-1 rounded bg-[var(--surface)]/90 border border-[var(--border)] text-xs text-[var(--text)] hover:bg-[var(--raised)]"
                >
                  {flickerActive ? 'Pause' : 'Play'}
                </button>
              </div>
              <img
                src={flickerToggle ? enhancedSrc : leftImageSrc}
                alt="Flicker Frame"
                className="max-h-[62vh] max-w-[80vw] object-contain block"
              />
            </div>
          )}

          {/* 4. ERROR MAP MODE */}
          {mode === 'error-map' && (
            <div className="relative overflow-hidden rounded-[var(--radius-control)] border border-[var(--border)] shadow-2xl">
              {diffMapUrl ? (
                <>
                  <img
                    src={diffMapUrl}
                    alt="Error Heatmap"
                    className="max-h-[62vh] max-w-[80vw] object-contain block"
                  />
                  {/* Colormap Legend */}
                  <div className="absolute bottom-3 right-3 z-20 flex flex-col items-end gap-1 p-2 rounded bg-[var(--surface)]/90 backdrop-blur border border-[var(--border)] font-mono text-[10px]">
                    <div className="text-[var(--text)] font-semibold">|Model − GroundTruth|</div>
                    <div className="w-32 h-2.5 rounded bg-gradient-to-r from-blue-900 via-emerald-400 to-red-500 border border-[var(--border)]" />
                    <div className="w-full flex justify-between text-[var(--muted)]">
                      <span>0.00 (Match)</span>
                      <span>Max: {diffStats?.maxDiff || 0.25}</span>
                    </div>
                  </div>
                </>
              ) : (
                <div className="p-12 text-center font-mono text-sm text-[var(--muted)]">
                  Computing difference map...
                </div>
              )}
            </div>
          )}

          {/* Synchronized Magnifier Lens Rectangle */}
          {lensState && lensState.active && (
            <div
              className={`absolute pointer-events-none border-2 transition-opacity duration-150 z-20 ${
                lensState.locked ? 'border-[var(--amber)] shadow-[0_0_12px_rgba(255,176,32,0.8)]' : 'border-[var(--accent)] shadow-[0_0_10px_rgba(56,225,255,0.7)]'
              }`}
              style={{
                left: `${lensState.x * 100}%`,
                top: `${lensState.y * 100}%`,
                width: 64,
                height: 64,
                transform: 'translate(-50%, -50%)',
              }}
            >
              <span className="absolute -top-5 left-0 font-mono text-[9px] bg-black/80 px-1 py-0.2 rounded text-[var(--accent)]">
                {lensState.locked ? 'LOCKED' : '64×64 LENS'}
              </span>
            </div>
          )}

          {/* Enhance Sweep Beam */}
          <AnimatePresence>
            {isEnhanceSweeping && (
              <motion.div
                initial={{ top: '0%', opacity: 1 }}
                animate={{ top: '100%', opacity: 0.8 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
                className="absolute inset-x-0 h-1.5 bg-[var(--accent)] shadow-[0_0_20px_#38E1FF] z-30 pointer-events-none"
              />
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Bottom Status & Utility Bar */}
      <div className="flex items-center justify-between px-5 py-2.5 bg-[var(--surface)] border-t border-[var(--border)] z-30">
        {/* Bottom Left Resolution Caption */}
        <div className="flex items-center gap-3 font-mono text-xs text-[var(--muted)]">
          <span>
            {result?.inputSize ? `${result.inputSize[0]}×${result.inputSize[1]}` : '128×128'}
            {' → '}
            <strong className="text-[var(--accent)]">
              {result?.outputSize ? `${result.outputSize[0]}×${result.outputSize[1]}` : '512×512'}
            </strong>
          </span>
          {result && (
            <span className="text-[11px] px-2 py-0.5 rounded bg-[var(--raised)] border border-[var(--border)]">
              {result.ms} ms
            </span>
          )}
        </div>

        {/* Bottom Right Action Icons */}
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setZoomLevel((z) => Math.min(3, z + 0.25))}
            title="Zoom In"
            className="p-1.5 rounded hover:bg-[var(--raised)] text-[var(--muted)] hover:text-[var(--text)] transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
          </button>
          <button
            type="button"
            onClick={() => setZoomLevel((z) => Math.max(0.5, z - 0.25))}
            title="Zoom Out"
            className="p-1.5 rounded hover:bg-[var(--raised)] text-[var(--muted)] hover:text-[var(--text)] transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
            </svg>
          </button>
          <button
            type="button"
            onClick={() => setZoomLevel(1)}
            title="Reset Zoom"
            className="p-1.5 rounded hover:bg-[var(--raised)] text-[var(--muted)] hover:text-[var(--text)] text-xs font-mono transition-colors"
          >
            100%
          </button>
          <div className="w-[1px] h-4 bg-[var(--border)] mx-1" />
          <button
            type="button"
            onClick={downloadResult}
            title="Download Enhanced Result (PNG)"
            className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[var(--raised)] hover:bg-[var(--accent)] hover:text-black border border-[var(--border)] text-xs font-medium text-[var(--text)] transition-colors"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            <span>Download</span>
          </button>
        </div>
      </div>
    </div>
  );
};

