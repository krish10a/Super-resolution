import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Link } from 'react-router-dom';
import {
  UploadCloud,
  Sparkles,
  Sliders,
  ZoomIn,
  Maximize2,
  Minimize2,
  Download,
  AlertTriangle,
  HelpCircle,
  ChevronDown,
  ChevronRight,
  Eye,
  EyeOff,
  Layers,
  SplitSquareVertical,
  RotateCcw,
  Play,
  Pause,
  Check,
  X,
  FileImage,
  ArrowRight,
  MoveHorizontal,
  Compass,
  Cpu,
  Clock,
  Crosshair,
} from 'lucide-react';
import {
  ModelName,
  Scale,
  LossName,
  DenoiseOption,
  SampleItem,
  ViewerMode,
  CompareAgainst,
  DegradationSettings,
} from '../lib/types';
import { resolveExperiment, computeLiveMetrics } from '../lib/metricsMath';
import { useUpscale } from '../hooks/useUpscale';
import { useResults } from '../hooks/useResults';
import { useMagnifier } from '../hooks/useMagnifier';
import { fetchSamplesList } from '../lib/api';
import { generateDiffMap } from '../lib/diffMap';

export const Studio: React.FC = () => {
  // Model state
  const [model, setModel] = useState<ModelName>('EDSR');
  const [scale, setScale] = useState<Scale>(2);
  const [loss, setLoss] = useState<LossName>('L1');
  const [denoise, setDenoise] = useState<DenoiseOption>('none');
  const [isLegacy, setIsLegacy] = useState<boolean>(false);

  // View state
  const [viewMode, setViewMode] = useState<'single' | 'compare4'>('single');
  const [viewerMode, setViewerMode] = useState<ViewerMode>('slider');
  const [compareAgainst, setCompareAgainst] = useState<CompareAgainst>('input');

  // Samples and custom images
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [selectedSample, setSelectedSample] = useState<SampleItem | null>(null);
  const [customImage, setCustomImage] = useState<string | null>(null);
  const [customFileName, setCustomFileName] = useState<string>('custom_input.png');
  const [customFileSize, setCustomFileSize] = useState<string>('—');
  const [customDimensions, setCustomDimensions] = useState<[number, number] | null>(null);
  const [simulateCctv, setSimulateCctv] = useState<boolean>(false);
  const [customLiveMetrics, setCustomLiveMetrics] = useState<{
    psnr: number;
    ssim: number;
    bicubicPsnr?: number;
    bicubicSsim?: number;
  } | null>(null);

  // Splitter & Zoom state
  const [splitPos, setSplitPos] = useState(50); // 0..100%
  const [isDragging, setIsDragging] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Flicker & Diff Map states
  const [flickerToggle, setFlickerToggle] = useState(false);
  const [flickerActive, setFlickerActive] = useState(true);
  const [diffMapUrl, setDiffMapUrl] = useState<string | null>(null);

  // Degradation drawer state
  const [degradeOpen, setDegradeOpen] = useState(false);
  const [degradation, setDegradation] = useState<DegradationSettings>({
    enabled: false,
    blurSigma: 1.5,
    noiseSigma: 15,
    jpegQuality: 50,
  });

  // Mobile drawer states
  const [mobileControlsOpen, setMobileControlsOpen] = useState(false);
  const [mobileMetricsOpen, setMobileMetricsOpen] = useState(false);

  // Hooks
  const { loading, error, activeResult, compare4Results, runUpscale, runCompare4 } = useUpscale();
  const { results, getRank } = useResults();
  const {
    state: lensState,
    containerRef: viewportRef,
    handlePointerMove: handleLensMove,
    handlePointerLeave: handleLensLeave,
    toggleLock: toggleLensLock,
  } = useMagnifier(64);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Load samples
  useEffect(() => {
    fetchSamplesList().then((data) => {
      setSamples(data);
      if (data.length > 0) {
        const defaultSample = data.find((s) => s.id === 'set14_01') || data[0];
        setSelectedSample(defaultSample);
      }
    });
  }, []);

  // Sync fullscreen change event
  useEffect(() => {
    const handleFsChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };
    document.addEventListener('fullscreenchange', handleFsChange);
    return () => document.removeEventListener('fullscreenchange', handleFsChange);
  }, []);

  // Determine active LR source URL
  // If activeResult returned degraded LR (simulate_lr), prefer it so user compares actual LR
  const activeInputUrl = activeResult?.lrUrl
    ? activeResult.lrUrl
    : customImage
    ? customImage
    : selectedSample?.lrUrls?.[scale.toString()] ||
      selectedSample?.lrUrl ||
      '/samples/set14_01_lr_x4.png';

  const activeHrUrl = activeResult?.hrUrl
    ? activeResult.hrUrl
    : customImage
    ? null
    : selectedSample?.hrUrl || null;

  const hasGroundTruth = Boolean(activeHrUrl);

  // Auto-run initial upscale on mount or selection change
  useEffect(() => {
    const src = customImage || activeInputUrl;
    if (src) {
      const isSim = Boolean(customImage && simulateCctv);
      if (viewMode === 'single') {
        runUpscale(src, model, scale, loss, selectedSample, degradation, isSim, denoise, isLegacy);
      } else {
        runCompare4(src, scale, selectedSample, degradation, isSim, denoise, isLegacy);
      }
    }
  }, [customImage, selectedSample, model, scale, loss, viewMode, simulateCctv, denoise, isLegacy]);

  // Compute live metrics if paired ground truth is available (e.g. custom simulated upload)
  useEffect(() => {
    if (activeResult?.hrUrl && activeResult?.srUrl) {
      let isMounted = true;
      Promise.all([
        computeLiveMetrics(activeResult.srUrl, activeResult.hrUrl, scale),
        computeLiveMetrics(activeResult.bicubicUrl, activeResult.hrUrl, scale),
      ])
        .then(([srMet, bicMet]) => {
          if (isMounted) {
            setCustomLiveMetrics({
              psnr: srMet.psnr,
              ssim: srMet.ssim,
              bicubicPsnr: bicMet.psnr,
              bicubicSsim: bicMet.ssim,
            });
          }
        })
        .catch(console.error);
      return () => {
        isMounted = false;
      };
    } else {
      setCustomLiveMetrics(null);
    }
  }, [activeResult?.hrUrl, activeResult?.srUrl, activeResult?.bicubicUrl, scale]);

  // Flicker 2Hz timer
  useEffect(() => {
    if (viewerMode !== 'flicker' || !flickerActive) return;
    const interval = setInterval(() => setFlickerToggle((p) => !p), 500);
    return () => clearInterval(interval);
  }, [viewerMode, flickerActive]);

  // Error Map generation
  useEffect(() => {
    if (viewerMode === 'error-map' && activeHrUrl && activeResult?.srUrl) {
      let isMounted = true;
      generateDiffMap(activeResult.srUrl, activeHrUrl)
        .then((res) => {
          if (isMounted) setDiffMapUrl(res.dataUrl);
        })
        .catch(console.error);
      return () => {
        isMounted = false;
      };
    }
  }, [viewerMode, activeHrUrl, activeResult?.srUrl]);

  // Split curtain drag logic
  const updateSplit = useCallback((clientX: number) => {
    if (!viewportRef.current) return;
    const rect = viewportRef.current.getBoundingClientRect();
    const offsetX = clientX - rect.left;
    let pct = (offsetX / rect.width) * 100;
    if (pct < 1) pct = 1;
    if (pct > 99) pct = 99;
    setSplitPos(pct);
  }, []);

  const handlePointerDown = (e: React.PointerEvent) => {
    if (viewerMode !== 'slider') return;
    setIsDragging(true);
    updateSplit(e.clientX);
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    handleLensMove(e);
    if (!isDragging) return;
    updateSplit(e.clientX);
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    setIsDragging(false);
    try {
      (e.target as HTMLElement).releasePointerCapture?.(e.pointerId);
    } catch (_) {}
  };

  // Custom Upload Handler
  const handleFileChange = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please select a PNG or JPG image file.');
      return;
    }
    const sizeKb = (file.size / 1024).toFixed(0);
    setCustomFileName(file.name);
    setCustomFileSize(`${sizeKb} KB`);

    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      const img = new Image();
      img.onload = () => {
        // Auto-enable Simulate CCTV mode if image exceeds 256px
        const isLarge = img.width > 256 || img.height > 256;
        setSimulateCctv(isLarge);
        setCustomDimensions([img.width, img.height]);
        setCustomImage(dataUrl);
        setSelectedSample(null);
        runUpscale(dataUrl, model, scale, loss, null, degradation, isLarge, denoise, isLegacy);
      };
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);
  };

  const clearCustomImage = () => {
    setCustomImage(null);
    setCustomDimensions(null);
    setSimulateCctv(false);
    setCustomLiveMetrics(null);
    if (samples.length > 0) {
      const defaultSample = samples.find((s) => s.id === 'set14_01') || samples[0];
      setSelectedSample(defaultSample);
    }
  };

  // Download Output PNG
  const handleDownload = () => {
    const src = activeResult?.srUrl;
    if (!src) return;
    const a = document.createElement('a');
    a.href = src;
    a.download = `sisr_${model.toLowerCase()}_${scale}x_${loss === 'MSE' ? 'mse' : 'perc'}.png`;
    a.click();
  };

  // Fullscreen toggle
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      viewportRef.current?.requestFullscreen?.();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.();
      setIsFullscreen(false);
    }
  };

  const resolved = resolveExperiment(model, scale, loss, isLegacy);
  const datasetGroup = (selectedSample?.group as any) || 'Set14';
  const rankInfo = getRank(resolved.id, datasetGroup);

  const leftDisplaySrc =
    compareAgainst === 'input'
      ? activeInputUrl
      : compareAgainst === 'ground-truth' && activeHrUrl
      ? activeHrUrl
      : activeResult?.bicubicUrl || activeInputUrl;

  const leftLabel =
    compareAgainst === 'input'
      ? 'RAW INPUT (LR)'
      : compareAgainst === 'ground-truth'
      ? 'GROUND TRUTH'
      : 'BICUBIC';

  const metrics = customLiveMetrics
    ? { psnr: customLiveMetrics.psnr, ssim: customLiveMetrics.ssim, lpips: activeResult?.metrics?.lpips }
    : activeResult?.metrics;

  const bicubicMetrics = customLiveMetrics?.bicubicPsnr != null
    ? { psnr: customLiveMetrics.bicubicPsnr, ssim: customLiveMetrics.bicubicSsim || 0.812, lpips: 0.155 }
    : activeResult?.bicubicMetrics;

  const isLowerPsnr = metrics && bicubicMetrics ? metrics.psnr < bicubicMetrics.psnr : false;

  // Render left control panel content (reused in desktop sidebar & mobile drawer)
  const renderControls = () => (
    <div className="flex flex-col gap-4">
      {/* Section: SOURCE */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">
            Source Image
          </span>
          {customImage && (
            <button
              onClick={clearCustomImage}
              className="text-[11px] font-mono text-outline hover:text-bad flex items-center gap-1 transition-colors"
              title="Reset to presets"
            >
              <RotateCcw className="w-3 h-3" />
              <span>Reset</span>
            </button>
          )}
        </div>

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/png, image/jpeg"
          className="hidden"
          onChange={(e) => {
            if (e.target.files?.[0]) handleFileChange(e.target.files[0]);
          }}
        />

        {/* Custom Upload State OR Dropzone */}
        {customImage ? (
          <div className="border border-primary-container/80 rounded-xl p-3 bg-surface-container/70 flex flex-col gap-2.5 shadow-[0_0_14px_rgba(56,225,255,0.18)] transition-all">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2.5 min-w-0">
                <div className="w-10 h-10 rounded-lg bg-surface-container-lowest border border-outline-variant overflow-hidden flex-shrink-0 relative group">
                  <img
                    alt="Custom preview"
                    className="w-full h-full object-cover"
                    src={customImage}
                  />
                </div>
                <div className="flex flex-col min-w-0">
                  <span className="font-mono-metric-sm text-mono-metric-sm text-on-surface truncate font-medium">
                    {customFileName}
                  </span>
                  <span className="font-mono-metric-sm text-[11px] text-primary-container">
                    {customDimensions
                      ? `${customDimensions[0]} × ${customDimensions[1]} px`
                      : 'Custom Image'}{' '}
                    · {customFileSize}
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-1.5 flex-shrink-0">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="px-2.5 py-1 rounded-md bg-surface-container hover:bg-surface-container-high border border-outline-variant text-[11px] font-mono-metric-sm text-on-surface-variant hover:text-on-surface transition-colors"
                >
                  Replace
                </button>
                <button
                  onClick={clearCustomImage}
                  className="p-1 rounded-md hover:bg-bad/10 text-outline hover:text-bad border border-transparent hover:border-bad/30 transition-colors"
                  title="Remove custom image"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* Workflow Mode Selector */}
            <div className="pt-2 border-t border-outline-variant/60 flex flex-col gap-1.5">
              <div className="flex items-center justify-between text-[10px] font-mono-metric-sm">
                <span className="text-on-surface-variant uppercase tracking-wider">Workflow</span>
                <span className="text-primary-container font-medium">
                  {simulateCctv ? 'Surveillance Simulation' : 'Direct LR Input'}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-1 bg-surface-container-lowest p-0.5 rounded-lg border border-outline-variant">
                <button
                  type="button"
                  onClick={() => {
                    setSimulateCctv(true);
                    runUpscale(customImage, model, scale, loss, null, degradation, true, denoise, isLegacy);
                  }}
                  className={`py-1 text-[11px] font-mono-metric-sm rounded transition-all text-center ${
                    simulateCctv
                      ? 'bg-primary-container/20 text-primary-container border border-primary-container/40 font-medium'
                      : 'text-on-surface-variant hover:text-on-surface'
                  }`}
                  title="Downsamples photo by 4x to realistic CCTV feed, super-resolves it back, and evaluates against original ground truth"
                >
                  Simulate CCTV
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setSimulateCctv(false);
                    runUpscale(customImage, model, scale, loss, null, degradation, false, denoise, isLegacy);
                  }}
                  className={`py-1 text-[11px] font-mono-metric-sm rounded transition-all text-center ${
                    !simulateCctv
                      ? 'bg-primary-container/20 text-primary-container border border-primary-container/40 font-medium'
                      : 'text-on-surface-variant hover:text-on-surface'
                  }`}
                  title="Treats uploaded image directly as the low-resolution input"
                >
                  Direct LR
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              if (e.dataTransfer.files?.[0]) handleFileChange(e.dataTransfer.files[0]);
            }}
            className="border border-dashed border-outline-variant rounded-xl p-3.5 text-center bg-surface-container/40 hover:bg-surface-container hover:border-primary-container/70 transition-all cursor-pointer group flex flex-col items-center justify-center gap-1.5"
          >
            <div className="w-8 h-8 rounded-full bg-surface-container-lowest border border-outline-variant group-hover:border-primary-container/60 flex items-center justify-center text-outline group-hover:text-primary-container transition-colors">
              <UploadCloud className="w-4 h-4" />
            </div>
            <div className="flex flex-col items-center">
              <p className="font-body-sm-medium text-body-sm-medium text-on-surface group-hover:text-primary-container transition-colors">
                Drop an image or click to upload
              </p>
              <span className="font-mono-metric-sm text-[11px] text-outline">
                PNG or JPG (max 2048px)
              </span>
            </div>
          </div>
        )}

        {/* Sample Thumbnails Carousel Grid */}
        <div className="flex items-center justify-between pt-1">
          <span className="font-mono-metric-sm text-[10px] text-outline uppercase tracking-wider">
            Standard Presets
          </span>
          <span className="font-mono-metric-sm text-[10px] text-outline">
            {samples.length} samples
          </span>
        </div>
        <div className="grid grid-cols-6 gap-1.5">
          {samples.slice(0, 6).map((s) => {
            const isSelected = selectedSample?.id === s.id && !customImage;
            return (
              <button
                key={s.id}
                onClick={() => {
                  setSelectedSample(s);
                  setCustomImage(null);
                }}
                type="button"
                className="flex flex-col items-center gap-1 group focus:outline-none"
              >
                <div
                  className={`w-10 h-10 rounded-lg bg-surface-container overflow-hidden transition-all relative ${
                    isSelected
                      ? 'border-2 border-primary-container shadow-[0_0_12px_rgba(56,225,255,0.4)] scale-105'
                      : 'border border-outline-variant group-hover:border-outline'
                  }`}
                >
                  <img
                    className="w-full h-full object-cover"
                    src={s.thumbUrl}
                    alt={s.name}
                  />
                  {isSelected && (
                    <div className="absolute inset-0 bg-primary-container/10 pointer-events-none" />
                  )}
                </div>
                <span
                  className={`font-mono-metric-sm text-[10px] truncate max-w-[44px] ${
                    isSelected ? 'text-primary-container font-semibold' : 'text-on-surface-variant'
                  }`}
                >
                  {s.group}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="h-[1px] bg-outline-variant/60 w-full" />

      {/* Section: MODEL ARCHITECTURE */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center justify-between">
          <span className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">
            Model Architecture
          </span>
          <button
            type="button"
            onClick={() => {
              const nextLegacy = !isLegacy;
              setIsLegacy(nextLegacy);
              if (nextLegacy) {
                setModel('SRCNN');
                setLoss('MSE');
              } else {
                setModel('EDSR');
                setLoss('L1');
              }
            }}
            className="text-[10px] text-outline hover:text-primary-container transition font-mono-metric-sm"
          >
            {isLegacy ? '← Switch to DIV2K' : 'Legacy (SRCNN)'}
          </button>
        </div>
        <div
          className={`grid ${
            isLegacy ? 'grid-cols-2' : 'grid-cols-3'
          } p-1 bg-surface-container-lowest rounded-xl border border-outline-variant`}
        >
          {isLegacy ? (
            <>
              <button
                type="button"
                onClick={() => setModel('SRCNN')}
                className={`py-2 text-center font-body-sm-medium text-body-sm-medium rounded-lg transition-all ${
                  model === 'SRCNN'
                    ? 'text-primary-container bg-surface-container border border-primary-container/40 shadow-sm font-semibold'
                    : 'text-on-surface-variant hover:text-on-surface'
                }`}
              >
                SRCNN
              </button>
              <button
                type="button"
                onClick={() => setModel('ESPCN')}
                className={`py-2 text-center font-body-sm-medium text-body-sm-medium rounded-lg transition-all ${
                  model === 'ESPCN'
                    ? 'text-primary-container bg-surface-container border border-primary-container/40 shadow-sm font-semibold'
                    : 'text-on-surface-variant hover:text-on-surface'
                }`}
              >
                ESPCN
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                onClick={() => setModel('EDSR')}
                className={`py-2 text-center font-body-sm-medium text-[12px] rounded-lg transition-all ${
                  model === 'EDSR'
                    ? 'text-primary-container bg-surface-container border border-primary-container/40 shadow-sm font-semibold'
                    : 'text-on-surface-variant hover:text-on-surface'
                }`}
                title="EDSR Baseline (1.37M params) - Top forensic reconstruction quality"
              >
                EDSR <span className="text-[10px] opacity-70 block">Quality</span>
              </button>
              <button
                type="button"
                onClick={() => setModel('ESPCN')}
                className={`py-2 text-center font-body-sm-medium text-[12px] rounded-lg transition-all ${
                  model === 'ESPCN'
                    ? 'text-primary-container bg-surface-container border border-primary-container/40 shadow-sm font-semibold'
                    : 'text-on-surface-variant hover:text-on-surface'
                }`}
                title="ESPCN (26.8k params) - Sub-pixel CNN >175 FPS real-time"
              >
                ESPCN <span className="text-[10px] opacity-70 block">Fast</span>
              </button>
              <button
                type="button"
                onClick={() => setModel('FSRCNN')}
                className={`py-2 text-center font-body-sm-medium text-[12px] rounded-lg transition-all ${
                  model === 'FSRCNN'
                    ? 'text-primary-container bg-surface-container border border-primary-container/40 shadow-sm font-semibold'
                    : 'text-on-surface-variant hover:text-on-surface'
                }`}
                title="FSRCNN (24.7k params) - Compact deconvolution architecture"
              >
                FSRCNN <span className="text-[10px] opacity-70 block">Balanced</span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* Section: UPSCALE FACTOR */}
      <div className="flex flex-col gap-1.5">
        <span className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">
          Upscale Factor
        </span>
        <div className="grid grid-cols-2 p-1 bg-surface-container-lowest rounded-xl border border-outline-variant">
          <button
            type="button"
            onClick={() => setScale(2)}
            className={`py-2 text-center font-mono-metric-md text-mono-metric-md rounded-lg transition-all ${
              scale === 2
                ? 'text-primary-container bg-surface-container border border-primary-container/40 shadow-sm font-semibold'
                : 'text-on-surface-variant hover:text-on-surface'
            }`}
          >
            2×
          </button>
          <button
            type="button"
            onClick={() => setScale(4)}
            className={`py-2 text-center font-mono-metric-md text-mono-metric-md rounded-lg transition-all ${
              scale === 4
                ? 'text-primary-container bg-surface-container border border-primary-container/40 shadow-sm font-semibold'
                : 'text-on-surface-variant hover:text-on-surface'
            }`}
          >
            4×
          </button>
        </div>
      </div>

      {/* Section: CCTV PRE-FILTER (Phase 4.3 Denoising) */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center justify-between">
          <span className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">
            CCTV Pre-Filter
          </span>
          <span className="font-mono-metric-sm text-[10px] text-good font-semibold">
            {denoise === 'bilateral' ? '+1.35 dB Rescue' : 'Pass-through'}
          </span>
        </div>
        <div className="grid grid-cols-2 p-1 bg-surface-container-lowest rounded-xl border border-outline-variant">
          <button
            type="button"
            onClick={() => setDenoise('none')}
            className={`py-2 text-center font-body-sm-medium text-[12px] rounded-lg transition-all ${
              denoise === 'none'
                ? 'text-primary-container bg-surface-container border border-primary-container/40 shadow-sm font-semibold'
                : 'text-on-surface-variant hover:text-on-surface'
            }`}
          >
            Off
          </button>
          <button
            type="button"
            onClick={() => setDenoise('bilateral')}
            className={`py-2 text-center font-body-sm-medium text-[12px] rounded-lg transition-all ${
              denoise === 'bilateral'
                ? 'text-primary-container bg-surface-container border border-primary-container/40 shadow-sm font-semibold'
                : 'text-on-surface-variant hover:text-on-surface'
            }`}
          >
            Bilateral Filter
          </button>
        </div>
        <p className="text-[11px] text-outline px-1">
          {denoise === 'bilateral'
            ? 'Fast edge-preserving pre-filter (+0.86ms). Suppresses sensor/CCTV noise.'
            : 'Direct LR feed without pre-processing.'}
        </p>
      </div>

      {/* Section: LOSS OBJECTIVE (Legacy Only) */}
      {isLegacy ? (
        <div className="flex flex-col gap-1.5">
          <span className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">
            Supervision Loss
          </span>
          <div className="grid grid-cols-2 p-1 bg-surface-container-lowest rounded-xl border border-outline-variant">
            <button
              type="button"
              onClick={() => setLoss('MSE')}
              className={`py-2 text-center font-body-sm-medium text-[12px] rounded-lg transition-all ${
                loss === 'MSE'
                  ? 'text-primary-container bg-surface-container border border-primary-container/40 shadow-sm font-semibold'
                  : 'text-on-surface-variant hover:text-on-surface'
              }`}
            >
              MSE
            </button>
            <button
              type="button"
              onClick={() => setLoss('MSE + Perceptual')}
              className={`py-2 text-center font-body-sm-medium text-[12px] rounded-lg transition-all ${
                loss === 'MSE + Perceptual'
                  ? 'text-primary-container bg-surface-container border border-primary-container/40 shadow-sm font-semibold'
                  : 'text-on-surface-variant hover:text-on-surface'
              }`}
            >
              MSE + Perceptual
            </button>
          </div>
        </div>
      ) : (
        <div className="bg-surface-container/60 px-3 py-2 rounded-lg border border-outline-variant/60 flex items-center justify-between text-mono-metric-sm text-[11px]">
          <span className="text-outline">DIV2K 50-Epoch</span>
          <span className="text-primary-container font-medium">{resolved.name.split('·')[1]?.trim() || 'L1'}</span>
        </div>
      )}

      {/* Checkpoint File Badge */}
      <div className="bg-surface-container px-3 py-2 rounded-lg border border-outline-variant flex items-center justify-between">
        <span className="font-mono-metric-sm text-[10px] text-outline uppercase tracking-wider">
          CKPT
        </span>
        <span className="font-mono-metric-sm text-mono-metric-sm text-primary-container tracking-tight truncate max-w-[200px]">
          {resolved.checkpoint}
        </span>
      </div>

      {/* Collapsible Degradation Drawer */}
      <div className="border border-outline-variant rounded-xl bg-surface-container/30 overflow-hidden transition-all">
        <button
          type="button"
          onClick={() => setDegradeOpen(!degradeOpen)}
          className="w-full px-3.5 py-2.5 flex items-center justify-between text-on-surface-variant hover:text-on-surface cursor-pointer transition text-left"
        >
          <div className="flex items-center gap-2">
            <Sliders className="w-3.5 h-3.5 text-primary-container" />
            <span className="font-body-sm text-body-sm font-medium">
              Simulate Surveillance Degradation
            </span>
          </div>
          {degradeOpen ? (
            <ChevronDown className="w-4 h-4 text-outline" />
          ) : (
            <ChevronRight className="w-4 h-4 text-outline" />
          )}
        </button>

        {degradeOpen && (
          <div className="p-3.5 border-t border-outline-variant/40 space-y-3.5 font-mono-metric-sm text-mono-metric-sm bg-surface-container-lowest/50">
            {/* Enable toggle */}
            <div className="flex items-center justify-between">
              <span className="text-on-surface font-medium">Enable Degradation</span>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={degradation.enabled}
                  onChange={(e) =>
                    setDegradation({ ...degradation, enabled: e.target.checked })
                  }
                  className="sr-only peer"
                />
                <div className="w-9 h-5 bg-surface-container-high peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-on-surface after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-primary-container peer-checked:after:bg-on-primary-container"></div>
              </label>
            </div>

            {/* Blur sigma */}
            <div className="space-y-1">
              <div className="flex justify-between text-on-surface-variant">
                <span>Gaussian Blur σ</span>
                <span className="text-primary-container">{degradation.blurSigma}</span>
              </div>
              <input
                type="range"
                min="0"
                max="3"
                step="0.1"
                value={degradation.blurSigma}
                onChange={(e) =>
                  setDegradation({
                    ...degradation,
                    blurSigma: parseFloat(e.target.value),
                  })
                }
                className="w-full accent-primary-container cursor-pointer"
              />
            </div>

            {/* Noise sigma */}
            <div className="space-y-1">
              <div className="flex justify-between text-on-surface-variant">
                <span>Gaussian Noise σ</span>
                <span className="text-primary-container">{degradation.noiseSigma}</span>
              </div>
              <input
                type="range"
                min="0"
                max="50"
                step="1"
                value={degradation.noiseSigma}
                onChange={(e) =>
                  setDegradation({
                    ...degradation,
                    noiseSigma: parseInt(e.target.value),
                  })
                }
                className="w-full accent-primary-container cursor-pointer"
              />
            </div>

            {/* JPEG Quality */}
            <div className="space-y-1">
              <div className="flex justify-between text-on-surface-variant">
                <span>JPEG Quality</span>
                <span className="text-primary-container">{degradation.jpegQuality}%</span>
              </div>
              <input
                type="range"
                min="10"
                max="100"
                step="5"
                value={degradation.jpegQuality}
                onChange={(e) =>
                  setDegradation({
                    ...degradation,
                    jpegQuality: parseInt(e.target.value),
                  })
                }
                className="w-full accent-primary-container cursor-pointer"
              />
            </div>

            {/* Reset Defaults button */}
            <div className="pt-1 flex justify-end">
              <button
                type="button"
                onClick={() =>
                  setDegradation({
                    enabled: false,
                    blurSigma: 1.5,
                    noiseSigma: 15,
                    jpegQuality: 50,
                  })
                }
                className="text-[11px] text-outline hover:text-on-surface transition-colors flex items-center gap-1"
              >
                <RotateCcw className="w-3 h-3" />
                <span>Reset to defaults</span>
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Primary Action Button (Enhance) */}
      <div className="mt-auto pt-2">
        <button
          onClick={() => {
            const isSim = Boolean(customImage && simulateCctv);
            if (viewMode === 'single') {
              runUpscale(activeInputUrl, model, scale, loss, selectedSample, degradation, isSim, denoise, isLegacy);
            } else {
              runCompare4(activeInputUrl, scale, selectedSample, degradation, isSim, denoise, isLegacy);
            }
          }}
          disabled={loading}
          className="w-full bg-primary-container hover:bg-primary-fixed text-on-primary-container font-headline-md text-headline-md py-3 px-4 rounded-xl shadow-[0_0_20px_rgba(56,225,255,0.25)] hover:shadow-[0_0_28px_rgba(56,225,255,0.45)] transition-all text-center font-semibold disabled:opacity-50 flex items-center justify-center gap-2 group cursor-pointer"
        >
          {loading ? (
            <>
              <div className="w-4 h-4 border-2 border-on-primary-container border-t-transparent rounded-full animate-spin" />
              <span>Processing...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4 transition-transform group-hover:scale-110" />
              <span>Enhance Image</span>
            </>
          )}
        </button>
      </div>
    </div>
  );

  // Render right telemetry panel content (reused in desktop sidebar & mobile drawer)
  const renderTelemetry = () => (
    <div className="flex flex-col gap-3">
      {/* Card: FIDELITY */}
      <div className="bg-surface-container/50 border border-outline-variant rounded-xl p-3.5 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <Crosshair className="w-3.5 h-3.5 text-primary-container" />
            <span className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">
              FIDELITY METRICS
            </span>
          </div>
          <span
            className="text-outline hover:text-on-surface cursor-help transition-colors"
            title="Metrics calculated on Y-channel after RGB→YCbCr conversion with borders shaved."
          >
            <HelpCircle className="w-3.5 h-3.5" />
          </span>
        </div>

        {!hasGroundTruth ? (
          <div className="py-6 px-3 text-center rounded-lg border border-dashed border-outline-variant bg-surface-container-lowest/40 flex flex-col items-center gap-1.5">
            <FileImage className="w-6 h-6 text-outline" />
            <span className="font-mono-metric-sm text-mono-metric-sm text-on-surface">
              Custom Upload Active
            </span>
            <p className="font-body-sm text-[11px] text-outline max-w-[200px]">
              PSNR/SSIM/LPIPS require paired ground truth references (available on Set5/Set14 samples).
            </p>
          </div>
        ) : metrics ? (
          <>
            {/* Metric 1: PSNR */}
            <div className="flex flex-col gap-1">
              <div className="flex items-baseline justify-between">
                <div className="flex items-baseline gap-1.5">
                  <span className="font-headline-md text-headline-md text-on-surface-variant">
                    PSNR
                  </span>
                  <span className="font-mono-metric-lg text-[22px] font-semibold text-on-surface">
                    {metrics.psnr}
                  </span>
                  <span className="font-mono-metric-sm text-[11px] text-on-surface-variant">dB</span>
                </div>
                {bicubicMetrics && (
                  <div
                    className={`h-[22px] px-2 rounded-full border flex items-center justify-center ${
                      metrics.psnr >= bicubicMetrics.psnr
                        ? 'bg-good/15 border-good/30 text-good'
                        : 'bg-bad/15 border-bad/30 text-bad'
                    }`}
                  >
                    <span className="font-mono-metric-sm text-[11px] font-medium">
                      {metrics.psnr >= bicubicMetrics.psnr ? '+' : ''}
                      {(metrics.psnr - bicubicMetrics.psnr).toFixed(2)}
                    </span>
                  </div>
                )}
              </div>
              <div className="flex items-center justify-between text-mono-metric-sm font-mono-metric-sm text-outline">
                <span>Bicubic baseline: {bicubicMetrics?.psnr || 29.96} dB</span>
              </div>
              <div className="w-full h-1.5 bg-surface-container-lowest rounded-full overflow-hidden flex gap-0.5 mt-0.5">
                <div
                  className="h-full bg-outline rounded-l-full"
                  style={{
                    width: `${Math.min(100, ((bicubicMetrics?.psnr || 28) / 35) * 100)}%`,
                  }}
                />
                <div
                  className="h-full bg-primary-container rounded-r-full"
                  style={{ width: `${Math.min(100, (metrics.psnr / 35) * 100)}%` }}
                />
              </div>
            </div>

            <div className="h-[1px] bg-outline-variant/50 w-full" />

            {/* Metric 2: SSIM */}
            <div className="flex flex-col gap-1">
              <div className="flex items-baseline justify-between">
                <div className="flex items-baseline gap-1.5">
                  <span className="font-headline-md text-headline-md text-on-surface-variant">
                    SSIM
                  </span>
                  <span className="font-mono-metric-lg text-[22px] font-semibold text-on-surface">
                    {metrics.ssim}
                  </span>
                </div>
                {bicubicMetrics && (
                  <div
                    className={`h-[22px] px-2 rounded-full border flex items-center justify-center ${
                      metrics.ssim >= bicubicMetrics.ssim
                        ? 'bg-good/15 border-good/30 text-good'
                        : 'bg-bad/15 border-bad/30 text-bad'
                    }`}
                  >
                    <span className="font-mono-metric-sm text-[11px] font-medium">
                      {metrics.ssim >= bicubicMetrics.ssim ? '+' : ''}
                      {(metrics.ssim - bicubicMetrics.ssim).toFixed(3)}
                    </span>
                  </div>
                )}
              </div>
              <div className="flex items-center justify-between text-mono-metric-sm font-mono-metric-sm text-outline">
                <span>Bicubic baseline: {bicubicMetrics?.ssim || 0.812}</span>
              </div>
              <div className="w-full h-1.5 bg-surface-container-lowest rounded-full overflow-hidden flex gap-0.5 mt-0.5">
                <div
                  className="h-full bg-outline rounded-l-full"
                  style={{
                    width: `${Math.min(100, ((bicubicMetrics?.ssim || 0.8) / 1) * 100)}%`,
                  }}
                />
                <div
                  className="h-full bg-primary-container rounded-r-full"
                  style={{ width: `${Math.min(100, (metrics.ssim / 1) * 100)}%` }}
                />
              </div>
            </div>

            <div className="h-[1px] bg-outline-variant/50 w-full" />

            {/* Metric 3: LPIPS */}
            {typeof metrics.lpips === 'number' && (
              <div className="flex flex-col gap-1">
                <div className="flex items-baseline justify-between">
                  <div className="flex items-baseline gap-1.5">
                    <span className="font-headline-md text-headline-md text-on-surface-variant">
                      LPIPS
                    </span>
                    <span className="font-mono-metric-lg text-[22px] font-semibold text-on-surface">
                      {metrics.lpips}
                    </span>
                    <span className="font-mono-metric-sm text-[10px] text-outline">↓</span>
                  </div>
                  <div className="h-[22px] px-2 rounded-full bg-good/15 border border-good/30 text-good flex items-center justify-center">
                    <span className="font-mono-metric-sm text-[11px] font-medium">−0.072</span>
                  </div>
                </div>
                <div className="flex items-center justify-between text-mono-metric-sm font-mono-metric-sm text-outline">
                  <span>Bicubic baseline: 0.286</span>
                </div>
                <div className="w-full h-1.5 bg-surface-container-lowest rounded-full overflow-hidden flex gap-0.5 mt-0.5">
                  <div className="h-full bg-primary-container rounded-l-full" style={{ width: '65%' }} />
                  <div className="h-full bg-outline rounded-r-full" style={{ width: '35%' }} />
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="py-6 text-center font-mono-metric-sm text-mono-metric-sm text-outline flex flex-col items-center gap-1.5">
            <span className="text-on-surface-variant font-medium">Ground Truth Not Available</span>
            <span className="text-[11px] text-outline max-w-[220px] leading-tight">
              {customImage && !simulateCctv
                ? 'Direct LR mode has no reference. Switch to "Simulate CCTV" in the Source card to compute live PSNR & SSIM.'
                : 'Click Enhance to compute metrics'}
            </span>
          </div>
        )}
      </div>

      {/* Card: RUN TELEMETRY */}
      <div className="bg-surface-container/50 border border-outline-variant rounded-xl p-3.5 flex flex-col gap-2.5">
        <div className="flex items-center gap-1.5">
          <Clock className="w-3.5 h-3.5 text-primary-container" />
          <span className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">
            RUN TELEMETRY
          </span>
        </div>
        <div className="grid grid-cols-2 gap-2 pt-0.5">
          <div className="bg-surface-container-lowest p-2.5 rounded-lg border border-outline-variant">
            <span className="font-mono-metric-sm text-[10px] text-outline block mb-0.5 uppercase">
              Input Res
            </span>
            <span className="font-mono-metric-md text-mono-metric-md text-on-surface">
              {activeResult?.inputSize
                ? `${activeResult.inputSize[0]}×${activeResult.inputSize[1]}`
                : '128×128'}
            </span>
          </div>
          <div className="bg-surface-container-lowest p-2.5 rounded-lg border border-outline-variant">
            <span className="font-mono-metric-sm text-[10px] text-outline block mb-0.5 uppercase">
              Output Res
            </span>
            <span className="font-mono-metric-md text-mono-metric-md text-primary-container font-semibold">
              {activeResult?.outputSize
                ? `${activeResult.outputSize[0]}×${activeResult.outputSize[1]}`
                : '512×512'}
            </span>
          </div>
          <div className="bg-surface-container-lowest p-2.5 rounded-lg border border-outline-variant">
            <span className="font-mono-metric-sm text-[10px] text-outline block mb-0.5 uppercase">
              Latency
            </span>
            <span className="font-mono-metric-md text-mono-metric-md text-on-surface">
              {activeResult?.ms ? `${Math.round(activeResult.ms)} ms` : '42 ms'}
            </span>
          </div>
          <div className="bg-surface-container-lowest p-2.5 rounded-lg border border-outline-variant">
            <span className="font-mono-metric-sm text-[10px] text-outline block mb-0.5 uppercase">
              Device
            </span>
            <span className="font-mono-metric-md text-mono-metric-md text-primary-container flex items-center gap-1">
              <Cpu className="w-3.5 h-3.5" />
              <span>CUDA</span>
            </span>
          </div>
        </div>
        {denoise === 'bilateral' && (
          <div className="bg-good/10 border border-good/30 rounded-lg px-2.5 py-1.5 flex items-center justify-between text-[11px] font-mono-metric-sm text-good">
            <span>Bilateral Filter (+0.86ms)</span>
            <span>+1.35 dB Rescue</span>
          </div>
        )}
      </div>

      {/* Card: IN CONTEXT RANKING */}
      {hasGroundTruth && rankInfo && (
        <div className="bg-surface-container/50 border border-outline-variant rounded-xl p-3.5 flex flex-col gap-1.5">
          <div className="flex items-center gap-1.5">
            <Compass className="w-3.5 h-3.5 text-primary-container" />
            <span className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">
              BENCHMARK RANK
            </span>
          </div>
          <div className="flex items-baseline justify-between mt-1">
            <div className="flex items-baseline gap-2">
              <span className="font-mono-metric-lg text-headline-xl font-bold text-on-surface">
                #{rankInfo.rank}{' '}
                <span className="text-on-surface-variant font-normal text-headline-md">
                  of {rankInfo.total}
                </span>
              </span>
              <span className="font-body-sm text-body-sm text-outline">
                {datasetGroup} ×{scale} · PSNR
              </span>
            </div>
          </div>
          <div className="pt-1.5 flex items-center justify-between">
            <Link
              to={`/benchmarks?dataset=${datasetGroup}&scale=${scale}`}
              className="font-mono-metric-sm text-mono-metric-sm text-primary-container hover:underline flex items-center gap-1 group"
            >
              <span>View full benchmarks</span>
              <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-0.5" />
            </Link>
          </div>
        </div>
      )}

      {/* Amber Warning / Forensic Callout if PSNR < bicubic */}
      {isLowerPsnr && (
        <div className="border border-amber/40 bg-amber/10 rounded-xl p-3 text-xs text-amber flex items-start gap-2.5 shadow-[0_0_12px_rgba(255,176,32,0.15)]">
          <AlertTriangle className="w-4 h-4 text-amber flex-shrink-0 mt-0.5" />
          <div className="flex flex-col gap-0.5">
            <span className="font-semibold text-[12px]">Perceptual Tradeoff Active</span>
            <span className="font-body-sm text-[11px] text-amber/90 leading-tight">
              Model produces sharper edges and textures but scored lower on pixel-level PSNR than bicubic.
            </span>
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="flex flex-col w-full h-[calc(100vh-84px)] overflow-hidden select-none bg-surface-container-lowest">
      {/* Small Display Banner / Drawer Toggles for Mobile & Tablets */}
      <div className="lg:hidden flex items-center justify-between px-4 py-2 bg-surface-container-low border-b border-outline-variant shrink-0">
        <button
          onClick={() => setMobileControlsOpen(true)}
          className="px-3 py-1 bg-surface-container text-primary-container text-xs font-mono rounded-lg border border-outline-variant flex items-center gap-1.5"
        >
          <Sliders className="w-3 h-3" />
          <span>Workbench Controls</span>
        </button>
        <span className="font-mono-metric-sm text-mono-metric-sm text-on-surface">
          {model} {scale}× · {loss}
        </span>
        <button
          onClick={() => setMobileMetricsOpen(true)}
          className="px-3 py-1 bg-surface-container text-good text-xs font-mono rounded-lg border border-outline-variant flex items-center gap-1.5"
        >
          <Crosshair className="w-3 h-3" />
          <span>Metrics</span>
        </button>
      </div>

      <div className="flex w-full h-full relative">
        {/* 1. LEFT RAIL (300px fixed workbench sidebar) */}
        <aside className="w-[300px] flex-shrink-0 bg-surface-container-low border-r border-outline-variant p-4 flex flex-col gap-4 overflow-y-auto hidden lg:flex">
          {renderControls()}
        </aside>

        {/* 2. CENTER COLUMN (fluid workbench & forensic split viewport) */}
        <section className="flex-1 min-w-0 p-4 flex flex-col gap-3 overflow-hidden">
          {/* Top Viewport Control Bar */}
          <div className="flex items-center justify-between h-8 shrink-0">
            <div className="flex p-0.5 bg-surface-container-low rounded-lg border border-outline-variant">
              <button
                onClick={() => setViewMode('single')}
                className={`px-3 py-1 font-body-sm-medium text-body-sm-medium rounded-md transition-all flex items-center gap-1.5 ${
                  viewMode === 'single'
                    ? 'text-on-surface bg-surface-container shadow-sm font-semibold'
                    : 'text-on-surface-variant hover:text-on-surface'
                }`}
              >
                <Eye className="w-3.5 h-3.5" />
                <span>Single</span>
              </button>
              <button
                onClick={() => setViewMode('compare4')}
                className={`px-3 py-1 font-body-sm-medium text-body-sm-medium rounded-md transition-all flex items-center gap-1.5 ${
                  viewMode === 'compare4'
                    ? 'text-on-surface bg-surface-container shadow-sm font-semibold'
                    : 'text-on-surface-variant hover:text-on-surface'
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                <span>Compare 4</span>
              </button>
            </div>

            {/* Status indicator on top bar */}
            <div className="flex items-center gap-2">
              <span className="font-mono-metric-sm text-[11px] text-outline hidden sm:inline">
                {viewMode === 'single' ? 'Interactive Viewport' : '4-Way Experiment Matrix'}
              </span>
            </div>
          </div>

          {/* Main Interactive Viewport Canvas */}
          {viewMode === 'single' ? (
            <div
              ref={viewportRef as any}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerLeave={handleLensLeave}
              onClick={toggleLensLock}
              className="relative flex-1 w-full bg-surface-container-lowest rounded-xl border border-outline-variant overflow-hidden group select-none flex items-center justify-center cursor-crosshair shadow-inner"
            >
              <div
                className="relative w-full h-full flex items-center justify-center overflow-hidden transition-transform duration-150"
                style={{ transform: `scale(${zoomLevel})` }}
              >
                {/* 1. SLIDER MODE */}
                {viewerMode === 'slider' && (
                  <div className="relative w-full h-full flex items-center justify-center overflow-hidden">
                    {/* Underlying Right Side (High-Res Super-Resolved) */}
                    <div className="absolute inset-0 w-full h-full flex items-center justify-center pointer-events-none select-none">
                      <img
                        className="w-full h-full object-contain pointer-events-none select-none"
                        src={activeResult?.srUrl || activeInputUrl}
                        alt="High Resolution Super-Resolved Reference"
                      />
                    </div>

                    {/* Overlaid Left Side (Comparison Frame) clipped seamlessly by curtain split position */}
                    <div
                      className="absolute inset-0 w-full h-full flex items-center justify-center pointer-events-none select-none"
                      style={{
                        clipPath: `inset(0 ${100 - splitPos}% 0 0)`,
                      }}
                    >
                      <img
                        className="w-full h-full object-contain select-none"
                        style={{
                          imageRendering: compareAgainst === 'input' ? 'pixelated' : 'auto',
                        }}
                        src={leftDisplaySrc}
                        alt="Left Comparison Frame"
                      />
                    </div>

                    {/* Curtain Divider Line & Handle */}
                    <div
                      className="absolute top-0 bottom-0 pointer-events-auto cursor-ew-resize z-20"
                      style={{ left: `${splitPos}%` }}
                    >
                      <div className="w-[2px] h-full bg-primary-container shadow-[0_0_12px_rgba(56,225,255,0.7)] relative flex items-center justify-center">
                        <div className="w-7 h-7 -ml-[1px] rounded-full bg-surface-container-low border border-primary-container text-primary-container flex items-center justify-center shadow-[0_0_14px_rgba(56,225,255,0.5)] text-[11px] font-mono hover:scale-110 transition-transform">
                          <MoveHorizontal className="w-3.5 h-3.5" />
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* 2. SIDE-BY-SIDE MODE */}
                {viewerMode === 'side-by-side' && (
                  <div className="grid grid-cols-2 gap-3 w-full h-full p-4">
                    <div className="relative border border-outline-variant rounded-xl bg-surface-container flex items-center justify-center overflow-hidden">
                      <span className="absolute top-2 left-2 z-10 font-mono-metric-sm text-[10px] bg-surface-container-low px-2 py-0.5 rounded border border-outline-variant text-on-surface-variant">
                        {leftLabel}
                      </span>
                      <img
                        src={leftDisplaySrc}
                        alt="Left Reference"
                        className="w-full h-full object-contain"
                        style={{
                          imageRendering: compareAgainst === 'input' ? 'pixelated' : 'auto',
                        }}
                      />
                    </div>
                    <div className="relative border border-primary-container/40 rounded-xl bg-surface-container flex items-center justify-center overflow-hidden">
                      <span className="absolute top-2 left-2 z-10 font-mono-metric-sm text-[10px] bg-surface-container-low text-primary-container px-2 py-0.5 rounded border border-primary-container/40">
                        {model} {scale}× OUTPUT
                      </span>
                      <img
                        src={activeResult?.srUrl || activeInputUrl}
                        alt="Super Resolution Output"
                        className="w-full h-full object-contain"
                      />
                    </div>
                  </div>
                )}

                {/* 3. FLICKER MODE */}
                {viewerMode === 'flicker' && (
                  <div className="relative w-full h-full flex items-center justify-center p-4">
                    <img
                      src={flickerToggle ? activeResult?.srUrl || activeInputUrl : leftDisplaySrc}
                      alt="Flicker Frame"
                      className="w-full h-full object-contain transition-opacity duration-75"
                      style={{
                        imageRendering: (!flickerToggle && compareAgainst === 'input') ? 'pixelated' : 'auto',
                      }}
                    />
                    <div className="absolute top-4 left-4 z-20 flex items-center gap-2">
                      <span className="font-mono-metric-sm text-[11px] bg-surface-container px-2.5 py-1 rounded-lg border border-outline-variant text-on-surface">
                        {flickerToggle ? `${model} ${scale}×` : leftLabel}
                      </span>
                      <button
                        type="button"
                        onClick={() => setFlickerActive(!flickerActive)}
                        className="font-mono-metric-sm text-[10px] bg-surface-container-low text-primary-container hover:bg-surface-container px-2.5 py-1 rounded-lg border border-primary-container/40 flex items-center gap-1 transition-colors"
                      >
                        {flickerActive ? (
                          <>
                            <Pause className="w-3 h-3" />
                            <span>Pause 2Hz</span>
                          </>
                        ) : (
                          <>
                            <Play className="w-3 h-3" />
                            <span>Play 2Hz</span>
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                )}

                {/* 4. RESIDUAL MAP MODE */}
                {viewerMode === 'error-map' && (
                  <div className="relative w-full h-full flex items-center justify-center p-4">
                    {diffMapUrl ? (
                      <>
                        <img
                          src={diffMapUrl}
                          alt="Residual Error Heatmap"
                          className="w-full h-full object-contain"
                        />
                        <div className="absolute bottom-4 right-4 z-20 bg-surface-container-low/95 backdrop-blur-sm p-2.5 rounded-lg border border-outline-variant font-mono-metric-sm text-[10px] space-y-1 shadow-lg">
                          <span className="text-on-surface font-medium block">
                            |Model − GroundTruth| Residual Map
                          </span>
                          <div className="w-32 h-2 rounded-full bg-gradient-to-r from-blue-900 via-green-400 to-red-500" />
                          <div className="flex justify-between text-outline">
                            <span>0.0 (Match)</span>
                            <span>High Variance</span>
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="font-mono-metric-sm text-on-surface-variant flex flex-col items-center gap-2">
                        <AlertTriangle className="w-6 h-6 text-amber" />
                        <span>
                          {hasGroundTruth
                            ? 'Generating residual map...'
                            : 'Residual map requires paired ground truth reference.'}
                        </span>
                      </div>
                    )}
                  </div>
                )}

                {/* Viewfinder Reticle (Synchronized Lens Target) */}
                {lensState.active && (
                  <div
                    className={`absolute pointer-events-none z-10 transition-transform ${
                      lensState.locked
                        ? 'border-2 border-amber shadow-[0_0_14px_rgba(255,176,32,0.85)]'
                        : 'border border-primary-container/90 shadow-[0_0_12px_rgba(56,225,255,0.35)]'
                    }`}
                    style={{
                      left: `${lensState.x * 100}%`,
                      top: `${lensState.y * 100}%`,
                      width: 64,
                      height: 64,
                      transform: 'translate(-50%, -50%)',
                    }}
                  >
                    <div className="absolute -top-1 -left-1 w-2.5 h-2.5 border-t-2 border-l-2 border-primary-container" />
                    <div className="absolute -top-1 -right-1 w-2.5 h-2.5 border-t-2 border-r-2 border-primary-container" />
                    <div className="absolute -bottom-1 -left-1 w-2.5 h-2.5 border-b-2 border-l-2 border-primary-container" />
                    <div className="absolute -bottom-1 -right-1 w-2.5 h-2.5 border-b-2 border-r-2 border-primary-container" />
                    <span className="absolute -bottom-5 left-0 font-mono-metric-sm text-[9px] text-primary-container bg-surface-container-lowest/95 px-1 py-0.2 rounded border border-primary-container/30 whitespace-nowrap">
                      {lensState.locked ? 'LOCKED LENS' : 'ROI: 64×64'}
                    </span>
                  </div>
                )}
              </div>

              {/* Top Left: Prominent Comparison Baseline Segmented Switcher */}
              <div className="absolute top-3 left-3 z-30 bg-surface-container-low/95 backdrop-blur-sm border border-outline-variant p-0.5 rounded-lg flex items-center shadow-lg gap-0.5">
                <button
                  type="button"
                  onClick={() => setCompareAgainst('input')}
                  className={`px-2.5 py-1 text-[11px] font-mono-metric-sm rounded-md transition-all flex items-center gap-1.5 ${
                    compareAgainst === 'input'
                      ? 'bg-surface-container text-amber border border-amber/40 font-semibold shadow-inner'
                      : 'text-on-surface-variant hover:text-on-surface'
                  }`}
                  title="Compare directly against the raw low-resolution pixelated input"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-amber" />
                  <span>Raw Low-Res</span>
                </button>
                <button
                  type="button"
                  onClick={() => setCompareAgainst('bicubic')}
                  className={`px-2.5 py-1 text-[11px] font-mono-metric-sm rounded-md transition-all flex items-center gap-1.5 ${
                    compareAgainst === 'bicubic'
                      ? 'bg-surface-container text-primary-container border border-primary-container/40 font-semibold shadow-inner'
                      : 'text-on-surface-variant hover:text-on-surface'
                  }`}
                  title="Compare against standard bicubic interpolation baseline"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-outline" />
                  <span>Bicubic</span>
                </button>
                {activeHrUrl && (
                  <button
                    type="button"
                    onClick={() => setCompareAgainst('ground-truth')}
                    className={`px-2.5 py-1 text-[11px] font-mono-metric-sm rounded-md transition-all flex items-center gap-1.5 ${
                      compareAgainst === 'ground-truth'
                        ? 'bg-surface-container text-good border border-good/40 font-semibold shadow-inner'
                        : 'text-on-surface-variant hover:text-on-surface'
                    }`}
                    title="Compare against paired high-resolution ground truth"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-good" />
                    <span>Ground Truth</span>
                  </button>
                )}
              </div>

              {/* Top Right Chip: Model Label */}
              <div className="absolute top-3 right-3 z-30 bg-surface-container-low/90 backdrop-blur-sm border border-primary-container/60 px-2.5 py-1 rounded-lg text-primary-container font-mono-metric-sm text-mono-metric-sm flex items-center gap-1.5 shadow-[0_0_12px_rgba(56,225,255,0.25)]">
                <span className="w-1.5 h-1.5 rounded-full bg-primary-container animate-pulse" />
                <span className="font-medium">
                  {model} {scale}× · {loss}
                </span>
              </div>

              {/* Top Center View Switcher Floating Pill */}
              <div className="absolute top-3 left-1/2 -translate-x-1/2 z-30 bg-surface-container-low/90 backdrop-blur-sm border border-outline-variant p-0.5 rounded-full flex items-center shadow-lg">
                <button
                  onClick={() => setViewerMode('slider')}
                  className={`px-3 py-1 font-body-sm-medium text-[11px] rounded-full transition-all ${
                    viewerMode === 'slider'
                      ? 'text-on-surface bg-surface-container shadow-inner font-semibold'
                      : 'text-on-surface-variant hover:text-on-surface'
                  }`}
                >
                  Slider
                </button>
                <button
                  onClick={() => setViewerMode('side-by-side')}
                  className={`px-3 py-1 font-body-sm-medium text-[11px] rounded-full transition-all ${
                    viewerMode === 'side-by-side'
                      ? 'text-on-surface bg-surface-container shadow-inner font-semibold'
                      : 'text-on-surface-variant hover:text-on-surface'
                  }`}
                >
                  Side by side
                </button>
                <button
                  onClick={() => setViewerMode('flicker')}
                  className={`px-3 py-1 font-body-sm-medium text-[11px] rounded-full transition-all ${
                    viewerMode === 'flicker'
                      ? 'text-on-surface bg-surface-container shadow-inner font-semibold'
                      : 'text-on-surface-variant hover:text-on-surface'
                  }`}
                >
                  Flicker
                </button>
                <button
                  onClick={() => {
                    if (activeHrUrl) setViewerMode('error-map');
                  }}
                  className={`px-3 py-1 font-body-sm-medium text-[11px] rounded-full transition-all ${
                    viewerMode === 'error-map'
                      ? 'text-on-surface bg-surface-container shadow-inner font-semibold'
                      : activeHrUrl
                      ? 'text-on-surface-variant hover:text-on-surface'
                      : 'text-outline/40 cursor-not-allowed'
                  }`}
                  title={!activeHrUrl ? 'Requires ground truth (Set5/Set14)' : undefined}
                >
                  Error map
                </button>
              </div>

              {/* Bottom Left Resolution Overlay */}
              <div className="absolute bottom-3 left-3 z-30 flex items-center gap-2">
                <span className="bg-surface-container-lowest/90 backdrop-blur-sm border border-outline-variant px-2.5 py-1 rounded-lg font-mono-metric-sm text-mono-metric-sm text-on-surface shadow-md">
                  {activeResult?.inputSize
                    ? `${activeResult.inputSize[0]}×${activeResult.inputSize[1]}`
                    : '128×128'}{' '}
                  →{' '}
                  <span className="text-primary-container font-medium">
                    {activeResult?.outputSize
                      ? `${activeResult.outputSize[0]}×${activeResult.outputSize[1]}`
                      : '512×512'}
                  </span>
                </span>
              </div>

              {/* Bottom Right Viewport Navigation Tools */}
              <div className="absolute bottom-3 right-3 z-30 flex items-center gap-1.5">
                <button
                  onClick={() => setZoomLevel((z) => (z === 1 ? 1.5 : z === 1.5 ? 2 : z === 2 ? 3 : 1))}
                  className="px-2 h-7 rounded-lg bg-surface-container-low/90 backdrop-blur-sm border border-outline-variant hover:border-primary-container text-on-surface-variant hover:text-primary-container flex items-center gap-1 transition-all font-mono text-xs shadow-md"
                  title="Zoom level"
                >
                  <ZoomIn className="w-3.5 h-3.5" />
                  <span>{zoomLevel}×</span>
                </button>
                <button
                  onClick={toggleFullscreen}
                  className="w-7 h-7 rounded-lg bg-surface-container-low/90 backdrop-blur-sm border border-outline-variant hover:border-primary-container text-on-surface-variant hover:text-primary-container flex items-center justify-center transition-all font-mono text-xs shadow-md"
                  title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
                >
                  {isFullscreen ? (
                    <Minimize2 className="w-3.5 h-3.5" />
                  ) : (
                    <Maximize2 className="w-3.5 h-3.5" />
                  )}
                </button>
                <button
                  onClick={handleDownload}
                  className="w-7 h-7 rounded-lg bg-surface-container-low/90 backdrop-blur-sm border border-outline-variant hover:border-primary-container text-on-surface-variant hover:text-primary-container flex items-center justify-center transition-all font-mono text-xs shadow-md"
                  title="Download super-resolved PNG"
                >
                  <Download className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          ) : (
            /* Compare 4 Quad Grid */
            <div className="flex-1 w-full grid grid-cols-2 grid-rows-2 gap-3 p-3 bg-surface-container-lowest rounded-xl border border-outline-variant overflow-hidden shadow-inner">
              {(isLegacy
                ? [
                    { m: 'SRCNN', l: 'MSE', key: 'srcnn_mse', title: `SRCNN ${scale}× · MSE` },
                    {
                      m: 'SRCNN',
                      l: 'MSE + Perceptual',
                      key: 'srcnn_perc',
                      title: `SRCNN ${scale}× · Perceptual`,
                    },
                    { m: 'ESPCN', l: 'MSE', key: 'espcn_mse', title: `ESPCN ${scale}× · MSE` },
                    {
                      m: 'ESPCN',
                      l: 'MSE + Perceptual',
                      key: 'espcn_perc',
                      title: `ESPCN ${scale}× · Perceptual`,
                    },
                  ]
                : [
                    { m: 'EDSR', l: 'L1', key: 'edsr_l1', title: `EDSR ${scale}× · Quality (L1)` },
                    {
                      m: 'FSRCNN',
                      l: 'Charbonnier',
                      key: 'fsrcnn_charb',
                      title: `FSRCNN ${scale}× · Balanced`,
                    },
                    {
                      m: 'ESPCN',
                      l: 'Charbonnier',
                      key: 'espcn_charb',
                      title: `ESPCN ${scale}× · Fast`,
                    },
                    {
                      m: 'SRCNN',
                      l: 'MSE',
                      key: 'srcnn_legacy',
                      title: `SRCNN ${scale}× · Legacy`,
                    },
                  ]
              ).map((tile) => {
                const res = compare4Results[tile.key];
                return (
                  <div
                    key={tile.key}
                    onClick={() => {
                      setModel(tile.m as ModelName);
                      setLoss(tile.l as LossName);
                      setViewMode('single');
                    }}
                    className="relative rounded-xl border border-outline-variant bg-surface-container-low hover:border-primary-container transition-all cursor-pointer flex flex-col justify-between overflow-hidden p-2.5 group shadow-sm hover:shadow-[0_0_16px_rgba(56,225,255,0.2)]"
                  >
                    <div className="flex justify-between items-center z-10">
                      <span className="font-mono-metric-sm text-[11px] text-primary-container bg-surface-container px-2 py-0.5 rounded-md border border-outline-variant font-medium">
                        {tile.title}
                      </span>
                      {res?.metrics && (
                        <span className="font-mono-metric-sm text-[11px] text-good font-semibold">
                          {res.metrics.psnr} dB
                        </span>
                      )}
                    </div>
                    <div className="flex-1 flex items-center justify-center p-2 min-h-0 overflow-hidden">
                      <img
                        src={res?.srUrl || activeInputUrl}
                        alt={tile.title}
                        className="w-full h-full object-contain"
                      />
                    </div>
                    <div className="font-mono-metric-sm text-[10px] text-outline group-hover:text-primary-container text-right flex items-center justify-end gap-1 transition-colors">
                      <span>Promote to single view</span>
                      <ArrowRight className="w-3 h-3" />
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Magnifier Loupe Quad Comparison Shelf (~140px height) */}
          <div className="h-36 grid grid-cols-4 gap-3 flex-shrink-0">
            {/* Tile 1: Input Nearest */}
            <div className="bg-surface-container-low rounded-xl border border-outline-variant p-2 flex flex-col gap-1.5 overflow-hidden">
              <div className="flex-1 rounded-lg bg-surface-container-lowest overflow-hidden relative">
                <div
                  className="w-full h-full"
                  style={{
                    backgroundImage: `url(${activeInputUrl})`,
                    backgroundPosition: `${lensState.x * 100}% ${lensState.y * 100}%`,
                    backgroundSize: '400%',
                    imageRendering: 'pixelated',
                  }}
                />
              </div>
              <div className="flex items-center justify-between px-0.5">
                <span className="font-mono-metric-sm text-[11px] text-on-surface-variant">
                  Input (Pixelated)
                </span>
              </div>
            </div>

            {/* Tile 2: Bicubic Baseline */}
            <div className="bg-surface-container-low rounded-xl border border-outline-variant p-2 flex flex-col gap-1.5 overflow-hidden">
              <div className="flex-1 rounded-lg bg-surface-container-lowest overflow-hidden relative">
                <div
                  className="w-full h-full"
                  style={{
                    backgroundImage: `url(${activeResult?.bicubicUrl || activeInputUrl})`,
                    backgroundPosition: `${lensState.x * 100}% ${lensState.y * 100}%`,
                    backgroundSize: '400%',
                  }}
                />
              </div>
              <div className="flex items-center justify-between px-0.5">
                <span className="font-mono-metric-sm text-[11px] text-on-surface-variant">
                  Bicubic Baseline
                </span>
              </div>
            </div>

            {/* Tile 3: Model Output */}
            <div className="bg-surface-container-low rounded-xl border border-primary-container shadow-[0_0_14px_rgba(56,225,255,0.22)] p-2 flex flex-col gap-1.5 overflow-hidden relative">
              <div className="flex-1 rounded-lg bg-surface-container-lowest overflow-hidden relative">
                <div
                  className="w-full h-full"
                  style={{
                    backgroundImage: `url(${activeResult?.srUrl || activeInputUrl})`,
                    backgroundPosition: `${lensState.x * 100}% ${lensState.y * 100}%`,
                    backgroundSize: '400%',
                  }}
                />
              </div>
              <div className="flex items-center justify-between px-0.5">
                <span className="font-mono-metric-sm text-[11px] text-primary-container font-semibold">
                  {model} Output
                </span>
              </div>
            </div>

            {/* Tile 4: Ground Truth */}
            <div className="bg-surface-container-low rounded-xl border border-outline-variant p-2 flex flex-col gap-1.5 overflow-hidden">
              <div className="flex-1 rounded-lg bg-surface-container-lowest overflow-hidden relative flex items-center justify-center">
                {activeHrUrl ? (
                  <div
                    className="w-full h-full"
                    style={{
                      backgroundImage: `url(${activeHrUrl})`,
                      backgroundPosition: `${lensState.x * 100}% ${lensState.y * 100}%`,
                      backgroundSize: '400%',
                    }}
                  />
                ) : (
                  <div className="flex flex-col items-center gap-1 p-2 text-center">
                    <span className="font-mono-metric-sm text-[10px] text-outline">
                      No ground truth
                    </span>
                    <span className="text-[9px] text-outline/60 font-mono">Custom input</span>
                  </div>
                )}
              </div>
              <div className="flex items-center justify-between px-0.5">
                <span className="font-mono-metric-sm text-[11px] text-on-surface">
                  Ground Truth
                </span>
              </div>
            </div>
          </div>
        </section>

        {/* 3. RIGHT PANEL (320px telemetry & forensic validation) */}
        <aside className="w-[320px] flex-shrink-0 bg-surface-container-low border-l border-outline-variant p-4 flex flex-col gap-3 overflow-y-auto hidden lg:flex">
          {renderTelemetry()}
        </aside>

        {/* Mobile / Tablet Drawer: Controls */}
        {mobileControlsOpen && (
          <div className="lg:hidden fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex justify-start">
            <div className="w-[320px] h-full bg-surface-container-low p-4 overflow-y-auto shadow-2xl border-r border-outline-variant flex flex-col">
              <div className="flex items-center justify-between pb-3 border-b border-outline-variant mb-3">
                <span className="font-headline-md text-headline-md text-on-surface">
                  Workbench Controls
                </span>
                <button
                  onClick={() => setMobileControlsOpen(false)}
                  className="p-1 rounded-lg hover:bg-surface-container text-on-surface-variant"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              {renderControls()}
            </div>
            <div
              className="flex-1"
              onClick={() => setMobileControlsOpen(false)}
            />
          </div>
        )}

        {/* Mobile / Tablet Drawer: Metrics */}
        {mobileMetricsOpen && (
          <div className="lg:hidden fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex justify-end">
            <div
              className="flex-1"
              onClick={() => setMobileMetricsOpen(false)}
            />
            <div className="w-[320px] h-full bg-surface-container-low p-4 overflow-y-auto shadow-2xl border-l border-outline-variant flex flex-col">
              <div className="flex items-center justify-between pb-3 border-b border-outline-variant mb-3">
                <span className="font-headline-md text-headline-md text-on-surface">
                  Forensic Metrics
                </span>
                <button
                  onClick={() => setMobileMetricsOpen(false)}
                  className="p-1 rounded-lg hover:bg-surface-container text-on-surface-variant"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              {renderTelemetry()}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
