/**
 * Image comparison workspace: LR input vs bicubic baseline vs SR output,
 * with an interactive before/after slider and real-time detail loupe.
 */
import React, { useState, useRef, useCallback } from 'react';
import {
  Image as ImageIcon,
  Maximize,
  Minimize,
  Layout as LayoutIcon,
  Sliders,
  ZoomIn,
  Download,
  Grid,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, Badge, Button } from '../components/ui';

export default function Comparison({ lrImage, bicubic, srImage, srMeta, onDownload }) {
  const [zoomed, setZoomed] = useState(null); // 'lr' | 'bicubic' | 'sr' | null
  const [slider, setSlider] = useState(50); // percent (0 to 100)
  const [layout, setLayout] = useState('slider'); // default to 'slider' for immediate comparison
  const [beforeSource, setBeforeSource] = useState('lr'); // 'lr' | 'bicubic'
  const [pixelated, setPixelated] = useState(true); // pixelated rendering for LR
  const [showLoupe, setShowLoupe] = useState(false);

  if (!srImage && !bicubic && !lrImage) return null;

  const closeZoom = () => setZoomed(null);
  const activeBefore = beforeSource === 'bicubic' && bicubic ? bicubic : lrImage;
  const beforeLabel = beforeSource === 'bicubic' && bicubic ? 'Bicubic Baseline' : 'LR Input (Raw)';

  return (
    <Card className="overflow-hidden">
      <CardHeader className="flex-wrap justify-between gap-3 border-b border-slate-100 pb-4 mb-4">
        <div className="flex items-center gap-2">
          <Sliders className="w-5 h-5 text-blue-600" />
          <CardTitle>Super-Resolution Visual Inspection</CardTitle>
          {srMeta?.scale && (
            <Badge variant="brand" className="font-mono text-xs">
              {srMeta.scale}× Upscale
            </Badge>
          )}
        </div>

        {/* View Layout Controls */}
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex bg-slate-100 p-0.5 rounded-xl text-xs font-medium">
            <button
              onClick={() => setLayout('slider')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-all duration-150 ${
                layout === 'slider'
                  ? 'bg-white text-blue-700 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Sliders size={13} />
              Before / After Slider
            </button>
            <button
              onClick={() => setLayout('side')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-all duration-150 ${
                layout === 'side'
                  ? 'bg-white text-blue-700 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <LayoutIcon size={13} />
              Side-by-Side
            </button>
          </div>

          {/* Loupe Toggle */}
          <button
            onClick={() => setShowLoupe((prev) => !prev)}
            className={`flex items-center gap-1 px-2.5 py-1.5 rounded-xl border text-xs font-medium transition-all ${
              showLoupe
                ? 'bg-blue-50 border-blue-200 text-blue-700 shadow-sm'
                : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
            }`}
            title="Toggle Magnifier Loupe"
          >
            <ZoomIn size={14} />
            <span className="hidden sm:inline">Magnifier</span>
          </button>
        </div>
      </CardHeader>

      {/* Control Toolbar for Slider */}
      {layout === 'slider' && (
        <div className="flex items-center justify-between flex-wrap gap-3 mb-4 px-1 text-xs">
          <div className="flex items-center gap-2">
            <span className="font-medium text-slate-500 uppercase tracking-wider text-[11px]">
              Compare Against:
            </span>
            <div className="inline-flex rounded-lg border border-slate-200 p-0.5 bg-slate-50">
              <button
                onClick={() => setBeforeSource('lr')}
                className={`px-2.5 py-1 rounded-md transition-colors ${
                  beforeSource === 'lr'
                    ? 'bg-white text-slate-800 shadow-xs font-medium'
                    : 'text-slate-500 hover:text-slate-800'
                }`}
              >
                Raw LR Sensor
              </button>
              {bicubic && (
                <button
                  onClick={() => setBeforeSource('bicubic')}
                  className={`px-2.5 py-1 rounded-md transition-colors ${
                    beforeSource === 'bicubic'
                      ? 'bg-white text-slate-800 shadow-xs font-medium'
                      : 'text-slate-500 hover:text-slate-800'
                  }`}
                >
                  Bicubic Interpolation
                </button>
              )}
            </div>
          </div>

          <div className="flex items-center gap-3">
            {beforeSource === 'lr' && (
              <label className="flex items-center gap-1.5 cursor-pointer text-slate-600 select-none">
                <input
                  type="checkbox"
                  checked={pixelated}
                  onChange={(e) => setPixelated(e.target.checked)}
                  className="rounded text-blue-600 focus:ring-blue-500"
                />
                <span>Pixelated LR (CCTV Raw Pixels)</span>
              </label>
            )}
            <span className="font-mono text-slate-400">{slider.toFixed(0)}%</span>
          </div>
        </div>
      )}

      {/* Main Workspace Display */}
      {layout === 'side' ? (
        <SideBySide
          lrImage={lrImage}
          bicubic={bicubic}
          srImage={srImage}
          onZoom={setZoomed}
          onDownload={onDownload}
          srMeta={srMeta}
          pixelated={pixelated}
        />
      ) : (
        <BeforeAfterSlider
          beforeImage={activeBefore}
          afterImage={srImage}
          beforeLabel={beforeLabel}
          afterLabel={srMeta?.model_id ? `${srMeta.model_id} (${srMeta.scale}×)` : 'Neural SR'}
          slider={slider}
          onSlider={setSlider}
          pixelated={beforeSource === 'lr' && pixelated}
          showLoupe={showLoupe}
        />
      )}

      {/* Zoom Modal */}
      {zoomed && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4 backdrop-blur-sm"
          onClick={closeZoom}
        >
          <div
            className="relative bg-white rounded-2xl p-3 shadow-2xl max-w-[92vw] max-h-[92vh] flex flex-col items-center"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={closeZoom}
              className="absolute top-3 right-3 w-8 h-8 rounded-full bg-slate-100 hover:bg-slate-200 text-slate-600 flex items-center justify-center transition-colors"
            >
              ✕
            </button>
            <img
              src={zoomed === 'lr' ? lrImage : zoomed === 'bicubic' ? bicubic : srImage}
              alt="zoom"
              className="max-w-[88vw] max-h-[78vh] object-contain rounded-xl bg-slate-50"
              style={{
                imageRendering: zoomed === 'lr' && pixelated ? 'pixelated' : 'auto',
              }}
            />
            <div className="mt-3 text-xs font-medium text-slate-600 uppercase tracking-wider">
              {zoomed === 'lr' ? 'LR Input' : zoomed === 'bicubic' ? 'Bicubic Baseline' : 'SR Output'}
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}

function SideBySide({ lrImage, bicubic, srImage, onZoom, onDownload, srMeta, pixelated }) {
  const panels = [
    {
      key: 'lr',
      label: 'LR Input',
      sub: 'Low-Resolution Raw',
      src: lrImage,
      meta: null,
      pixelated: pixelated,
    },
    {
      key: 'bicubic',
      label: 'Bicubic Baseline',
      sub: 'Standard Interpolation',
      src: bicubic,
      meta: null,
      pixelated: false,
    },
    {
      key: 'sr',
      label: 'Neural SR Output',
      sub: srMeta?.architecture ? `${srMeta.architecture.toUpperCase()} Deep Learning` : 'Enhanced',
      src: srImage,
      meta: srMeta,
      pixelated: false,
    },
  ].filter((p) => p.src);

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {panels.map((p) => (
        <div
          key={p.key}
          className="rounded-2xl border border-slate-200 bg-white overflow-hidden group hover:shadow-lg transition-all duration-200 flex flex-col"
        >
          <div className="px-4 py-3 bg-slate-50/80 border-b border-slate-100 flex items-center justify-between">
            <div>
              <span className="text-xs font-semibold text-slate-800">{p.label}</span>
              <p className="text-[10px] text-slate-500">{p.sub}</p>
            </div>
            {p.meta?.scale && (
              <span className="text-[10px] font-mono text-blue-700 bg-blue-50 px-2 py-0.5 rounded-full border border-blue-100">
                {p.meta.scale}×
              </span>
            )}
          </div>

          <div
            className="aspect-[4/3] relative flex items-center justify-center bg-slate-100 cursor-zoom-in"
            onClick={() => onZoom(p.key)}
          >
            <img
              src={p.src}
              alt={p.label}
              className="w-full h-full object-contain bg-slate-900/5"
              style={{
                imageRendering: p.pixelated ? 'pixelated' : 'auto',
              }}
            />
            <div className="absolute inset-0 bg-black/5 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
              <span className="bg-black/60 text-white text-xs px-2.5 py-1 rounded-full backdrop-blur-sm">
                Click to inspect
              </span>
            </div>
          </div>

          {p.meta && (
            <div className="p-3 bg-white text-[11px] text-slate-500 border-t border-slate-100 flex items-center justify-between">
              <span>{p.meta.output_size?.join(' × ')} px</span>
              <span>{p.meta.latency_ms?.toFixed(1)} ms</span>
            </div>
          )}

          {p.key === 'sr' && onDownload && (
            <div className="p-3 pt-0 bg-white">
              <Button
                variant="secondary"
                size="sm"
                onClick={() => onDownload(p.src, `${srMeta?.model_id || 'super_resolved'}.png`)}
                className="w-full text-xs"
              >
                <Download size={13} /> Download Result
              </Button>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function BeforeAfterSlider({
  beforeImage,
  afterImage,
  beforeLabel,
  afterLabel,
  slider,
  onSlider,
  pixelated,
  showLoupe,
}) {
  const containerRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loupePos, setLoupePos] = useState({ x: 50, y: 50, active: false });

  const updatePosition = useCallback(
    (clientX) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      let pct = ((clientX - rect.left) / rect.width) * 100;
      pct = Math.max(0, Math.min(100, pct));
      onSlider(pct);
    },
    [onSlider]
  );

  const onPointerDown = (e) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    setIsDragging(true);
    updatePosition(e.clientX);
  };

  const onPointerMove = (e) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (rect) {
      const lx = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
      const ly = Math.max(0, Math.min(100, ((e.clientY - rect.top) / rect.height) * 100));
      setLoupePos({ x: lx, y: ly, active: true });
    }
    if (isDragging) {
      updatePosition(e.clientX);
    }
  };

  const onPointerUp = (e) => {
    setIsDragging(false);
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch (_) {
      // ignore
    }
  };

  const onPointerLeave = () => {
    setLoupePos((prev) => ({ ...prev, active: false }));
  };

  const onKeyDown = (e) => {
    const step = e.shiftKey ? 10 : 2;
    if (e.key === 'ArrowLeft') onSlider(Math.max(0, slider - step));
    else if (e.key === 'ArrowRight') onSlider(Math.min(100, slider + step));
  };

  return (
    <div className="space-y-3 select-none">
      <div
        ref={containerRef}
        className="relative w-full aspect-[4/3] rounded-2xl overflow-hidden bg-slate-900 border border-slate-200 shadow-inner cursor-ew-resize touch-none focus:outline-none focus:ring-2 focus:ring-blue-500"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerLeave}
        onKeyDown={onKeyDown}
        tabIndex={0}
      >
        {/* Layer 1: BEFORE (Bottom layer: 0% to 100%) */}
        <div className="absolute inset-0 flex items-center justify-center">
          <img
            src={beforeImage}
            alt="Before"
            className="w-full h-full object-contain pointer-events-none"
            style={{
              imageRendering: pixelated ? 'pixelated' : 'auto',
            }}
          />
        </div>

        {/* Layer 2: AFTER (Top layer: clipped from left by slider %) */}
        <div
          className="absolute inset-0 flex items-center justify-center pointer-events-none"
          style={{
            clipPath: `inset(0 0 0 ${slider}%)`,
          }}
        >
          <img
            src={afterImage}
            alt="After"
            className="w-full h-full object-contain pointer-events-none"
          />
        </div>

        {/* Vertical Divider Line */}
        <div
          className="absolute top-0 bottom-0 pointer-events-none z-20"
          style={{ left: `${slider}%` }}
        >
          <div className="absolute top-0 bottom-0 -left-px w-0.5 bg-white shadow-[0_0_10px_rgba(0,0,0,0.5)]" />

          {/* Draggable Grip Handle Button */}
          <div
            className={`absolute top-1/2 -left-4 -translate-y-1/2 w-8 h-8 rounded-full bg-white text-slate-800 shadow-xl border-2 border-blue-600 flex items-center justify-center transition-transform ${
              isDragging ? 'scale-110 shadow-blue-500/50' : 'hover:scale-105'
            }`}
          >
            <div className="flex items-center gap-0.5 text-[10px] font-bold text-blue-600">
              ◀▶
            </div>
          </div>
        </div>

        {/* Corner Indicator Badges */}
        <div className="absolute top-3 left-3 z-10 pointer-events-none">
          <span className="px-2.5 py-1 rounded-lg bg-black/60 text-white text-xs backdrop-blur-md font-medium shadow-sm">
            {beforeLabel}
          </span>
        </div>
        <div className="absolute top-3 right-3 z-10 pointer-events-none">
          <span className="px-2.5 py-1 rounded-lg bg-blue-600/90 text-white text-xs backdrop-blur-md font-medium shadow-sm">
            {afterLabel}
          </span>
        </div>

        {/* Magnifier Loupe Overlay */}
        {showLoupe && loupePos.active && (
          <div
            className="absolute z-30 pointer-events-none rounded-2xl overflow-hidden border-2 border-white shadow-2xl bg-slate-900"
            style={{
              width: '160px',
              height: '160px',
              left: `${Math.min(80, Math.max(20, loupePos.x))}%`,
              top: `${Math.min(80, Math.max(20, loupePos.y))}%`,
              transform: 'translate(-50%, -50%)',
            }}
          >
            {/* Loupe Background: Zoomed Before */}
            <div
              className="absolute inset-0"
              style={{
                backgroundImage: `url(${beforeImage})`,
                backgroundPosition: `${loupePos.x}% ${loupePos.y}%`,
                backgroundSize: '400%',
                imageRendering: pixelated ? 'pixelated' : 'auto',
              }}
            />
            {/* Loupe Foreground: Zoomed After clipped at 50% */}
            <div
              className="absolute inset-0 border-l border-white/80"
              style={{
                backgroundImage: `url(${afterImage})`,
                backgroundPosition: `${loupePos.x}% ${loupePos.y}%`,
                backgroundSize: '400%',
                clipPath: 'inset(0 0 0 50%)',
              }}
            />
            {/* Loupe Center Crosshair */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-1 h-1 rounded-full bg-red-500 shadow-xs" />
            </div>
            <div className="absolute bottom-1 inset-x-0 flex justify-between px-2 text-[9px] text-white/90 bg-black/60 font-mono">
              <span>Before (4×)</span>
              <span>After</span>
            </div>
          </div>
        )}
      </div>

      <div className="flex items-center justify-between text-xs text-slate-500 px-1">
        <span className="flex items-center gap-1.5 font-medium text-slate-700">
          ◀ {beforeLabel}
        </span>
        <span className="text-[11px] text-slate-400">
          Drag slider or use arrow keys ← →
        </span>
        <span className="flex items-center gap-1.5 font-medium text-blue-600">
          {afterLabel} ▶
        </span>
      </div>
    </div>
  );
}
