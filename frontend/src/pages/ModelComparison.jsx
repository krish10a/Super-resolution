/**
 * Model Comparison page — run compatible (same-scale) models on the same input
 * and show outputs with side-by-side inspection and Model-vs-Model slider.
 */
import React, { useState, useEffect, useRef } from 'react';
import { api } from '../api/client';
import {
  Image as ImageIcon,
  RefreshCw,
  Play,
  Download,
  ShieldOff,
  AlertCircle,
  Sliders,
  Camera,
  Layers,
  Sparkles,
} from 'lucide-react';
import ImageUpload from '../components/ImageUpload';
import {
  Card,
  CardHeader,
  CardTitle,
  Badge,
  Button,
  SectionTitle,
  SectionSubtitle,
  ImagePanel,
  EmptyState,
  Spinner,
} from '../components/ui';

const SAMPLE_PRESETS = [
  { id: 'cctv_signage', name: 'CCTV Signage', path: '/samples/cctv_signage.png' },
  { id: 'surveillance_face', name: 'Surveillance Face', path: '/samples/surveillance_face.png' },
  { id: 'security_person', name: 'Security Person', path: '/samples/security_person.png' },
  { id: 'surveillance_vehicle', name: 'Vehicle & Plate', path: '/samples/surveillance_vehicle.png' },
];

export default function ModelComparison() {
  const [models, setModels] = useState([]);
  const [image, setImage] = useState(null); // {file, preview}
  const [selected, setSelected] = useState([]); // model ids
  const [results, setResults] = useState([]); // {model_id, image_base64, ...}
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [scale, setScale] = useState(null);

  // Model-vs-Model A/B Slider state
  const [sliderModelA, setSliderModelA] = useState(0);
  const [sliderModelB, setSliderModelB] = useState(1);
  const [sliderPct, setSliderPct] = useState(50);
  const sliderContainerRef = useRef(null);

  useEffect(() => {
    let mounted = true;
    api
      .getModels()
      .then((m) => {
        if (mounted) {
          setModels(m);
          const byScale = {};
          m.forEach((x) => {
            (byScale[x.scale] || (byScale[x.scale] = [])).push(x);
          });
          const scales = Object.keys(byScale);
          if (scales.length) {
            const s = parseInt(scales[0], 10);
            setScale(s);
            const avail = byScale[s].filter((x) => x.available);
            const ids =
              avail.length >= 2 ? avail.slice(0, 2).map((x) => x.model_id) : avail.map((x) => x.model_id);
            setSelected(ids);
          }
        }
      })
      .catch((err) => {
        if (mounted) setError(err.message || 'Failed to load models.');
      });
    return () => {
      mounted = false;
    };
  }, []);

  const handleFile = (file) => {
    setImage({ file, preview: URL.createObjectURL(file) });
    setResults([]);
    setError(null);
  };

  const handleRemove = () => {
    if (image) URL.revokeObjectURL(image.preview);
    setImage(null);
    setResults([]);
    setError(null);
  };

  const handleLoadPreset = async (preset) => {
    setError(null);
    try {
      const response = await fetch(preset.path);
      if (!response.ok) throw new Error('Failed to load sample image');
      const blob = await response.blob();
      const file = new File([blob], `${preset.id}.png`, { type: 'image/png' });
      handleFile(file);
    } catch (err) {
      setError(`Could not load sample: ${err.message}`);
    }
  };

  const toggle = (id) =>
    setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));

  const runCompare = async () => {
    if (!image || selected.length < 2) {
      setError('Select at least two models.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await api.compare(image.file, selected.join(','));
      setResults(res.models);
      setSliderModelA(0);
      setSliderModelB(Math.min(1, res.models.length - 1));
    } catch (err) {
      setError(err.message || 'Comparison failed');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = (b64, name) => {
    if (!b64) return;
    const a = document.createElement('a');
    a.href = b64.startsWith('data:') ? b64 : `data:image/png;base64,${b64}`;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // Slider dragging
  const handleSliderMove = (clientX) => {
    const rect = sliderContainerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const pct = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
    setSliderPct(pct);
  };

  if (error && !models.length) {
    return (
      <Card className="p-12 text-center border-amber-200 bg-amber-50">
        <div className="flex items-center justify-center gap-2 text-sm text-amber-800 mb-4">
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
        <Button variant="outline" size="sm" onClick={() => window.location.reload()}>
          Refresh page
        </Button>
      </Card>
    );
  }

  const availableByScale = {};
  models
    .filter((m) => m.available)
    .forEach((m) => {
      (availableByScale[m.scale] || (availableByScale[m.scale] = [])).push(m);
    });
  const scaleOptions = Object.keys(availableByScale).map(Number);

  const modelAData = results[sliderModelA];
  const modelBData = results[sliderModelB];

  return (
    <div className="space-y-8">
      <div>
        <SectionTitle>Model Comparison</SectionTitle>
        <SectionSubtitle>
          Run compatible models on the exact same frame to evaluate MSE (pixel loss) vs Perceptual (feature loss).
        </SectionSubtitle>
      </div>

      {/* Preset Feeds Quick-Bar */}
      <Card className="bg-slate-50 border-slate-200 p-3.5">
        <div className="flex items-center gap-2 mb-2 text-xs font-semibold text-slate-700 uppercase tracking-wider">
          <Camera size={13} className="text-blue-600" />
          <span>Quick-Load Sample Surveillance Feeds</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {SAMPLE_PRESETS.map((p) => (
            <button
              key={p.id}
              onClick={() => handleLoadPreset(p)}
              disabled={loading}
              className="px-3 py-1.5 rounded-lg border border-slate-200 bg-white hover:border-blue-300 hover:text-blue-600 text-xs font-medium text-slate-700 transition-colors shadow-2xs"
            >
              {p.name}
            </button>
          ))}
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-4 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Input Image</CardTitle>
            </CardHeader>
            <ImageUpload image={image} onFile={handleFile} onRemove={handleRemove} disabled={loading} />
          </Card>

          <div className="space-y-4">
            <Label>Upscale Scale Factor</Label>
            <div className="flex gap-2 flex-wrap">
              {scaleOptions.map((s) => (
                <button
                  key={s}
                  onClick={() => {
                    const avail = availableByScale[s]?.map((x) => x.model_id) || [];
                    setScale(s);
                    setSelected(avail.length >= 2 ? avail.slice(0, 2) : avail);
                  }}
                  className={`px-4 py-2 rounded-xl text-xs font-medium transition-all duration-200 ${
                    scale === s
                      ? 'bg-blue-600 text-white shadow-md shadow-blue-100/50'
                      : 'bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 hover:border-slate-300'
                  }`}
                >
                  {s}× Upscale
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <Label>Select Models to Compare</Label>
            <div className="max-h-56 overflow-y-auto border border-slate-200 rounded-xl p-3 space-y-2 bg-slate-50/50">
              {availableByScale[scale]?.map((m) => (
                <label
                  key={m.model_id}
                  className="flex items-center gap-2.5 p-2 rounded-lg bg-white border border-slate-200/80 cursor-pointer hover:border-blue-300 transition-colors select-none"
                >
                  <input
                    type="checkbox"
                    checked={selected.includes(m.model_id)}
                    onChange={() => toggle(m.model_id)}
                    disabled={loading}
                    className="rounded text-blue-600 focus:ring-blue-500 w-4 h-4"
                  />
                  <div className="flex-1 min-w-0">
                    <span className="text-xs font-semibold text-slate-800 block truncate">
                      {m.display_name}
                    </span>
                    <span className="text-[10px] text-slate-400 font-mono block">
                      {m.model_id}
                    </span>
                  </div>
                </label>
              ))}
              {availableByScale[scale]?.length === 0 && (
                <p className="text-xs text-slate-400">No available models at this scale.</p>
              )}
            </div>
          </div>

          <Button
            onClick={runCompare}
            disabled={loading || !image || selected.length < 2}
            className="w-full py-4 flex items-center justify-center gap-2 shadow-lg shadow-blue-100/50 text-sm font-semibold"
          >
            {loading ? (
              <>
                <Spinner className="mr-2" />
                Comparing Selected Models…
              </>
            ) : (
              <>
                <Play size={18} /> Run Multi-Model Comparison
              </>
            )}
          </Button>

          {error && (
            <div className="p-3 rounded-xl border border-red-100 bg-red-50 text-red-700 text-xs flex items-center gap-2">
              <ShieldOff size={16} /> {error}
            </div>
          )}
        </div>

        <div className="lg:col-span-8 space-y-6">
          {results.length ? (
            <div className="space-y-6">
              {/* Model A vs Model B Slider */}
              {results.length >= 2 && modelAData && modelBData && (
                <Card className="overflow-hidden border-blue-100 shadow-md">
                  <CardHeader className="flex-wrap justify-between gap-2 border-b border-slate-100 pb-3">
                    <div className="flex items-center gap-2">
                      <Sliders size={16} className="text-blue-600" />
                      <CardTitle className="text-base">Direct Model A / B Slider</CardTitle>
                    </div>

                    <div className="flex items-center gap-2 text-xs">
                      <select
                        value={sliderModelA}
                        onChange={(e) => setSliderModelA(Number(e.target.value))}
                        className="rounded-lg border border-slate-200 px-2.5 py-1 text-xs bg-slate-50 font-medium"
                      >
                        {results.map((r, i) => (
                          <option key={r.model_id} value={i}>
                            Model A: {r.model_id}
                          </option>
                        ))}
                      </select>

                      <span className="text-slate-400 font-bold">vs</span>

                      <select
                        value={sliderModelB}
                        onChange={(e) => setSliderModelB(Number(e.target.value))}
                        className="rounded-lg border border-slate-200 px-2.5 py-1 text-xs bg-slate-50 font-medium"
                      >
                        {results.map((r, i) => (
                          <option key={r.model_id} value={i}>
                            Model B: {r.model_id}
                          </option>
                        ))}
                      </select>
                    </div>
                  </CardHeader>

                  <div
                    ref={sliderContainerRef}
                    className="relative w-full aspect-[4/3] bg-slate-900 overflow-hidden cursor-ew-resize select-none touch-none"
                    onPointerDown={(e) => {
                      e.currentTarget.setPointerCapture(e.pointerId);
                      handleSliderMove(e.clientX);
                    }}
                    onPointerMove={(e) => {
                      if (e.buttons > 0) handleSliderMove(e.clientX);
                    }}
                  >
                    {/* Model A (Left Layer) */}
                    <div className="absolute inset-0 flex items-center justify-center">
                      <img
                        src={`data:image/png;base64,${modelAData.image_base64}`}
                        alt="Model A"
                        className="w-full h-full object-contain pointer-events-none"
                      />
                    </div>

                    {/* Model B (Right Layer, clipped from left) */}
                    <div
                      className="absolute inset-0 flex items-center justify-center pointer-events-none"
                      style={{ clipPath: `inset(0 0 0 ${sliderPct}%)` }}
                    >
                      <img
                        src={`data:image/png;base64,${modelBData.image_base64}`}
                        alt="Model B"
                        className="w-full h-full object-contain pointer-events-none"
                      />
                    </div>

                    {/* Divider Line */}
                    <div
                      className="absolute top-0 bottom-0 pointer-events-none z-20"
                      style={{ left: `${sliderPct}%` }}
                    >
                      <div className="absolute top-0 bottom-0 -left-px w-0.5 bg-white shadow-[0_0_8px_rgba(0,0,0,0.6)]" />
                      <div className="absolute top-1/2 -left-3.5 -translate-y-1/2 w-7 h-7 rounded-full bg-white shadow-xl border-2 border-blue-600 flex items-center justify-center text-[9px] font-bold text-blue-600">
                        ◀▶
                      </div>
                    </div>

                    <div className="absolute top-3 left-3 z-10 pointer-events-none">
                      <span className="px-2.5 py-1 rounded-lg bg-black/60 text-white text-xs backdrop-blur-md font-medium">
                        Model A: {modelAData.model_id}
                      </span>
                    </div>
                    <div className="absolute top-3 right-3 z-10 pointer-events-none">
                      <span className="px-2.5 py-1 rounded-lg bg-blue-600/90 text-white text-xs backdrop-blur-md font-medium">
                        Model B: {modelBData.model_id}
                      </span>
                    </div>
                  </div>

                  <div className="p-3 bg-slate-50 border-t border-slate-100 flex justify-between text-xs text-slate-500">
                    <span>◀ {modelAData.model_id}</span>
                    <span className="text-[11px] text-slate-400">Drag to inspect edge & texture differences</span>
                    <span>{modelBData.model_id} ▶</span>
                  </div>
                </Card>
              )}

              {/* Grid of Results */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {results.map((r) => (
                  <div
                    key={r.model_id}
                    className="rounded-2xl border border-slate-200 bg-white p-4 hover:shadow-md transition-all duration-200 flex flex-col justify-between"
                  >
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-bold text-slate-800 truncate">{r.model_id}</span>
                        <Badge variant="brand" className="text-[10px]">
                          {r.architecture.toUpperCase()} • {r.scale}×
                        </Badge>
                      </div>

                      <ImagePanel
                        src={`data:image/png;base64,${r.image_base64}`}
                        alt={r.model_id}
                        meta={{ model_id: r.model_id, scale: r.scale }}
                        onDownload={() => handleDownload(r.image_base64, `${r.model_id}.png`)}
                      />

                      <div className="grid grid-cols-2 gap-2 text-xs text-slate-500 my-3">
                        <div className="p-2 rounded-lg bg-slate-50 border border-slate-100">
                          <span className="text-[10px] block text-slate-400 uppercase">Resolution</span>
                          <span className="font-mono text-slate-800">{r.output_size?.join('×')} px</span>
                        </div>
                        <div className="p-2 rounded-lg bg-slate-50 border border-slate-100">
                          <span className="text-[10px] block text-slate-400 uppercase">Latency</span>
                          <span className="font-mono text-slate-800">{r.latency_ms?.toFixed(1)} ms</span>
                        </div>
                      </div>
                    </div>

                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => handleDownload(r.image_base64, `${r.model_id}.png`)}
                      className="w-full text-xs"
                    >
                      <Download size={13} /> Download Output
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <EmptyState
              icon={<ImageIcon size={32} className="text-slate-400" />}
              title="No comparison runs yet"
              desc="Upload an image or pick a preset, select ≥2 models, and click Run Multi-Model Comparison."
            />
          )}
        </div>
      </div>
    </div>
  );
}

function Label({ children }) {
  return (
    <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">
      {children}
    </label>
  );
}
