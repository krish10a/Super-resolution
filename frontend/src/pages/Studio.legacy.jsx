/**
 * Super-Resolution Studio — the primary feature.
 * Upload any custom image or pick a CCTV surveillance preset, choose a model,
 * and run real PyTorch super-resolution with dual ingest modes:
 * - Mode 1: Surveillance Simulation & Recovery (Recommended for any full-res photo)
 * - Mode 2: Direct Super-Resolution (For native low-res CCTV frames)
 */
import React, { useState, useCallback, useRef, useEffect } from 'react';
import { api } from '../api/client';
import {
  RefreshCw,
  Play,
  Download,
  Link,
  ImageMinus,
  AlertCircle,
  CheckCircle,
  Camera,
  Layers,
  Sparkles,
  SlidersHorizontal,
  Crop,
  ShieldCheck,
  Eye,
} from 'lucide-react';
import ImageUpload from '../components/ImageUpload';
import ModelSelector from '../components/ModelSelector';
import Comparison from './Comparison';
import {
  Card,
  CardHeader,
  CardTitle,
  Badge,
  Button,
  SectionTitle,
  SectionSubtitle,
  ImagePanel,
  Divider,
  Spinner,
} from '../components/ui';

const SAMPLE_PRESETS = [
  {
    id: 'cctv_signage',
    name: 'CCTV Signage / Text',
    desc: 'Low-res characters & fine edges',
    path: '/samples/cctv_signage.png',
  },
  {
    id: 'surveillance_face',
    name: 'Surveillance Face',
    desc: 'Facial features & skin contour',
    path: '/samples/surveillance_face.png',
  },
  {
    id: 'security_person',
    name: 'Security Person',
    desc: 'Defocus & shadow recovery',
    path: '/samples/security_person.png',
  },
  {
    id: 'surveillance_vehicle',
    name: 'Vehicle & Plate',
    desc: 'High-contrast edge resolution',
    path: '/samples/surveillance_vehicle.png',
  },
];

export default function Studio() {
  const [image, setImage] = useState(null); // {file, preview, width, height}
  const [modelId, setModelId] = useState('espcn_4x_mse');
  const [srImage, setSrImage] = useState(null); // base64 data URL
  const [srMeta, setSrMeta] = useState(null);
  const [bicubic, setBicubic] = useState(null); // base64 data URL
  const [lrDisplay, setLrDisplay] = useState(null); // base64 or preview data URL
  const [hrGroundTruth, setHrGroundTruth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showCompare, setShowCompare] = useState(false);
  const [lastScale, setLastScale] = useState(null);
  const [loadingPreset, setLoadingPreset] = useState(false);
  const [simulateMode, setSimulateMode] = useState(false); // auto-detected or toggled
  const compareRef = useRef(null);

  // Auto-detect whether image is full-res and needs simulation mode
  const handleFile = useCallback((file) => {
    const preview = URL.createObjectURL(file);
    const imgObj = new Image();
    imgObj.onload = () => {
      const isHigh = imgObj.naturalWidth > 320 || imgObj.naturalHeight > 320;
      setImage({ file, preview, width: imgObj.naturalWidth, height: imgObj.naturalHeight });
      setSimulateMode(isHigh); // default to simulation mode for large/sharp photos!
    };
    imgObj.src = preview;

    setSrImage(null);
    setSrMeta(null);
    setBicubic(null);
    setLrDisplay(null);
    setHrGroundTruth(null);
    setError(null);
    setShowCompare(false);
    setLastScale(null);
  }, []);

  const handleRemove = useCallback(() => {
    if (image?.preview) URL.revokeObjectURL(image.preview);
    setImage(null);
    setSrImage(null);
    setSrMeta(null);
    setBicubic(null);
    setLrDisplay(null);
    setHrGroundTruth(null);
    setError(null);
    setShowCompare(false);
    setLastScale(null);
  }, [image]);

  const handleLoadPreset = async (preset) => {
    setLoadingPreset(true);
    setError(null);
    try {
      const response = await fetch(preset.path);
      if (!response.ok) throw new Error('Failed to load sample image');
      const blob = await response.blob();
      const file = new File([blob], `${preset.id}.png`, { type: 'image/png' });
      handleFile(file);
      setModelId('espcn_4x_mse');
      setSimulateMode(false); // Presets are already native low-res
    } catch (err) {
      setError(`Could not load sample: ${err.message}`);
    } finally {
      setLoadingPreset(false);
    }
  };

  const handleEnhance = async () => {
    if (!image || !modelId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.superResolve(image.file, modelId, simulateMode);
      setSrImage(`data:image/png;base64,${res.image_base64}`);
      setSrMeta(res);
      setLastScale(res.scale);

      // Set the LR to display in comparison:
      // If simulate_lr was used, the backend returns the degraded LR frame!
      if (res.lr_base64) {
        setLrDisplay(`data:image/png;base64,${res.lr_base64}`);
      } else {
        setLrDisplay(image.preview);
      }

      if (res.hr_base64) {
        setHrGroundTruth(`data:image/png;base64,${res.hr_base64}`);
      } else {
        setHrGroundTruth(null);
      }

      // Fetch bicubic baseline
      try {
        const bc = await api.bicubic(image.file, res.scale);
        setBicubic(`data:image/png;base64,${bc.image_base64}`);
      } catch (_) {
        setBicubic(null);
      }

      setShowCompare(true);

      // Smooth scroll to comparison workspace
      setTimeout(() => {
        compareRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 250);
    } catch (err) {
      setError(err.message || 'Inference failed');
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

  const modelName = modelId || '—';
  const modelScale = lastScale || (srMeta?.scale || null);

  const inPixels = srMeta?.input_size ? srMeta.input_size[0] * srMeta.input_size[1] : 0;
  const outPixels = srMeta?.output_size ? srMeta.output_size[0] * srMeta.output_size[1] : 0;
  const pixelMultiplier = inPixels > 0 ? (outPixels / inPixels).toFixed(0) : null;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <SectionTitle>Super-Resolution Studio</SectionTitle>
        <SectionSubtitle>
          Deep learning super-resolution for surveillance feeds, license plates, and low-quality imagery.
        </SectionSubtitle>
      </div>

      {/* Preset Feeds Quick-Bar */}
      <Card className="bg-gradient-to-r from-blue-50/60 via-slate-50 to-indigo-50/40 border-blue-100/70 p-4">
        <div className="flex items-center gap-2 mb-2 text-xs font-semibold text-blue-900 uppercase tracking-wider">
          <Camera size={14} className="text-blue-600" />
          <span>Curated Surveillance Presets (Instant 1-Click Verification)</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          {SAMPLE_PRESETS.map((p) => (
            <button
              key={p.id}
              onClick={() => handleLoadPreset(p)}
              disabled={loading || loadingPreset}
              className="flex flex-col text-left p-2.5 rounded-xl border border-slate-200 bg-white hover:border-blue-400 hover:shadow-xs transition-all group active:scale-98"
            >
              <span className="text-xs font-semibold text-slate-800 group-hover:text-blue-600 transition-colors">
                {p.name}
              </span>
              <span className="text-[10px] text-slate-400 mt-0.5 line-clamp-1">{p.desc}</span>
            </button>
          ))}
        </div>
      </Card>

      {/* Main Grid: Upload & Controls | Enhanced Output */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-5 space-y-6">
          {/* Upload Box */}
          <Card>
            <CardHeader className="justify-between">
              <CardTitle>1. Upload Any Image</CardTitle>
              {image && (
                <Badge variant="available">
                  {image.width} × {image.height} px
                </Badge>
              )}
            </CardHeader>
            <ImageUpload
              image={image}
              onFile={handleFile}
              onRemove={handleRemove}
              disabled={loading}
            />

            {/* Custom Image Mode Selector */}
            {image && (
              <div className="mt-4 p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-2.5 text-xs">
                <span className="font-semibold text-slate-700 uppercase tracking-wider text-[10px] block">
                  Processing Ingest Mode:
                </span>

                <div className="space-y-2">
                  <label className="flex items-start gap-2.5 cursor-pointer select-none">
                    <input
                      type="radio"
                      name="ingestMode"
                      checked={simulateMode}
                      onChange={() => setSimulateMode(true)}
                      className="mt-0.5 text-blue-600 focus:ring-blue-500"
                    />
                    <div>
                      <span className="font-semibold text-slate-800 block">
                        Surveillance Simulation & Recovery
                      </span>
                      <p className="text-[11px] text-slate-500 mt-0.5">
                        Downsamples custom photos to realistic CCTV resolution with sensor noise, then recovers details via deep learning. (Recommended for custom photos so enhancement is unmistakable).
                      </p>
                    </div>
                  </label>

                  <label className="flex items-start gap-2.5 cursor-pointer select-none">
                    <input
                      type="radio"
                      name="ingestMode"
                      checked={!simulateMode}
                      onChange={() => setSimulateMode(false)}
                      className="mt-0.5 text-blue-600 focus:ring-blue-500"
                    />
                    <div>
                      <span className="font-semibold text-slate-800 block">
                        Direct Super-Resolution Upscale
                      </span>
                      <p className="text-[11px] text-slate-500 mt-0.5">
                        Directly feeds the uploaded image into the neural network without downsampling (ideal for images that are already low-res CCTV frames).
                      </p>
                    </div>
                  </label>
                </div>
              </div>
            )}

            {error && (
              <div className="mt-4 p-3 rounded-xl border border-red-100 bg-red-50 text-red-700 text-xs flex items-center gap-2">
                <AlertCircle size={16} className="shrink-0" /> {error}
              </div>
            )}
          </Card>

          {/* Model Selector */}
          <Card>
            <CardHeader className="justify-between">
              <CardTitle>2. Select Model</CardTitle>
              <Badge variant="brand">8 Verified Models</Badge>
            </CardHeader>
            <ModelSelector onSelect={setModelId} disabled={loading} />
          </Card>

          {/* Enhance Button */}
          <Button
            onClick={handleEnhance}
            disabled={loading || !image || !modelId}
            className="w-full py-4 flex items-center justify-center gap-2 text-base font-semibold shadow-lg shadow-blue-100/60"
          >
            {loading ? (
              <>
                <Spinner className="mr-2" />
                Super-Resolving Frame…
              </>
            ) : (
              <>
                <Sparkles size={18} /> Enhance Resolution Now
              </>
            )}
          </Button>
        </div>

        {/* Output Panel */}
        <div className="lg:col-span-7 space-y-6">
          {srImage ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <CardHeader>
                  <CheckCircle className="w-5 h-5 text-emerald-600" />
                  <CardTitle>Enhanced Super-Resolution Output</CardTitle>
                </CardHeader>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => handleDownload(srImage, `sr_${modelId}.png`)}
                >
                  <Download size={14} /> Download Result
                </Button>
              </div>

              {/* Pixel Gain Banner */}
              {pixelMultiplier && (
                <div className="p-3.5 rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-sm flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2 font-medium">
                    <Sparkles size={16} />
                    <span>
                      Processed {srMeta.input_size.join('×')} px → Reconstructed {srMeta.output_size.join('×')} px
                    </span>
                  </div>
                  <span className="font-bold bg-white/20 px-2.5 py-1 rounded-lg backdrop-blur-xs font-mono">
                    +{pixelMultiplier}× Pixels ({srMeta.scale}× Scale)
                  </span>
                </div>
              )}

              <ImagePanel
                src={srImage}
                alt="SR output"
                meta={srMeta}
                onDownload={() => handleDownload(srImage, `sr_${modelId}.png`)}
              />

              {/* Specs grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                {[
                  { label: 'Model', value: srMeta?.model_id || modelName },
                  { label: 'Scale Factor', value: modelScale ? `${modelScale}×` : '—' },
                  { label: 'Inference Latency', value: srMeta?.latency_ms ? `${srMeta.latency_ms.toFixed(1)} ms` : '—' },
                  { label: 'Mode', value: simulateMode ? 'CCTV Simulation' : 'Direct Upscale' },
                  { label: 'Input Dim', value: srMeta?.input_size?.join(' × ') || '—' },
                  { label: 'Output Dim', value: srMeta?.output_size?.join(' × ') || '—' },
                  { label: 'Execution Device', value: srMeta?.device || '—' },
                  { label: 'Pipeline', value: srMeta?.pipeline || '—' },
                ].map(({ label, value }) => (
                  <div key={label} className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
                    <span className="text-[10px] text-slate-400 block uppercase tracking-wider mb-0.5">
                      {label}
                    </span>
                    <span className="text-xs font-semibold text-slate-800 font-mono truncate block">
                      {value}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full min-h-[420px] rounded-2xl border-2 border-dashed border-slate-200 bg-slate-50/50 p-8 text-center">
              <ImageMinus size={48} className="text-slate-300 mb-3" />
              <p className="text-sm font-semibold text-slate-700">No enhanced image yet</p>
              <p className="text-xs text-slate-400 mt-1 max-w-sm">
                Upload your custom photo or click one of the presets on the left, then click <strong>Enhance Resolution Now</strong>.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Comparison Workspace */}
      <div ref={compareRef}>
        {showCompare && (srImage || bicubic || image) && (
          <>
            <Divider />
            <Comparison
              lrImage={lrDisplay || (image ? image.preview : null)}
              bicubic={bicubic}
              srImage={srImage}
              srMeta={srMeta}
              onDownload={handleDownload}
            />
          </>
        )}
      </div>
    </div>
  );
}
