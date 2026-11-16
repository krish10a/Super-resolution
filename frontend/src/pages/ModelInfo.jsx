/**
 * Model Information page — architecture explanations, loss functions, and
 * honest limitations.
 */
import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import { Layers, BookOpen, AlertTriangle, ShieldAlert, Code2, Zap, AlertCircle } from 'lucide-react';
import {
  Card,
  CardHeader,
  CardTitle,
  Badge,
  SectionTitle,
  SectionSubtitle,
} from '../components/ui';

export default function ModelInfo() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let mounted = true;
    api.getModels()
      .then((m) => { if (mounted) { setModels(m); setLoading(false); } })
      .catch((err) => { if (mounted) { setError(err.message || 'Failed to load models.'); setLoading(false); } });
    return () => { mounted = false; };
  }, []);

  if (loading) return (
    <Card className="p-12 text-center">
      <div className="flex items-center justify-center gap-2 text-sm text-slate-500">
        <Zap className="w-5 h-5 animate-pulse" />
        Loading model info…
      </div>
    </Card>
  );

  if (error) return (
    <Card className="p-12 text-center border-amber-200 bg-amber-50">
      <div className="flex items-center justify-center gap-2 text-sm text-amber-800 mb-4">
        <AlertCircle size={20} />
        <span>{error}</span>
      </div>
      <button
        onClick={() => window.location.reload()}
        className="text-xs underline underline-offset-4 hover:text-amber-700"
      >
        Refresh page
      </button>
    </Card>
  );

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div>
        <SectionTitle>Model Information</SectionTitle>
        <SectionSubtitle>
          Architecture details, loss functions, and honest limitations.
        </SectionSubtitle>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ArchCard
          name="SRCNN"
          icon={<Layers className="w-6 h-6" />}
          desc="Super-Resolution Convolutional Neural Network. A shallow CNN that maps a bicubic-upsampled LR image to a high-resolution output."
          prep="LR upload → bicubic pre-upsample → SRCNN → SR output."
          layers={[
            'conv1: 3→64 filters, 9×9 kernel, ReLU',
            'conv2: 64→32 filters, 1×1 kernel, ReLU',
            'conv3: 32→3 filters, 5×5 kernel, Tanh output',
          ]}
          note="The bicubic pre-upsample is required — SRCNN convolutions are same-size, so the network never upsamples by itself."
        />
        <ArchCard
          name="ESPCN"
          icon={<Code2 className="w-6 h-6" />}
          desc="Efficient Sub-Pixel Convolutional Neural Network. Learns LR features then upscales via sub-pixel convolution (PixelShuffle)."
          prep="LR upload → ESPCN → PixelShuffle → SR output. No bicubic pre-upsample."
          layers={[
            'conv1: 3→64 filters, 5×5 kernel, ReLU',
            'conv2: 64→64 filters, 3×3 kernel, ReLU',
            'conv3: 64→3·scale² filters, 3×3 kernel',
            'PixelShuffle(scale): rearranges depth into spatial resolution',
          ]}
          note="PixelShuffle performs the upsampling internally — feeding it an already-upsampled image would double-upscale."
        />
      </div>

      <Card>
        <CardHeader>
          <BookOpen className="w-5 h-5 text-blue-600" />
          <CardTitle>Loss functions</CardTitle>
        </CardHeader>
        <div className="space-y-3">
          <LossRow
            name="MSE"
            variant="brand"
            full="Mean Squared Error"
            desc="Pixel-wise L2 loss between SR and HR. Encourages accurate reconstruction but tends to produce smoother results."
          />
          <LossRow
            name="Perceptual"
            variant="default"
            full="VGG16 feature loss"
            desc="MSE + λ·MSE(VGG16 features). Encourages perceptually plausible textures; λ=0.05 in training. May invent plausible-but-inaccurate detail."
          />
          <LossRow
            name="Hybrid"
            variant="default"
            full="MSE + Perceptual"
            desc="Used in perceptual experiments (E2/E4/E6/E8). Checkpoints are raw state dicts — loss attribution is config-documented."
          />
        </div>
      </Card>

      <Card className="border-amber-100 bg-amber-50/50">
        <CardHeader className="flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-amber-600" />
          <CardTitle>Limitations — please read</CardTitle>
        </CardHeader>
        <ul className="space-y-2 text-sm text-amber-800">
          <li>Super-resolution produces a <strong>reconstruction</strong>, not guaranteed recovery of original detail.</li>
          <li>Perceptual models may produce visually plausible but factually inaccurate textures.</li>
          <li>Outputs must <strong>not</strong> be treated as forensic evidence.</li>
          <li>No claims are made about license-plate, face, or identification accuracy.</li>
          <li>PSNR/SSIM/LPIPS are shown only when a matching HR ground truth is supplied — never fabricated.</li>
        </ul>
      </Card>

      <Card>
        <CardHeader>
          <ShieldAlert className="w-5 h-5 text-slate-600" />
          <CardTitle>Registered checkpoints</CardTitle>
        </CardHeader>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="border-b border-slate-200 text-left text-xs text-slate-500 uppercase">
                <th className="pb-3 pr-4">Model</th>
                <th className="pb-3">Arch</th>
                <th className="pb-3">Scale</th>
                <th className="pb-3">Loss</th>
                <th className="pb-3">Checkpoint</th>
                <th className="pb-3">Available</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {models.map((m) => (
                <tr key={m.model_id} className="hover:bg-slate-50/50 transition-colors">
                  <td className="py-3 pr-4 font-medium text-slate-800">{m.model_id}</td>
                  <td className="py-3 capitalize text-slate-600">{m.architecture}</td>
                  <td className="py-3 font-mono text-slate-600">{m.scale}×</td>
                  <td className="py-3 capitalize text-slate-600">{m.loss.replace('_', ' ')}</td>
                  <td className="py-3 text-slate-500 font-mono text-xs">{m.checkpoint}</td>
                  <td className="py-3">
                    <Badge variant={m.available ? 'available' : 'unavailable'}>
                      {m.available ? 'Available' : 'Unavailable'}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function ArchCard({ name, icon, desc, prep, layers, note }) {
  return (
    <Card>
      <div className="flex items-center gap-2 mb-3">
        <div className="p-3 rounded-xl bg-blue-50 text-blue-600">{icon}</div>
        <span className="text-lg font-bold text-slate-900">{name}</span>
      </div>
      <p className="text-sm text-slate-700 mb-4">{desc}</p>
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Input preparation</h4>
      <p className="text-sm text-slate-700 mb-4">{prep}</p>
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Layers</h4>
      <ul className="list-disc list-inside text-sm text-slate-700 space-y-1 mb-4">
        {layers.map((l, i) => <li key={i}>{l}</li>)}
      </ul>
      <p className="text-xs text-slate-500 italic">{note}</p>
    </Card>
  );
}

function LossRow({ name, variant, full, desc }) {
  return (
    <div className="p-3 rounded-xl border border-slate-200 bg-slate-50">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-xs font-semibold text-blue-600 bg-blue-50 px-2 py-0.5 rounded">{name}</span>
        <span className="ml-auto text-xs text-slate-500">{full}</span>
      </div>
      <p className="text-sm text-slate-700">{desc}</p>
    </div>
  );
}
