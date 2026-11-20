import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ExperimentId, ModelName } from '../lib/types';
import { useResults } from '../hooks/useResults';
import { useTrainingLog } from '../hooks/useTrainingLog';
import { ArrowRight, Layers, Cpu, Award } from 'lucide-react';

interface ExperimentMeta {
  id: ExperimentId;
  model: ModelName;
  scale: 2 | 4;
  loss: string;
  bestValPsnr: number;
  checkpoint: string;
  epochsRun: number;
  color: string;
  sparklineD: string;
  category: 'production' | 'legacy';
  paramsCount: string;
  tag?: string;
}

export const Models: React.FC = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<'production' | 'legacy'>('production');
  const [selectedExpId, setSelectedExpId] = useState<ExperimentId>('E14');
  const [drawerOpen, setDrawerOpen] = useState(true);
  const { results } = useResults();

  // Training log for the currently selected experiment
  const { epochs } = useTrainingLog(selectedExpId);

  // Specifications for E1–E14
  const experimentSpecs: ExperimentMeta[] = [
    // Phase 3 DIV2K Production Models (E9-E14)
    {
      id: 'E14',
      model: 'EDSR',
      scale: 4,
      loss: 'L1',
      bestValPsnr: 31.04,
      checkpoint: 'phase3/E14_edsr_4x_best.pt',
      epochsRun: 50,
      color: '#F59E0B',
      sparklineD: 'M0,30 L15,16 L35,10 L60,6 L85,4 L115,3 L140,2',
      category: 'production',
      paramsCount: '1.37M',
      tag: '4× Peak Champion (+3.10 dB)',
    },
    {
      id: 'E11',
      model: 'EDSR',
      scale: 2,
      loss: 'L1',
      bestValPsnr: 36.87,
      checkpoint: 'phase3/E11_edsr_2x_best.pt',
      epochsRun: 50,
      color: '#F59E0B',
      sparklineD: 'M0,30 L15,14 L35,7 L60,4 L85,2 L115,1 L140,1',
      category: 'production',
      paramsCount: '1.37M',
      tag: '2× Peak Champion (+3.63 dB)',
    },
    {
      id: 'E12',
      model: 'ESPCN',
      scale: 4,
      loss: 'Charbonnier',
      bestValPsnr: 30.12,
      checkpoint: 'phase3/E12_espcn_4x_best.pt',
      epochsRun: 50,
      color: '#38BDF8',
      sparklineD: 'M0,28 L15,20 L35,14 L60,10 L85,8 L115,7 L140,6',
      category: 'production',
      paramsCount: '26.8k',
      tag: 'Ultra Fast (>175 FPS)',
    },
    {
      id: 'E9',
      model: 'ESPCN',
      scale: 2,
      loss: 'Charbonnier',
      bestValPsnr: 36.32,
      checkpoint: 'phase3/E9_espcn_2x_best.pt',
      epochsRun: 50,
      color: '#38BDF8',
      sparklineD: 'M0,28 L15,18 L35,10 L60,6 L85,4 L115,3 L140,2',
      category: 'production',
      paramsCount: '26.8k',
      tag: 'Real-Time Sub-Pixel',
    },
    {
      id: 'E13',
      model: 'FSRCNN',
      scale: 4,
      loss: 'Charbonnier',
      bestValPsnr: 30.29,
      checkpoint: 'phase3/E13_fsrcnn_4x_best.pt',
      epochsRun: 50,
      color: '#34D399',
      sparklineD: 'M0,28 L15,19 L35,13 L60,9 L85,7 L115,6 L140,5',
      category: 'production',
      paramsCount: '24.7k',
      tag: 'Compact Deconv',
    },
    {
      id: 'E10',
      model: 'FSRCNN',
      scale: 2,
      loss: 'Charbonnier',
      bestValPsnr: 36.45,
      checkpoint: 'phase3/E10_fsrcnn_2x_best.pt',
      epochsRun: 50,
      color: '#34D399',
      sparklineD: 'M0,28 L15,16 L35,9 L60,5 L85,3 L115,2 L140,2',
      category: 'production',
      paramsCount: '24.7k',
      tag: 'Fast Compact Deconv',
    },

    // Legacy T91 Models (E1-E8)
    {
      id: 'E1',
      model: 'SRCNN',
      scale: 2,
      loss: 'MSE',
      bestValPsnr: 26.65,
      checkpoint: 'srcnn_2x_mse.pt',
      epochsRun: 15,
      color: '#A78BFA',
      sparklineD: 'M0,28 L15,22 L35,16 L60,12 L85,9 L115,7 L140,7',
      category: 'legacy',
      paramsCount: '57.3k',
    },
    {
      id: 'E2',
      model: 'SRCNN',
      scale: 2,
      loss: 'MSE + Perceptual',
      bestValPsnr: 25.63,
      checkpoint: 'srcnn_2x_mse_perceptual.pt',
      epochsRun: 15,
      color: '#C084FC',
      sparklineD: 'M0,28 L15,23 L35,18 L60,15 L85,13 L115,11 L140,11',
      category: 'legacy',
      paramsCount: '57.3k',
    },
    {
      id: 'E3',
      model: 'SRCNN',
      scale: 4,
      loss: 'MSE',
      bestValPsnr: 25.64,
      checkpoint: 'srcnn_4x_mse.pt',
      epochsRun: 15,
      color: '#A78BFA',
      sparklineD: 'M0,30 L15,25 L35,20 L60,16 L85,13 L115,11 L140,10',
      category: 'legacy',
      paramsCount: '57.3k',
    },
    {
      id: 'E4',
      model: 'SRCNN',
      scale: 4,
      loss: 'MSE + Perceptual',
      bestValPsnr: 23.97,
      checkpoint: 'srcnn_4x_mse_perceptual.pt',
      epochsRun: 15,
      color: '#C084FC',
      sparklineD: 'M0,30 L15,26 L35,22 L60,19 L85,17 L115,15 L140,15',
      category: 'legacy',
      paramsCount: '57.3k',
    },
    {
      id: 'E5',
      model: 'ESPCN',
      scale: 2,
      loss: 'MSE',
      bestValPsnr: 27.68,
      checkpoint: 'espcn_2x_mse.pt',
      epochsRun: 15,
      color: '#818CF8',
      sparklineD: 'M0,28 L15,19 L35,12 L60,8 L85,5 L115,3 L140,3',
      category: 'legacy',
      paramsCount: '26.8k',
    },
    {
      id: 'E6',
      model: 'ESPCN',
      scale: 2,
      loss: 'MSE + Perceptual',
      bestValPsnr: 26.42,
      checkpoint: 'espcn_2x_mse_perceptual_verified.pt',
      epochsRun: 15,
      color: '#6366F1',
      sparklineD: 'M0,28 L15,20 L35,14 L60,10 L85,7 L115,5 L140,5',
      category: 'legacy',
      paramsCount: '26.8k',
    },
    {
      id: 'E7',
      model: 'ESPCN',
      scale: 4,
      loss: 'MSE',
      bestValPsnr: 24.98,
      checkpoint: 'espcn_4x_mse_verified.pt',
      epochsRun: 15,
      color: '#818CF8',
      sparklineD: 'M0,30 L15,26 L35,22 L60,19 L85,17 L115,16 L140,16',
      category: 'legacy',
      paramsCount: '26.8k',
    },
    {
      id: 'E8',
      model: 'ESPCN',
      scale: 4,
      loss: 'MSE + Perceptual',
      bestValPsnr: 22.33,
      checkpoint: 'espcn_4x_mse_perceptual_verified.pt',
      epochsRun: 15,
      color: '#6366F1',
      sparklineD: 'M0,30 L15,27 L35,24 L60,21 L85,19 L115,17 L140,17',
      category: 'legacy',
      paramsCount: '26.8k',
    },
  ];

  const visibleSpecs = experimentSpecs.filter((e) => e.category === activeTab);
  const activeExp = experimentSpecs.find((e) => e.id === selectedExpId) || experimentSpecs[0];

  const peakPsnr = epochs.length > 0
    ? Math.max(...epochs.map((e) => e.val_psnr)).toFixed(2)
    : (results?.[activeExp.id]?.results?.Set5?.avg_psnr?.toFixed(2) || activeExp.bestValPsnr.toFixed(2));

  const handleLoadIntoStudio = () => {
    navigate(`/studio?model=${activeExp.model}&scale=${activeExp.scale}&loss=${activeExp.loss}`);
  };

  return (
    <div className="w-full flex h-[calc(100vh-84px)] overflow-hidden relative">
      {/* MAIN VIEWPORT: Model Catalog Grid */}
      <main className="w-full pt-4 pb-7 flex-1 flex flex-col justify-start overflow-y-auto">
        <div className="max-w-[1440px] w-full mx-auto px-8 py-6 flex flex-col">
          {/* HEADER & CATEGORY SWITCHER */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="w-2 h-2 rounded-full bg-primary-container" />
                <span className="font-label-caps text-xs text-outline uppercase tracking-wider">
                  MODEL CATALOG & ARCHITECTURE EXPLORER
                </span>
              </div>
              <h1 className="font-headline font-semibold text-2xl text-on-surface tracking-tight">
                Model Architectures & Checkpoints
              </h1>
              <p className="font-mono text-xs text-outline mt-1">
                {activeTab === 'production'
                  ? '6 Production SISR Checkpoints · DIV2K 800-image dataset · 50 epochs'
                  : '8 Legacy SISR Checkpoints · T91 dataset · 15 epochs'}
              </p>
            </div>

            {/* Segmented Switcher */}
            <div className="flex items-center gap-1 bg-surface-container-low p-1 rounded-xl border border-outline-variant">
              <button
                onClick={() => {
                  setActiveTab('production');
                  setSelectedExpId('E14');
                }}
                className={`px-3.5 py-1.5 text-xs font-body-sm-medium rounded-lg transition-all ${
                  activeTab === 'production'
                    ? 'bg-surface-container text-primary-container font-semibold shadow-sm border border-primary-container/40'
                    : 'text-on-surface-variant hover:text-on-surface'
                }`}
              >
                Production (DIV2K · 50 Epochs)
              </button>
              <button
                onClick={() => {
                  setActiveTab('legacy');
                  setSelectedExpId('E7');
                }}
                className={`px-3.5 py-1.5 text-xs font-body-sm-medium rounded-lg transition-all ${
                  activeTab === 'legacy'
                    ? 'bg-surface-container text-primary-container font-semibold shadow-sm border border-primary-container/40'
                    : 'text-on-surface-variant hover:text-on-surface'
                }`}
              >
                Legacy Models (E1–E8)
              </button>
            </div>
          </div>

          {/* MODEL CARDS GRID */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {visibleSpecs.map((exp) => {
              const isSelected = selectedExpId === exp.id;
              const liveResultPsnr = results?.[exp.id]?.results?.Set5?.avg_psnr;
              const displayPsnr = liveResultPsnr ? liveResultPsnr.toFixed(2) : exp.bestValPsnr.toFixed(2);

              return (
                <div
                  key={exp.id}
                  onClick={() => {
                    setSelectedExpId(exp.id);
                    setDrawerOpen(true);
                  }}
                  className={`bg-surface-container-low rounded-xl p-5 relative overflow-hidden flex flex-col justify-between transition-all group cursor-pointer border ${
                    isSelected
                      ? 'border-primary-container ring-1 ring-primary-container/40 shadow-[0_0_24px_rgba(56,225,255,0.12)]'
                      : 'border-outline-variant hover:border-outline'
                  }`}
                >
                  <div
                    className="absolute top-0 left-0 right-0 h-[3px]"
                    style={{ backgroundColor: exp.color }}
                  />

                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <span className={`font-mono text-xs font-semibold ${isSelected ? 'text-primary-container' : 'text-outline'}`}>
                          {exp.id}
                        </span>
                        {exp.tag && (
                          <span className="text-[10px] font-mono-metric-sm px-1.5 py-0.5 rounded bg-primary-container/15 text-primary-container font-semibold">
                            {exp.tag}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="bg-surface-container-lowest border border-outline-variant px-2 py-0.5 rounded text-[11px] font-mono text-on-surface">
                          {exp.scale}×
                        </span>
                        <span className="bg-surface-container-lowest border border-outline-variant px-2 py-0.5 rounded text-[11px] font-mono text-outline truncate max-w-[90px]">
                          {exp.loss}
                        </span>
                      </div>
                    </div>

                    <div className="flex items-baseline justify-between">
                      <div className={`font-headline text-lg font-bold transition-colors ${isSelected ? 'text-on-surface' : 'text-on-surface-variant group-hover:text-on-surface'}`}>
                        {exp.model}
                      </div>
                      <span className="text-xs font-mono text-outline">{exp.paramsCount} params</span>
                    </div>

                    {/* Sparkline Curve */}
                    <div className="h-10 w-full my-3 flex items-center">
                      <svg className="w-full h-full overflow-visible" fill="none" viewBox="0 0 140 32">
                        <path
                          d={exp.sparklineD}
                          stroke={exp.color}
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                        />
                        <circle cx="140" cy="16" fill={exp.color} r="3" />
                      </svg>
                    </div>
                  </div>

                  <div className="pt-3 border-t border-outline-variant/60 flex items-center justify-between font-mono text-xs">
                    <div>
                      <span className="text-[10px] text-outline block uppercase">Peak Set5 PSNR</span>
                      <span className={`text-base font-bold ${isSelected ? 'text-primary-container' : 'text-on-surface'}`}>
                        {displayPsnr} dB
                      </span>
                    </div>
                    <div className="text-right">
                      <span className="text-[10px] text-outline block uppercase">Epochs</span>
                      <span className="text-on-surface font-medium">{exp.epochsRun} epochs</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </main>

      {/* BACKDROP OVERLAY WHEN DRAWER IS OPEN */}
      {drawerOpen && (
        <div
          onClick={() => setDrawerOpen(false)}
          className="fixed inset-0 top-14 bottom-7 bg-black/60 backdrop-blur-[1px] z-30 transition-opacity"
        />
      )}

      {/* DETAILED RIGHT DRAWER */}
      {drawerOpen && (
        <aside className="fixed right-0 top-14 bottom-7 w-[540px] bg-surface-container-low border-l border-outline-variant z-40 p-6 flex flex-col justify-between overflow-y-auto shadow-2xl">
          <div className="flex flex-col gap-6">
            {/* Top header */}
            <div className="flex items-center justify-between pb-3 border-b border-outline-variant">
              <div>
                <span className="text-[10px] font-mono uppercase text-primary-container font-semibold tracking-wider block">
                  ARCHITECTURE SPECIFICATION
                </span>
                <h2 className="font-headline font-bold text-xl text-on-surface tracking-tight mt-0.5">
                  {activeExp.id} · {activeExp.model} {activeExp.scale}×
                </h2>
              </div>
              <button
                onClick={() => setDrawerOpen(false)}
                aria-label="Close drawer"
                className="w-8 h-8 rounded-lg flex items-center justify-center text-outline hover:text-on-surface hover:bg-surface-container transition-colors font-mono text-base cursor-pointer"
              >
                ✕
              </button>
            </div>

            {/* Card 1: ARCHITECTURE DATAFLOW */}
            <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-5 flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs uppercase tracking-wider text-outline font-semibold">
                  TENSOR DATAFLOW PIPELINE
                </span>
                <span className="font-mono text-xs text-primary-container font-bold">{activeExp.paramsCount}</span>
              </div>

              {/* Connected nodes sequence */}
              <div className="w-full flex items-center justify-between overflow-x-auto py-2">
                <div className="flex flex-col items-center justify-center bg-surface-container border border-outline-variant rounded-lg px-2.5 py-2 shrink-0 text-center">
                  <span className="font-mono text-xs font-bold text-on-surface">LR</span>
                  <span className="text-[9px] text-outline">Input</span>
                </div>

                {activeExp.model === 'EDSR' ? (
                  <>
                    <div className="flex-1 min-w-[12px] h-px bg-outline-variant relative flex items-center justify-center">
                      <div className="w-1.5 h-1.5 rounded-full bg-[#F59E0B]" />
                    </div>
                    <div className="flex flex-col items-center justify-center bg-surface-container border border-outline-variant rounded-lg px-2 py-1.5 shrink-0 text-center">
                      <span className="font-mono text-[11px] font-semibold text-on-surface">Conv 3×3</span>
                      <span className="text-[9px] text-outline">64 ch</span>
                    </div>
                    <div className="flex-1 min-w-[12px] h-px bg-outline-variant relative flex items-center justify-center">
                      <div className="w-1.5 h-1.5 rounded-full bg-[#F59E0B]" />
                    </div>
                    <div className="flex flex-col items-center justify-center bg-surface-container border border-[#F59E0B]/50 rounded-lg px-2.5 py-1.5 shrink-0 text-center">
                      <span className="font-mono text-[11px] font-bold text-[#F59E0B]">16× ResBlocks</span>
                      <span className="text-[9px] text-outline">0.1 ResScale</span>
                    </div>
                    <div className="flex-1 min-w-[12px] h-px bg-outline-variant relative flex items-center justify-center">
                      <div className="w-1.5 h-1.5 rounded-full bg-[#F59E0B]" />
                    </div>
                    <div className="flex flex-col items-center justify-center bg-surface-container border border-outline-variant rounded-lg px-2 py-1.5 shrink-0 text-center">
                      <span className="font-mono text-[11px] font-bold text-primary-container">PixelShuffle</span>
                      <span className="text-[9px] text-outline">{activeExp.scale}×</span>
                    </div>
                  </>
                ) : activeExp.model === 'FSRCNN' ? (
                  <>
                    <div className="flex-1 min-w-[12px] h-px bg-outline-variant relative flex items-center justify-center">
                      <div className="w-1.5 h-1.5 rounded-full bg-[#34D399]" />
                    </div>
                    <div className="flex flex-col items-center justify-center bg-surface-container border border-outline-variant rounded-lg px-2 py-1.5 shrink-0 text-center">
                      <span className="font-mono text-[11px] font-semibold text-on-surface">FeatureExt</span>
                      <span className="text-[9px] text-outline">5×5 (56)</span>
                    </div>
                    <div className="flex-1 min-w-[12px] h-px bg-outline-variant relative flex items-center justify-center">
                      <div className="w-1.5 h-1.5 rounded-full bg-[#34D399]" />
                    </div>
                    <div className="flex flex-col items-center justify-center bg-surface-container border border-outline-variant rounded-lg px-2 py-1.5 shrink-0 text-center">
                      <span className="font-mono text-[11px] font-bold text-[#34D399]">4× Mapping</span>
                      <span className="text-[9px] text-outline">3×3 (12)</span>
                    </div>
                    <div className="flex-1 min-w-[12px] h-px bg-outline-variant relative flex items-center justify-center">
                      <div className="w-1.5 h-1.5 rounded-full bg-[#34D399]" />
                    </div>
                    <div className="flex flex-col items-center justify-center bg-surface-container border border-outline-variant rounded-lg px-2 py-1.5 shrink-0 text-center">
                      <span className="font-mono text-[11px] font-bold text-primary-container">Deconv</span>
                      <span className="text-[9px] text-outline">9×9 ({activeExp.scale}×)</span>
                    </div>
                  </>
                ) : activeExp.model === 'ESPCN' ? (
                  <>
                    <div className="flex-1 min-w-[12px] h-px bg-outline-variant relative flex items-center justify-center">
                      <div className="w-1.5 h-1.5 rounded-full bg-[#38BDF8]" />
                    </div>
                    <div className="flex flex-col items-center justify-center bg-surface-container border border-outline-variant rounded-lg px-2 py-1.5 shrink-0 text-center">
                      <span className="font-mono text-[11px] font-semibold text-on-surface">Conv 5×5</span>
                      <span className="text-[9px] text-outline">64 ch</span>
                    </div>
                    <div className="flex-1 min-w-[12px] h-px bg-outline-variant relative flex items-center justify-center">
                      <div className="w-1.5 h-1.5 rounded-full bg-[#38BDF8]" />
                    </div>
                    <div className="flex flex-col items-center justify-center bg-surface-container border border-outline-variant rounded-lg px-2 py-1.5 shrink-0 text-center">
                      <span className="font-mono text-[11px] font-semibold text-on-surface">Conv 3×3</span>
                      <span className="text-[9px] text-outline">32 ch</span>
                    </div>
                    <div className="flex-1 min-w-[12px] h-px bg-outline-variant relative flex items-center justify-center">
                      <div className="w-1.5 h-1.5 rounded-full bg-[#38BDF8]" />
                    </div>
                    <div className="flex flex-col items-center justify-center bg-surface-container border border-outline-variant rounded-lg px-2 py-1.5 shrink-0 text-center">
                      <span className="font-mono text-[11px] font-bold text-primary-container">PixelShuffle</span>
                      <span className="text-[9px] text-outline">{activeExp.scale}×</span>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="flex-1 min-w-[12px] h-px bg-outline-variant relative flex items-center justify-center">
                      <div className="w-1.5 h-1.5 rounded-full bg-[#A78BFA]" />
                    </div>
                    <div className="flex flex-col items-center justify-center bg-surface-container border border-outline-variant rounded-lg px-2 py-1.5 shrink-0 text-center">
                      <span className="font-mono text-[11px] font-semibold text-on-surface">Bicubic</span>
                      <span className="text-[9px] text-outline">{activeExp.scale}×</span>
                    </div>
                    <div className="flex-1 min-w-[12px] h-px bg-outline-variant relative flex items-center justify-center">
                      <div className="w-1.5 h-1.5 rounded-full bg-[#A78BFA]" />
                    </div>
                    <div className="flex flex-col items-center justify-center bg-surface-container border border-outline-variant rounded-lg px-2 py-1.5 shrink-0 text-center">
                      <span className="font-mono text-[11px] font-semibold text-on-surface">9-1-5 Conv</span>
                      <span className="text-[9px] text-outline">3 layers</span>
                    </div>
                  </>
                )}

                <div className="flex-1 min-w-[12px] h-px bg-outline-variant relative flex items-center justify-center">
                  <div className="w-1.5 h-1.5 rounded-full bg-primary-container" />
                </div>
                <div className="flex flex-col items-center justify-center bg-surface-container border border-outline-variant rounded-lg px-2.5 py-2 shrink-0 text-center">
                  <span className="font-mono text-xs font-bold text-primary-container">SR</span>
                  <span className="text-[9px] text-outline">Output</span>
                </div>
              </div>
            </div>

            {/* Card 2: TRAINING METRICS & LOG SUMMARY */}
            <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-5 flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs uppercase tracking-wider text-outline font-semibold">
                  TRAINING CONVERGENCE ({activeExp.epochsRun} EPOCHS)
                </span>
                <span className="font-mono text-xs text-on-surface">Loss: {activeExp.loss}</span>
              </div>

              <div className="grid grid-cols-2 gap-3 pt-1">
                <div className="bg-surface-container p-3 rounded-lg border border-outline-variant/60">
                  <span className="text-[10px] font-mono text-outline uppercase block mb-1">Peak Val PSNR</span>
                  <span className="text-xl font-bold font-mono text-good">{peakPsnr} dB</span>
                </div>
                <div className="bg-surface-container p-3 rounded-lg border border-outline-variant/60">
                  <span className="text-[10px] font-mono text-outline uppercase block mb-1">Checkpoint File</span>
                  <span className="text-xs font-mono text-primary-container truncate block font-medium" title={activeExp.checkpoint}>
                    {activeExp.checkpoint}
                  </span>
                </div>
              </div>

              <div className="text-xs text-outline leading-relaxed pt-2 border-t border-outline-variant/40">
                {activeExp.category === 'production'
                  ? 'Trained on 800 DIV2K high-resolution images with random 192×192 HR crops, horizontal/vertical flips, and Adam optimizer.'
                  : 'Trained on 91 T91 grayscale/RGB patches with MSE or Perceptual VGG feature supervision.'}
              </div>
            </div>
          </div>

          {/* Primary Action Button: Open in Studio */}
          <div className="pt-6 border-t border-outline-variant">
            <button
              onClick={handleLoadIntoStudio}
              className="w-full bg-primary-container hover:bg-primary-fixed text-on-primary-container font-headline font-semibold text-sm py-3 px-4 rounded-xl transition-all flex items-center justify-center gap-2 shadow-[0_0_20px_rgba(56,225,255,0.25)] hover:shadow-[0_0_28px_rgba(56,225,255,0.45)] cursor-pointer"
            >
              <span>Load {activeExp.id} into Forensic Studio</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </aside>
      )}
    </div>
  );
};
