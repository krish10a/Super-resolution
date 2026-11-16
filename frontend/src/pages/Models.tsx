import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ExperimentId } from '../lib/types';
import { useResults } from '../hooks/useResults';
import { useTrainingLog } from '../hooks/useTrainingLog';

interface ExperimentMeta {
  id: ExperimentId;
  model: 'SRCNN' | 'ESPCN';
  scale: 2 | 4;
  loss: 'MSE' | 'MSE + Perceptual';
  bestValPsnr: number;
  checkpoint: string;
  epochsRun: number;
  color: string;
  sparklineD: string;
}

export const Models: React.FC = () => {
  const navigate = useNavigate();
  const [selectedExpId, setSelectedExpId] = useState<ExperimentId>('E7');
  const [drawerOpen, setDrawerOpen] = useState(true);
  const { results } = useResults();

  // Training log for the currently selected experiment
  const { epochs } = useTrainingLog(selectedExpId);

  // Specifications for E1–E8
  const experimentSpecs: ExperimentMeta[] = [
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
    },
    {
      id: 'E2',
      model: 'SRCNN',
      scale: 2,
      loss: 'MSE + Perceptual',
      bestValPsnr: 25.63,
      checkpoint: 'srcnn_2x_mse_perceptual.pt',
      epochsRun: 15,
      color: '#A78BFA',
      sparklineD: 'M0,28 L15,23 L35,18 L60,15 L85,13 L115,11 L140,11',
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
    },
    {
      id: 'E4',
      model: 'SRCNN',
      scale: 4,
      loss: 'MSE + Perceptual',
      bestValPsnr: 23.97,
      checkpoint: 'srcnn_4x_mse_perceptual.pt',
      epochsRun: 15,
      color: '#A78BFA',
      sparklineD: 'M0,30 L15,26 L35,22 L60,19 L85,17 L115,15 L140,15',
    },
    {
      id: 'E5',
      model: 'ESPCN',
      scale: 2,
      loss: 'MSE',
      bestValPsnr: 27.68,
      checkpoint: 'espcn_2x_mse.pt',
      epochsRun: 15,
      color: '#22D3EE',
      sparklineD: 'M0,28 L15,19 L35,12 L60,8 L85,5 L115,3 L140,3',
    },
    {
      id: 'E6',
      model: 'ESPCN',
      scale: 2,
      loss: 'MSE + Perceptual',
      bestValPsnr: 26.42,
      checkpoint: 'espcn_2x_mse_perceptual_verified.pt',
      epochsRun: 15,
      color: '#22D3EE',
      sparklineD: 'M0,28 L15,20 L35,14 L60,10 L85,7 L115,5 L140,5',
    },
    {
      id: 'E7',
      model: 'ESPCN',
      scale: 4,
      loss: 'MSE',
      bestValPsnr: 24.98,
      checkpoint: 'espcn_4x_mse_verified.pt',
      epochsRun: 15,
      color: '#22D3EE',
      sparklineD: 'M0,30 L15,26 L35,22 L60,19 L85,17 L115,16 L140,16',
    },
    {
      id: 'E8',
      model: 'ESPCN',
      scale: 4,
      loss: 'MSE + Perceptual',
      bestValPsnr: 22.33,
      checkpoint: 'espcn_4x_mse_perceptual_verified.pt',
      epochsRun: 15,
      color: '#22D3EE',
      sparklineD: 'M0,30 L15,27 L35,24 L60,21 L85,19 L115,17 L140,17',
    },
  ];

  const activeExp = experimentSpecs.find((e) => e.id === selectedExpId) || experimentSpecs[6];
  const isSRCNN = activeExp.model === 'SRCNN';

  const peakPsnr = epochs.length > 0
    ? Math.max(...epochs.map((e) => e.val_psnr)).toFixed(2)
    : (results?.[activeExp.id]?.results?.BSD100?.avg_psnr?.toFixed(2) || activeExp.bestValPsnr.toFixed(2));

  const handleLoadIntoStudio = () => {
    const lossParam = activeExp.loss.includes('Perceptual') ? 'Perceptual' : 'MSE';
    navigate(`/studio?model=${activeExp.model}&scale=${activeExp.scale}&loss=${lossParam}`);
  };

  return (
    <div className="w-full flex h-[calc(100vh-84px)] overflow-hidden relative">
      {/* MAIN VIEWPORT: 4x2 Grid of Experiment Cards */}
      <main className="w-full pt-4 pb-7 flex-1 flex flex-col justify-start overflow-y-auto">
        <div className="max-w-[1440px] w-full mx-auto px-8 py-6 flex flex-col">
          {/* SIMPLIFIED HEADER */}
          <div className="flex flex-col mb-6">
            <h1 className="font-headline font-semibold text-2xl text-[#E2E2E7] tracking-tight">
              Experiments
            </h1>
            <p className="font-mono text-[12px] text-[#7E8B9B] mt-1">
              8 runs · T91 · 15 epochs max
            </p>
          </div>

          {/* 4x2 EXPERIMENT GRID */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
            {experimentSpecs.map((exp) => {
              const isSelected = selectedExpId === exp.id;
              const liveResultPsnr = results?.[exp.id]?.results?.BSD100?.avg_psnr;
              const displayPsnr = liveResultPsnr ? liveResultPsnr.toFixed(2) : exp.bestValPsnr.toFixed(2);

              return (
                <div
                  key={exp.id}
                  onClick={() => {
                    setSelectedExpId(exp.id);
                    setDrawerOpen(true);
                  }}
                  className={`bg-[#0E1217] rounded-[12px] p-6 relative overflow-hidden flex flex-col justify-between transition-colors group cursor-pointer ${
                    isSelected
                      ? 'border border-[#38E1FF]/60 ring-1 ring-[#38E1FF]/40 shadow-[0_0_20px_rgba(56,225,255,0.06)]'
                      : 'border border-[#1F2832] hover:border-outline-variant/70'
                  }`}
                >
                  <div
                    className="absolute top-0 left-0 right-0 h-[3px]"
                    style={{ backgroundColor: exp.color }}
                  />

                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <span className={`font-mono text-[12px] font-medium ${isSelected ? 'text-[#38E1FF]' : 'text-[#7E8B9B]'}`}>
                        {exp.id}
                      </span>
                      <div className="flex items-center gap-1.5">
                        <span className="bg-[#07090C] border border-[#1F2832] px-2 py-0.5 rounded text-[11px] font-mono text-[#E2E2E7]">
                          {exp.scale}×
                        </span>
                        <span className="bg-[#07090C] border border-[#1F2832] px-2 py-0.5 rounded text-[11px] font-mono text-[#7E8B9B] truncate max-w-[80px]">
                          {exp.loss === 'MSE' ? 'MSE' : 'Perceptual'}
                        </span>
                      </div>
                    </div>

                    <div className={`font-headline text-[17px] font-semibold transition-colors ${isSelected ? 'text-white' : 'text-[#E2E2E7] group-hover:text-white'}`}>
                      {exp.model}
                    </div>

                    {/* Sparkline */}
                    <div className="h-10 w-full my-4 flex items-center">
                      <svg className="w-full h-full overflow-visible" fill="none" viewBox="0 0 140 32">
                        <path
                          d={exp.sparklineD}
                          stroke={exp.color}
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="1.75"
                          strokeDasharray={exp.loss.includes('Perceptual') ? '3 2' : undefined}
                        />
                        <circle cx="140" cy="16" fill={exp.color} r="2.5" />
                      </svg>
                    </div>
                  </div>

                  <div className="pt-3 border-t border-[#1F2832] flex flex-col gap-1">
                    <div className="font-mono text-[11px] text-[#7E8B9B]">best val PSNR</div>
                    <div className="flex items-baseline justify-between font-mono">
                      <span className={`text-[17px] font-semibold ${isSelected ? 'text-[#38E1FF]' : 'text-[#E2E2E7]'}`}>
                        {displayPsnr} dB
                      </span>
                      <span className="text-[11px] text-[#7E8B9B]">{exp.epochsRun}/15 epochs</span>
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
          className="fixed inset-0 top-14 bottom-7 bg-[#07090C]/65 backdrop-blur-[1px] z-30 transition-opacity"
        />
      )}

      {/* SIMPLIFIED RIGHT DRAWER (520px wide) */}
      {drawerOpen && (
        <aside className="fixed right-0 top-14 bottom-7 w-[520px] bg-[#0E1217] border-l border-[#1F2832] z-40 p-6 flex flex-col justify-between overflow-y-auto">
          <div className="flex flex-col gap-6">
            {/* Top header */}
            <div className="flex items-center justify-between pb-2 border-b border-[#1F2832]">
              <h2 className="font-headline font-semibold text-[19px] text-[#E2E2E7] tracking-tight">
                {activeExp.id} · {activeExp.model} {activeExp.scale}× · {activeExp.loss}
              </h2>
              <button
                onClick={() => setDrawerOpen(false)}
                aria-label="Close drawer"
                className="w-8 h-8 rounded-lg flex items-center justify-center text-[#7E8B9B] hover:text-[#E2E2E7] hover:bg-[#1A222C] transition-colors font-mono text-[16px] leading-none cursor-pointer"
              >
                ✕
              </button>
            </div>

            {/* Card 1: ARCHITECTURE */}
            <div className="bg-[#07090C] border border-[#1F2832] rounded-[12px] p-6 flex flex-col">
              <span className="font-mono text-[11px] uppercase tracking-wider text-[#7E8B9B] font-medium mb-4">
                ARCHITECTURE
              </span>

              {/* Connected nodes sequence */}
              <div className="w-full flex items-center justify-between overflow-x-auto py-2">
                {/* Node: LR */}
                <div className="flex flex-col items-center justify-center bg-[#0E1217] border border-[#1F2832] rounded-lg px-2.5 py-2.5 min-w-[50px] shrink-0 text-center">
                  <span className="font-mono text-[12px] font-semibold text-[#E2E2E7]">LR</span>
                </div>

                {isSRCNN ? (
                  <>
                    <div className="flex-1 min-w-[14px] flex items-center justify-center relative px-1">
                      <div className="w-full h-px bg-[#1F2832]" />
                      <div className="absolute w-1.5 h-1.5 rounded-full bg-[#A78BFA] shadow-[0_0_6px_#A78BFA]" />
                    </div>
                    <div className="flex flex-col items-center justify-center bg-[#0E1217] border border-[#1F2832] rounded-lg px-2.5 py-1.5 shrink-0 text-center">
                      <span className="font-mono text-[12px] font-semibold text-[#E2E2E7]">Bicubic</span>
                      <span className="font-mono text-[10px] text-[#7E8B9B]">{activeExp.scale}×</span>
                    </div>
                    <div className="flex-1 min-w-[14px] flex items-center justify-center relative px-1">
                      <div className="w-full h-px bg-[#1F2832]" />
                      <div className="absolute w-1.5 h-1.5 rounded-full bg-[#A78BFA] shadow-[0_0_6px_#A78BFA]" />
                    </div>
                    <div className="flex flex-col items-center justify-center bg-[#0E1217] border border-[#1F2832] rounded-lg px-2.5 py-1.5 shrink-0 text-center">
                      <span className="font-mono text-[12px] font-semibold text-[#E2E2E7]">Conv</span>
                      <span className="font-mono text-[10px] text-[#7E8B9B]">9×9</span>
                    </div>
                    <div className="flex-1 min-w-[14px] flex items-center justify-center relative px-1">
                      <div className="w-full h-px bg-[#1F2832]" />
                      <div className="absolute w-1.5 h-1.5 rounded-full bg-[#A78BFA] shadow-[0_0_6px_#A78BFA]" />
                    </div>
                    <div className="flex flex-col items-center justify-center bg-[#0E1217] border border-[#1F2832] rounded-lg px-2.5 py-1.5 shrink-0 text-center">
                      <span className="font-mono text-[12px] font-semibold text-[#E2E2E7]">Conv</span>
                      <span className="font-mono text-[10px] text-[#7E8B9B]">1×1</span>
                    </div>
                    <div className="flex-1 min-w-[14px] flex items-center justify-center relative px-1">
                      <div className="w-full h-px bg-[#1F2832]" />
                      <div className="absolute w-1.5 h-1.5 rounded-full bg-[#A78BFA] shadow-[0_0_6px_#A78BFA]" />
                    </div>
                    <div className="flex flex-col items-center justify-center bg-[#0E1217] border border-[#1F2832] rounded-lg px-2.5 py-1.5 shrink-0 text-center">
                      <span className="font-mono text-[12px] font-semibold text-[#E2E2E7]">Conv</span>
                      <span className="font-mono text-[10px] text-[#7E8B9B]">5×5</span>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="flex-1 min-w-[14px] flex items-center justify-center relative px-1">
                      <div className="w-full h-px bg-[#1F2832]" />
                      <div className="absolute w-1.5 h-1.5 rounded-full bg-[#38E1FF] shadow-[0_0_6px_#38E1FF]" />
                    </div>
                    <div className="flex flex-col items-center justify-center bg-[#0E1217] border border-[#1F2832] rounded-lg px-2.5 py-1.5 shrink-0 text-center">
                      <span className="font-mono text-[12px] font-semibold text-[#E2E2E7]">Conv</span>
                      <span className="font-mono text-[10px] text-[#7E8B9B]">5×5</span>
                    </div>
                    <div className="flex-1 min-w-[14px] flex items-center justify-center relative px-1">
                      <div className="w-full h-px bg-[#1F2832]" />
                      <div className="absolute w-1.5 h-1.5 rounded-full bg-[#38E1FF] shadow-[0_0_6px_#38E1FF]" />
                    </div>
                    <div className="flex flex-col items-center justify-center bg-[#0E1217] border border-[#1F2832] rounded-lg px-2.5 py-1.5 shrink-0 text-center">
                      <span className="font-mono text-[12px] font-semibold text-[#E2E2E7]">Conv</span>
                      <span className="font-mono text-[10px] text-[#7E8B9B]">3×3</span>
                    </div>
                    <div className="flex-1 min-w-[14px] flex items-center justify-center relative px-1">
                      <div className="w-full h-px bg-[#1F2832]" />
                      <div className="absolute w-1.5 h-1.5 rounded-full bg-[#38E1FF] shadow-[0_0_6px_#38E1FF]" />
                    </div>
                    <div className="flex flex-col items-center justify-center bg-[#0E1217] border border-[#1F2832] rounded-lg px-2.5 py-1.5 shrink-0 text-center">
                      <span className="font-mono text-[12px] font-semibold text-[#E2E2E7]">Conv</span>
                      <span className="font-mono text-[10px] text-[#7E8B9B]">3×3</span>
                    </div>
                    <div className="flex-1 min-w-[14px] flex items-center justify-center relative px-1">
                      <div className="w-full h-px bg-[#1F2832]" />
                      <div className="absolute w-1.5 h-1.5 rounded-full bg-[#38E1FF] shadow-[0_0_6px_#38E1FF]" />
                    </div>
                    <div className="flex flex-col items-center justify-center bg-[#0E1217] border border-[#1F2832] rounded-lg px-2.5 py-1.5 shrink-0 text-center">
                      <span className="font-mono text-[11px] font-semibold text-[#38E1FF]">PixelShuffle</span>
                      <span className="font-mono text-[10px] text-[#7E8B9B]">×{activeExp.scale}</span>
                    </div>
                  </>
                )}

                <div className="flex-1 min-w-[14px] flex items-center justify-center relative px-1">
                  <div className="w-full h-px bg-[#1F2832]" />
                  <div className="absolute w-1.5 h-1.5 rounded-full bg-[#38E1FF] shadow-[0_0_6px_#38E1FF]" />
                </div>

                {/* Node: SR */}
                <div className="flex flex-col items-center justify-center bg-[#0E1217] border border-[#1F2832] rounded-lg px-2.5 py-2.5 min-w-[50px] shrink-0 text-center">
                  <span className="font-mono text-[12px] font-semibold text-[#E2E2E7]">SR</span>
                </div>
              </div>
            </div>

            {/* Card 2: TRAINING */}
            <div className="bg-[#07090C] border border-[#1F2832] rounded-[12px] p-6 flex flex-col">
              <div className="flex items-center justify-between mb-4">
                <span className="font-mono text-[11px] uppercase tracking-wider text-[#7E8B9B] font-medium">
                  TRAINING
                </span>
                <div className="flex items-center gap-4 font-mono text-[10px]">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-[2px] bg-[#38E1FF] inline-block rounded-full" />
                    <span className="text-[#E2E2E7]">val PSNR</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-[2px] bg-[#8793A2] inline-block rounded-full" />
                    <span className="text-[#7E8B9B]">train loss</span>
                  </div>
                </div>
              </div>

              {/* Chart: PSNR & Loss across 15 epochs */}
              <div className="relative w-full h-44 mb-3">
                <svg className="w-full h-full overflow-visible" fill="none" viewBox="0 0 420 150">
                  <line stroke="#1F2832" strokeDasharray="3 3" strokeWidth="1" x1="36" x2="410" y1="20" y2="20" />
                  <line stroke="#1F2832" strokeDasharray="3 3" strokeWidth="1" x1="36" x2="410" y1="58" y2="58" />
                  <line stroke="#1F2832" strokeDasharray="3 3" strokeWidth="1" x1="36" x2="410" y1="96" y2="96" />
                  <line stroke="#1F2832" strokeWidth="1" x1="36" x2="410" y1="134" y2="134" />

                  <text fill="#7E8B9B" fontFamily="JetBrains Mono" fontSize="10" textAnchor="end" x="30" y="24">24 dB</text>
                  <text fill="#7E8B9B" fontFamily="JetBrains Mono" fontSize="10" textAnchor="end" x="30" y="62">22 dB</text>
                  <text fill="#7E8B9B" fontFamily="JetBrains Mono" fontSize="10" textAnchor="end" x="30" y="100">20 dB</text>
                  <text fill="#7E8B9B" fontFamily="JetBrains Mono" fontSize="10" textAnchor="end" x="30" y="137">18 dB</text>

                  {/* Train Loss Curve */}
                  <path
                    d="M 40,30 C 70,68 110,95 160,110 C 220,122 300,126 410,128"
                    stroke="#8793A2"
                    strokeLinecap="round"
                    strokeWidth="1.75"
                  />
                  {/* Val PSNR Curve */}
                  <path
                    d="M 40,126 C 70,105 110,75 160,54 C 220,42 300,38 410,36"
                    stroke="#38E1FF"
                    strokeLinecap="round"
                    strokeWidth="2"
                  />
                  <circle cx="410" cy="36" fill="#38E1FF" r="3" />

                  {/* X Axis Labels */}
                  <text fill="#7E8B9B" fontFamily="JetBrains Mono" fontSize="10" textAnchor="start" x="40" y="148">Ep 1</text>
                  <text fill="#7E8B9B" fontFamily="JetBrains Mono" fontSize="10" textAnchor="middle" x="225" y="148">Ep 8</text>
                  <text fill="#7E8B9B" fontFamily="JetBrains Mono" fontSize="10" textAnchor="end" x="410" y="148">Ep 15</text>
                </svg>
              </div>

              {/* ONE mono summary line */}
              <div className="pt-3 border-t border-[#1F2832]">
                <p className="font-mono text-[12px] text-[#E2E2E7]">
                  Best val PSNR {peakPsnr} dB · 15/15 epochs
                </p>
              </div>
            </div>
          </div>

          {/* Primary Button: Open in Studio */}
          <div className="pt-6">
            <button
              onClick={handleLoadIntoStudio}
              className="w-full bg-[#38E1FF] hover:bg-[#6beaff] text-[#07090C] font-sans font-semibold text-[14px] py-3 px-4 rounded-[8px] transition-colors flex items-center justify-center gap-1.5 shadow-[0_0_15px_rgba(56,225,255,0.2)] cursor-pointer"
            >
              <span>Open in Studio</span>
              <span className="font-mono">→</span>
            </button>
          </div>
        </aside>
      )}
    </div>
  );
};
