import React from 'react';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip as RechartsTooltip,
  CartesianGrid,
  Legend,
} from 'recharts';
import { EvaluationResults, ExperimentId } from '../../lib/types';

export interface TradeoffScatterProps {
  results: EvaluationResults | null;
  dataset: 'Set5' | 'Set14' | 'BSD100';
  className?: string;
}

export const TradeoffScatter: React.FC<TradeoffScatterProps> = ({
  results,
  dataset,
  className = '',
}) => {
  if (!results) return null;

  // Gather pairs:
  // SRCNN: (E1, E2) 2x, (E3, E4) 4x
  // ESPCN: (E5, E6) 2x, (E7, E8) 4x
  const getDataPoint = (id: ExperimentId) => {
    const d = (results[id]?.results as any)?.[dataset];
    const cfg = results[id]?.config;
    if (!d || !cfg) return null;
    return {
      id,
      model: cfg.model.toUpperCase(),
      scale: `${cfg.scale}×`,
      loss: cfg.loss.includes('perceptual') ? 'Perceptual' : 'MSE',
      psnr: Math.round(d.avg_psnr * 100) / 100,
      lpips: Math.round((d.avg_lpips ?? 0.15) * 1000) / 1000,
    };
  };

  const srcnnData = ['E1', 'E2', 'E3', 'E4'].map(getDataPoint).filter(Boolean);
  const espcnData = ['E5', 'E6', 'E7', 'E8'].map(getDataPoint).filter(Boolean);

  return (
    <div className={`p-4 rounded-[var(--radius-card)] border border-[var(--border)] bg-[var(--surface)] space-y-3 ${className}`}>
      <div className="flex items-center justify-between text-xs">
        <div>
          <span className="font-semibold text-[var(--text)]">Perception-Distortion Trade-off</span>
          <span className="text-[var(--muted)] ml-2 font-mono">PSNR (Fidelity) vs. LPIPS (Lower is more realistic)</span>
        </div>
        <div className="flex items-center gap-4 text-[11px] font-mono text-[var(--muted)]">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#A78BFA]" /> SRCNN
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#22D3EE]" /> ESPCN
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 border border-white rounded-full" /> Perceptual (Hollow)
          </span>
        </div>
      </div>

      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 15, right: 20, bottom: 15, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1F2832" />
            <XAxis
              type="number"
              dataKey="psnr"
              name="PSNR"
              unit=" dB"
              domain={['auto', 'auto']}
              stroke="#8793A2"
              fontSize={11}
              tickLine={false}
              label={{ value: 'PSNR (dB) — Higher is better', position: 'bottom', offset: 0, fill: '#8793A2', fontSize: 11 }}
            />
            <YAxis
              type="number"
              dataKey="lpips"
              name="LPIPS"
              domain={['auto', 'auto']}
              stroke="#8793A2"
              fontSize={11}
              tickLine={false}
              label={{ value: 'LPIPS — Lower is better', angle: -90, position: 'left', fill: '#8793A2', fontSize: 11 }}
            />
            <ZAxis range={[90, 90]} />
            <RechartsTooltip
              cursor={{ strokeDasharray: '3 3' }}
              contentStyle={{
                backgroundColor: '#0E1217',
                borderColor: '#1F2832',
                borderRadius: '8px',
                fontFamily: 'JetBrains Mono',
                fontSize: '11px',
                color: '#E6EBF1',
              }}
              formatter={(val: any, name: any) => [val, name]}
            />
            <Scatter name="SRCNN" data={srcnnData as any} fill="#A78BFA" />
            <Scatter name="ESPCN" data={espcnData as any} fill="#22D3EE" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

