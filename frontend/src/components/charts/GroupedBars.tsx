import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
  Legend,
  CartesianGrid,
} from 'recharts';
import { EvaluationResults, ExperimentId } from '../../lib/types';

export interface GroupedBarsProps {
  results: EvaluationResults | null;
  dataset: 'Set5' | 'Set14' | 'BSD100';
  scale: 2 | 4;
  metric: 'psnr' | 'ssim' | 'lpips';
  className?: string;
}

export const GroupedBars: React.FC<GroupedBarsProps> = ({
  results,
  dataset,
  scale,
  metric,
  className = '',
}) => {
  if (!results) {
    return <div className="h-64 flex items-center justify-center font-mono text-xs text-[var(--muted)]">Loading chart...</div>;
  }

  // Group by model (SRCNN vs ESPCN) and loss (MSE vs Perceptual)
  const data = [
    {
      category: 'SRCNN',
      MSE: getMetric(results, scale === 2 ? 'E1' : 'E3', dataset, metric),
      Perceptual: getMetric(results, scale === 2 ? 'E2' : 'E4', dataset, metric),
    },
    {
      category: 'ESPCN',
      MSE: getMetric(results, scale === 2 ? 'E5' : 'E7', dataset, metric),
      Perceptual: getMetric(results, scale === 2 ? 'E6' : 'E8', dataset, metric),
    },
  ];

  return (
    <div className={`h-64 w-full p-2 ${className}`}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1F2832" vertical={false} />
          <XAxis dataKey="category" stroke="#8793A2" fontSize={12} tickLine={false} />
          <YAxis
            stroke="#8793A2"
            fontSize={11}
            tickLine={false}
            domain={metric === 'psnr' ? [20, 32] : [0.5, 1.0]}
          />
          <RechartsTooltip
            contentStyle={{
              backgroundColor: '#0E1217',
              borderColor: '#1F2832',
              borderRadius: '8px',
              fontFamily: 'JetBrains Mono',
              fontSize: '12px',
              color: '#E6EBF1',
            }}
          />
          <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'JetBrains Mono' }} />
          <Bar dataKey="MSE" fill="#38E1FF" radius={[4, 4, 0, 0]} />
          <Bar dataKey="Perceptual" fill="#FFB020" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

function getMetric(
  results: EvaluationResults,
  id: ExperimentId,
  dataset: string,
  metric: string
): number {
  const d = (results[id]?.results as any)?.[dataset];
  if (!d) return 0;
  const val = metric === 'psnr' ? d.avg_psnr : metric === 'ssim' ? d.avg_ssim : d.avg_lpips || 0.15;
  return Math.round(val * 100) / 100;
}

