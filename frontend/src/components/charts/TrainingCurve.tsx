import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ReferenceLine,
} from 'recharts';
import { TrainingEpoch } from '../../lib/types';

export interface TrainingCurveProps {
  epochs: TrainingEpoch[];
  hasWarmup?: boolean;
  className?: string;
}

export const TrainingCurve: React.FC<TrainingCurveProps> = ({
  epochs,
  hasWarmup = false,
  className = '',
}) => {
  if (!epochs || epochs.length === 0) {
    return (
      <div className="h-60 flex items-center justify-center font-mono text-xs text-[var(--muted)]">
        Loading training logs...
      </div>
    );
  }

  return (
    <div className={`space-y-2 ${className}`}>
      <div className="flex items-center justify-between text-xs font-mono text-[var(--muted)]">
        <span>Epochs 1 → {epochs.length}</span>
        {hasWarmup && (
          <span className="text-[var(--amber)] text-[11px] flex items-center gap-1">
            <span className="w-2 h-0.5 border-t border-dashed border-[var(--amber)]" />
            Epoch 3 MSE Warmup Boundary
          </span>
        )}
      </div>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={epochs} margin={{ top: 10, right: 15, left: -15, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1F2832" />
            <XAxis dataKey="epoch" stroke="#8793A2" fontSize={11} tickLine={false} />
            <YAxis
              yAxisId="psnr"
              stroke="#34D399"
              fontSize={11}
              tickLine={false}
              domain={['auto', 'auto']}
              unit=" dB"
            />
            <YAxis
              yAxisId="loss"
              orientation="right"
              stroke="#38E1FF"
              fontSize={11}
              tickLine={false}
              domain={['auto', 'auto']}
            />
            <RechartsTooltip
              contentStyle={{
                backgroundColor: '#0E1217',
                borderColor: '#1F2832',
                borderRadius: '8px',
                fontFamily: 'JetBrains Mono',
                fontSize: '11px',
                color: '#E6EBF1',
              }}
            />
            <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'JetBrains Mono' }} />
            {hasWarmup && (
              <ReferenceLine
                x={3}
                stroke="#FFB020"
                strokeDasharray="4 4"
                label={{ value: 'Warmup End', fill: '#FFB020', fontSize: 10, position: 'top' }}
              />
            )}
            <Line
              yAxisId="psnr"
              type="monotone"
              dataKey="val_psnr"
              name="Val PSNR (dB)"
              stroke="#34D399"
              strokeWidth={2}
              dot={{ r: 2 }}
            />
            <Line
              yAxisId="loss"
              type="monotone"
              dataKey="train_loss"
              name="Train Loss"
              stroke="#38E1FF"
              strokeWidth={1.5}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

