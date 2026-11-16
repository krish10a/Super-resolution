import React from 'react';
import { Card } from '../ui/Card';

export interface RunInfoCardProps {
  inputSize?: [number, number];
  outputSize?: [number, number];
  latencyMs?: number;
  device?: string;
  checkpoint?: string;
  className?: string;
}

export const RunInfoCard: React.FC<RunInfoCardProps> = ({
  inputSize = [128, 128],
  outputSize = [512, 512],
  latencyMs = 42,
  device = 'CUDA · RTX 2050',
  checkpoint = 'espcn_4x_mse.pt',
  className = '',
}) => {
  return (
    <Card variant="surface" className={`space-y-3 ${className}`}>
      <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">
        <span>Run Telemetry</span>
        <span className="font-mono text-[10px] text-[var(--accent)]">{device}</span>
      </div>

      <div className="grid grid-cols-2 gap-3 pt-1">
        <div className="p-2.5 rounded bg-[var(--raised)] border border-[var(--border)]">
          <div className="text-[10px] text-[var(--muted)] uppercase font-mono">Input Dim</div>
          <div className="font-mono text-xs font-bold text-[var(--text)] mt-0.5">
            {inputSize[0]} × {inputSize[1]} px
          </div>
        </div>

        <div className="p-2.5 rounded bg-[var(--raised)] border border-[var(--border)]">
          <div className="text-[10px] text-[var(--muted)] uppercase font-mono">Output Dim</div>
          <div className="font-mono text-xs font-bold text-[var(--accent)] mt-0.5">
            {outputSize[0]} × {outputSize[1]} px
          </div>
        </div>

        <div className="p-2.5 rounded bg-[var(--raised)] border border-[var(--border)]">
          <div className="text-[10px] text-[var(--muted)] uppercase font-mono">Latency</div>
          <div className="font-mono text-xs font-bold text-[var(--good)] mt-0.5">
            {latencyMs} ms
          </div>
        </div>

        <div className="p-2.5 rounded bg-[var(--raised)] border border-[var(--border)] truncate">
          <div className="text-[10px] text-[var(--muted)] uppercase font-mono">Checkpoint</div>
          <div className="font-mono text-[11px] font-semibold text-[var(--text)] truncate mt-0.5" title={checkpoint}>
            {checkpoint}
          </div>
        </div>
      </div>
    </Card>
  );
};

