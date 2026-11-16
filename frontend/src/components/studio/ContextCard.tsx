import React from 'react';
import { Link } from 'react-router-dom';
import { Card } from '../ui/Card';
import { Chip } from '../ui/Chip';

export interface ContextCardProps {
  rank: number;
  total: number;
  dataset: string;
  scale: number;
  metric: string;
  deltaToBicubic: number;
  className?: string;
}

export const ContextCard: React.FC<ContextCardProps> = ({
  rank = 1,
  total = 8,
  dataset = 'Set5',
  scale = 2,
  metric = 'PSNR',
  deltaToBicubic = -0.24,
  className = '',
}) => {
  return (
    <Card variant="surface" className={`space-y-3 ${className}`}>
      <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">
        <span>Benchmark Context</span>
        <Chip variant="raised" size="sm">
          #{rank} of {total}
        </Chip>
      </div>

      <div className="p-3 rounded bg-[var(--raised)] border border-[var(--border)] space-y-1.5">
        <div className="text-xs text-[var(--text)] font-medium">
          Ranks <strong className="text-[var(--accent)] font-mono">#{rank}</strong> across all {total} models on{' '}
          <strong className="text-[var(--text)]">{dataset}</strong> ×{scale} {metric}.
        </div>
        <div className="text-[11px] font-mono text-[var(--muted)]">
          Delta vs Bicubic Baseline: <span className={deltaToBicubic >= 0 ? 'text-[var(--good)]' : 'text-[var(--bad)]'}>
            {deltaToBicubic > 0 ? '+' : ''}{deltaToBicubic.toFixed(2)} dB
          </span>
        </div>
      </div>

      <Link
        to={`/benchmarks?dataset=${dataset}&scale=${scale}`}
        className="inline-flex items-center gap-1.5 text-xs text-[var(--accent)] hover:underline font-mono pt-1"
      >
        <span>Inspect full benchmark matrix</span>
        <span>→</span>
      </Link>
    </Card>
  );
};

