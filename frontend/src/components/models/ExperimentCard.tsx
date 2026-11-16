import React from 'react';
import { ExperimentId } from '../../lib/types';
import { Card } from '../ui/Card';
import { Chip } from '../ui/Chip';

export interface ExperimentCardProps {
  id: ExperimentId;
  model: 'SRCNN' | 'ESPCN';
  scale: 2 | 4;
  loss: 'MSE' | 'MSE + Perceptual';
  bestValPsnr: number;
  epochsRun: number;
  maxEpochs?: number;
  earlyStopped?: boolean;
  checkpoint: string;
  onClick: () => void;
  className?: string;
}

export const ExperimentCard: React.FC<ExperimentCardProps> = ({
  id,
  model,
  scale,
  loss,
  bestValPsnr,
  epochsRun,
  maxEpochs = 15,
  earlyStopped = false,
  checkpoint,
  onClick,
  className = '',
}) => {
  const isSRCNN = model === 'SRCNN';
  const stripeColor = isSRCNN ? 'bg-[var(--srcnn)]' : 'bg-[var(--espcn)]';

  return (
    <Card
      variant="surface"
      interactive={true}
      onClick={onClick}
      className={`relative overflow-hidden flex flex-col justify-between space-y-4 border border-[var(--border)] hover:border-[var(--accent)]/50 ${className}`}
    >
      {/* Top Model Color Stripe */}
      <div className={`absolute top-0 inset-x-0 h-1 ${stripeColor}`} />

      {/* Header */}
      <div className="flex items-center justify-between pt-1">
        <div className="flex items-center gap-2">
          <span className="font-mono text-base font-bold text-[var(--text)]">{id}</span>
          <Chip variant={isSRCNN ? 'srcnn' : 'espcn'} size="sm">
            {model}
          </Chip>
        </div>
        <div className="flex items-center gap-1.5">
          <Chip variant="raised" size="sm">
            {scale}×
          </Chip>
          <Chip variant={loss === 'MSE' ? 'muted' : 'amber'} size="sm">
            {loss === 'MSE' ? 'MSE' : 'Perc'}
          </Chip>
        </div>
      </div>

      {/* Metric Display */}
      <div className="py-2 px-3 rounded-[var(--radius-control)] bg-[var(--raised)] border border-[var(--border)]">
        <div className="text-[10px] uppercase font-mono text-[var(--muted)]">Best Val PSNR</div>
        <div className="font-mono text-xl font-bold text-[var(--good)] mt-0.5">
          {bestValPsnr.toFixed(2)}{' '}
          <span className="text-xs text-[var(--muted)] font-normal">dB</span>
        </div>
      </div>

      {/* Footer Info */}
      <div className="flex items-center justify-between text-xs font-mono text-[var(--muted)] pt-1 border-t border-[var(--border)]">
        <span>
          Epochs: {epochsRun}/{maxEpochs}
        </span>
        {earlyStopped ? (
          <span className="text-[10px] px-1.5 py-0.2 rounded bg-[var(--amber)]/10 text-[var(--amber)] border border-[var(--amber)]/30">
            Early-stopped
          </span>
        ) : (
          <span className="text-[10px] text-[var(--muted)]">Completed</span>
        )}
      </div>
    </Card>
  );
};

