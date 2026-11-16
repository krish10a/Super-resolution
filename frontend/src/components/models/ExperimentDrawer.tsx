import React from 'react';
import { ExperimentId, ExperimentData } from '../../lib/types';
import { useTrainingLog } from '../../hooks/useTrainingLog';
import { Drawer } from '../ui/Drawer';
import { Chip } from '../ui/Chip';
import { ArchitectureDiagram } from './ArchitectureDiagram';
import { TrainingCurve } from '../charts/TrainingCurve';

export interface ExperimentDrawerProps {
  expId: ExperimentId | null;
  expData: ExperimentData | null;
  open: boolean;
  onClose: () => void;
}

export const ExperimentDrawer: React.FC<ExperimentDrawerProps> = ({
  expId,
  expData,
  open,
  onClose,
}) => {
  const { epochs, hasWarmup } = useTrainingLog(expId);

  if (!expId || !expData) return null;

  const model = expData.config.model.toUpperCase() as 'SRCNN' | 'ESPCN';
  const scale = expData.config.scale as 2 | 4;
  const isMSE = expData.config.loss === 'mse';

  return (
    <Drawer
      open={open}
      onClose={onClose}
      title={`${expId} — ${model} ${scale}×`}
      subtitle={expData.config.ckpt}
      width="max-w-[580px]"
    >
      <div className="space-y-6">
        {/* Badges Strip */}
        <div className="flex items-center gap-2">
          <Chip variant={model === 'SRCNN' ? 'srcnn' : 'espcn'}>{model}</Chip>
          <Chip variant="raised">{scale}× Upscale</Chip>
          <Chip variant={isMSE ? 'muted' : 'amber'}>
            {isMSE ? 'MSE Loss' : 'MSE + Perceptual'}
          </Chip>
        </div>

        {/* Section 1: Architecture Pipeline */}
        <div className="space-y-2">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">
            Computational Graph
          </h4>
          <ArchitectureDiagram model={model} scale={scale} />
        </div>

        {/* Section 2: Training Convergence Curve */}
        <div className="space-y-2 pt-2 border-t border-[var(--border)]">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">
            Training & Validation Convergence
          </h4>
          <TrainingCurve epochs={epochs} hasWarmup={hasWarmup} />
        </div>

        {/* Section 3: Checkpoint Metadata */}
        <div className="space-y-2 pt-2 border-t border-[var(--border)]">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">
            Training Protocol
          </h4>
          <div className="p-3 rounded-[var(--radius-control)] bg-[var(--raised)] border border-[var(--border)] font-mono text-xs space-y-1.5 text-[var(--muted)]">
            <div className="flex justify-between">
              <span>Optimizer</span>
              <span className="text-[var(--text)]">Adam (lr=2e-4)</span>
            </div>
            <div className="flex justify-between">
              <span>Batch Size</span>
              <span className="text-[var(--text)]">{scale === 2 ? '16' : '8'}</span>
            </div>
            <div className="flex justify-between">
              <span>Early Stopping</span>
              <span className="text-[var(--text)]">Patience 6 epochs</span>
            </div>
            <div className="flex justify-between">
              <span>Dataset Split</span>
              <span className="text-[var(--text)]">T91 fixed (10% Val Holdout)</span>
            </div>
          </div>
        </div>
      </div>
    </Drawer>
  );
};

