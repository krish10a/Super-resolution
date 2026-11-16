import React from 'react';
import { Card } from '../ui/Card';

export const LossFormulaCard: React.FC<{ className?: string }> = ({ className = '' }) => {
  return (
    <Card variant="surface" className={`space-y-4 ${className}`}>
      <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">
        <span>Loss Formulation</span>
        <span className="font-mono text-[var(--amber)]">Hybrid Supervision</span>
      </div>

      <div className="p-4 rounded-[var(--radius-control)] bg-[var(--bg-base)] border border-[var(--border)] font-mono text-center space-y-2">
        <div className="text-sm font-bold text-[var(--accent)]">
          ℒ_total = ℒ_MSE + λ · ℒ_perc
        </div>
        <div className="text-[11px] text-[var(--muted)]">
          where λ = 0.05, ℒ_perc = ‖ φ_j(I_SR) − φ_j(I_HR) ‖₂²
        </div>
      </div>

      <div className="space-y-2 font-mono text-xs text-[var(--muted)]">
        <div className="p-2.5 rounded bg-[var(--raised)] border border-[var(--border)] flex items-start gap-2">
          <span className="text-[var(--good)]">✓</span>
          <p className="text-[11px] font-sans">
            <strong>VGG-11 Feature Backbone:</strong> Features extracted from frozen layers <code>relu2_2</code> and <code>relu3_3</code> to guide high-frequency edge reconstruction.
          </p>
        </div>

        <div className="p-2.5 rounded bg-[var(--raised)] border border-[var(--border)] flex items-start gap-2">
          <span className="text-[var(--amber)]">ℹ</span>
          <p className="text-[11px] font-sans">
            <strong>3-Epoch MSE Warmup Curriculum:</strong> Models train with pure MSE (λ=0) for epochs 1–3 before introducing perceptual gradients, preventing early feature collapse.
          </p>
        </div>
      </div>
    </Card>
  );
};

