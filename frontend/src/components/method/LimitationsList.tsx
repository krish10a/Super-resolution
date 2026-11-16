import React from 'react';
import { Card } from '../ui/Card';

export const LimitationsList: React.FC<{ className?: string }> = ({ className = '' }) => {
  const items = [
    {
      title: 'Sub-Bicubic PSNR Convergence',
      desc: 'All 8 trained models remain below the bicubic baseline on PSNR (closest: E5 at −0.24 dB on BSD100).',
    },
    {
      title: '15-Epoch Training Cap',
      desc: 'Training was capped at 15 epochs with early stopping (patience 6), constraining deep asymptotic convergence.',
    },
    {
      title: 'Perception-Distortion Trade-off',
      desc: 'Perceptual loss yields sharper surveillance edges and lower LPIPS at the expense of numeric PSNR/SSIM.',
    },
  ];

  return (
    <Card variant="surface" className={`space-y-4 ${className}`}>
      <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">
        <span>Academic Limitations</span>
        <span className="font-mono text-[var(--muted)]">Honest Disclosure</span>
      </div>

      <div className="space-y-2.5">
        {items.map((item, idx) => (
          <div
            key={idx}
            className="p-3 rounded-[var(--radius-control)] bg-[var(--raised)] border border-[var(--border)] flex items-start gap-2.5"
          >
            <span className="text-[var(--amber)] text-xs shrink-0 mt-0.5 font-mono">#{idx + 1}</span>
            <div>
              <div className="text-xs font-semibold text-[var(--text)]">{item.title}</div>
              <p className="text-[11px] text-[var(--muted)] mt-0.5 leading-snug">{item.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
};

