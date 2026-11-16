import React from 'react';
import { Card } from '../ui/Card';

export const FixTimeline: React.FC<{ className?: string }> = ({ className = '' }) => {
  const events = [
    {
      date: 'Step 1: Dataset Audit',
      title: 'Discovered T91 ×2 Off-By-One Misalignment',
      desc: 'Floor division (w // 2) on odd-width images produced 76 misaligned LR/HR image pairs in legacy data.',
      status: 'issue',
    },
    {
      date: 'Step 2: Corrective Pipeline',
      title: 'Non-Destructive Data Regeneration (pairs_fixed)',
      desc: 'All 458 training and benchmark LR pairs regenerated with strict MATLAB-compatible bicubic downsampling.',
      status: 'fix',
    },
    {
      date: 'Step 3: Verification',
      title: '0 Mismatches Verified Across All Sets',
      desc: 'Verified complete coordinate parity across Set5, Set14, BSD100, and T91 prior to retraining E1–E8.',
      status: 'verified',
    },
  ];

  return (
    <Card variant="surface" className={`space-y-4 ${className}`}>
      <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">
        <span>Engineering Timeline</span>
        <span className="font-mono text-[var(--good)]">Data Parity Fix</span>
      </div>

      <div className="relative border-l border-[var(--border)] ml-3 space-y-4 pl-4 font-mono text-xs">
        {events.map((ev, idx) => (
          <div key={idx} className="relative">
            <span
              className={`absolute -left-[21px] top-1 w-2.5 h-2.5 rounded-full border-2 bg-[var(--surface)] ${
                ev.status === 'issue'
                  ? 'border-[var(--bad)]'
                  : ev.status === 'fix'
                  ? 'border-[var(--amber)]'
                  : 'border-[var(--good)]'
              }`}
            />
            <div className="text-[10px] text-[var(--muted)]">{ev.date}</div>
            <div className="text-xs font-semibold text-[var(--text)] mt-0.5">{ev.title}</div>
            <p className="text-[11px] font-sans text-[var(--muted)] mt-1">{ev.desc}</p>
          </div>
        ))}
      </div>
    </Card>
  );
};

