import React from 'react';

export interface ChipProps {
  children: React.ReactNode;
  variant?: 'accent' | 'srcnn' | 'espcn' | 'amber' | 'good' | 'bad' | 'bicubic' | 'muted' | 'raised';
  size?: 'sm' | 'md';
  icon?: React.ReactNode;
  className?: string;
}

export const Chip: React.FC<ChipProps> = ({
  children,
  variant = 'muted',
  size = 'md',
  icon,
  className = '',
}) => {
  const sizeClasses = size === 'sm' ? 'px-2 py-0.5 text-[11px]' : 'px-2.5 py-1 text-xs';

  const variantClasses = {
    accent: 'bg-[var(--accent)]/10 text-[var(--accent)] border-[var(--accent)]/30',
    srcnn: 'bg-[var(--srcnn)]/10 text-[var(--srcnn)] border-[var(--srcnn)]/30',
    espcn: 'bg-[var(--espcn)]/10 text-[var(--espcn)] border-[var(--espcn)]/30',
    amber: 'bg-[var(--amber)]/10 text-[var(--amber)] border-[var(--amber)]/30',
    good: 'bg-[var(--good)]/10 text-[var(--good)] border-[var(--good)]/30',
    bad: 'bg-[var(--bad)]/10 text-[var(--bad)] border-[var(--bad)]/30',
    bicubic: 'bg-[var(--bicubic)]/10 text-[var(--bicubic)] border-[var(--bicubic)]/30',
    muted: 'bg-[var(--raised)] text-[var(--muted)] border-[var(--border)]',
    raised: 'bg-[var(--surface)] text-[var(--text)] border-[var(--border)]',
  }[variant];

  return (
    <span
      className={`inline-flex items-center gap-1.5 font-mono font-medium rounded-full border ${sizeClasses} ${variantClasses} ${className}`}
    >
      {icon && <span className="shrink-0">{icon}</span>}
      <span>{children}</span>
    </span>
  );
};

