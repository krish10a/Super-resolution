import React from 'react';

export interface DeltaPillProps {
  value: number;
  suffix?: string;
  inverse?: boolean; // If true, lower is better (e.g. LPIPS)
  className?: string;
}

export const DeltaPill: React.FC<DeltaPillProps> = ({
  value,
  suffix = ' dB',
  inverse = false,
  className = '',
}) => {
  const isPositive = value > 0.0001;
  const isZero = Math.abs(value) <= 0.0001;

  // For PSNR/SSIM: positive is good, negative is bad.
  // For LPIPS (inverse): negative is good, positive is bad.
  const isGood = inverse ? value < -0.0001 : isPositive;
  const isBad = inverse ? isPositive : value < -0.0001;

  const colorClasses = isZero
    ? 'bg-[var(--raised)] text-[var(--muted)] border-[var(--border)]'
    : isGood
    ? 'bg-[var(--good)]/10 text-[var(--good)] border-[var(--good)]/30'
    : 'bg-[var(--bad)]/10 text-[var(--bad)] border-[var(--bad)]/30';

  const sign = isPositive ? '+' : '';
  const formatted = `${sign}${value.toFixed(2)}${suffix}`;

  return (
    <span
      className={`inline-flex items-center gap-1 font-mono text-xs font-semibold px-2 py-0.5 rounded-full border ${colorClasses} ${className}`}
    >
      {!isZero && (
        <span className="text-[10px]">
          {isGood ? '↑' : '↓'}
        </span>
      )}
      <span>{formatted}</span>
    </span>
  );
};

