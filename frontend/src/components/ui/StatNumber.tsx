import React, { useEffect, useState } from 'react';

export interface StatNumberProps {
  value: number;
  duration?: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  className?: string;
}

export const StatNumber: React.FC<StatNumberProps> = ({
  value,
  duration = 600,
  decimals = 2,
  prefix = '',
  suffix = '',
  className = '',
}) => {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    // Check prefers-reduced-motion
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setDisplayValue(value);
      return;
    }

    let start = 0;
    const startTime = performance.now();

    const step = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // easeOutExpo: 1 - 2^(-10 * t)
      const ease = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
      const current = start + (value - start) * ease;
      setDisplayValue(current);

      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        setDisplayValue(value);
      }
    };

    requestAnimationFrame(step);
  }, [value, duration]);

  const formatted =
    decimals === 0 ? Math.round(displayValue).toString() : displayValue.toFixed(decimals);

  return (
    <span className={`font-mono font-bold tracking-tight ${className}`}>
      {prefix}
      {formatted}
      {suffix}
    </span>
  );
};

