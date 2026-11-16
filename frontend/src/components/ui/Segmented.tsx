import React from 'react';
import { motion } from 'framer-motion';

export interface SegmentedOption<T extends string | number> {
  value: T;
  label: React.ReactNode;
  badge?: React.ReactNode;
}

export interface SegmentedProps<T extends string | number> {
  options: SegmentedOption<T>[];
  value: T;
  onChange: (value: T) => void;
  id?: string;
  size?: 'sm' | 'md';
  className?: string;
}

export function Segmented<T extends string | number>({
  options,
  value,
  onChange,
  id = 'segmented-control',
  size = 'md',
  className = '',
}: SegmentedProps<T>) {
  const sizeClasses = size === 'sm' ? 'p-0.5 text-xs' : 'p-1 text-sm';
  const itemClasses = size === 'sm' ? 'py-1 px-2.5' : 'py-1.5 px-3.5';

  return (
    <div
      role="radiogroup"
      className={`relative inline-flex items-center bg-[var(--raised)] border border-[var(--border)] rounded-[var(--radius-control)] select-none ${sizeClasses} ${className}`}
    >
      {options.map((opt) => {
        const isSelected = opt.value === value;
        return (
          <button
            key={String(opt.value)}
            type="button"
            role="radio"
            aria-checked={isSelected}
            onClick={() => onChange(opt.value)}
            className={`relative z-10 flex items-center justify-center gap-1.5 font-medium transition-colors duration-150 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--accent)] rounded-[6px] ${itemClasses} ${
              isSelected ? 'text-[var(--text)]' : 'text-[var(--muted)] hover:text-[var(--text)]'
            }`}
          >
            {opt.label}
            {opt.badge && <span className="text-[10px] opacity-80">{opt.badge}</span>}

            {isSelected && (
              <motion.div
                layoutId={`segmented-thumb-${id}`}
                className="absolute inset-0 bg-[var(--surface)] border border-[var(--border)] rounded-[6px] shadow-sm -z-10"
                transition={{ type: 'spring', stiffness: 400, damping: 30 }}
              />
            )}
          </button>
        );
      })}
    </div>
  );
}

