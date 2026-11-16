import React from 'react';

export interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  onChange: (val: number) => void;
  disabled?: boolean;
  className?: string;
}

export const Slider: React.FC<SliderProps> = ({
  label,
  value,
  min,
  max,
  step = 1,
  unit = '',
  onChange,
  disabled = false,
  className = '',
}) => {
  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div className={`space-y-1.5 ${disabled ? 'opacity-40 pointer-events-none' : ''} ${className}`}>
      <div className="flex items-center justify-between text-xs">
        <span className="text-[var(--muted)] font-medium">{label}</span>
        <span className="font-mono text-[var(--accent)] font-semibold">
          {value}
          {unit}
        </span>
      </div>
      <div className="relative flex items-center h-5">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="w-full h-1.5 bg-[var(--raised)] border border-[var(--border)] rounded-lg appearance-none cursor-pointer accent-[var(--accent)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
          style={{
            background: `linear-gradient(to right, var(--accent) 0%, var(--accent) ${percentage}%, var(--raised) ${percentage}%, var(--raised) 100%)`,
          }}
        />
      </div>
    </div>
  );
};

