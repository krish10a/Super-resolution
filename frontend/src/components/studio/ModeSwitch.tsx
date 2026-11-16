import React from 'react';
import { ViewerMode } from '../../lib/types';
import { Segmented } from '../ui/Segmented';

export interface ModeSwitchProps {
  mode: ViewerMode;
  onChange: (mode: ViewerMode) => void;
  errorMapDisabled?: boolean;
  className?: string;
}

export const ModeSwitch: React.FC<ModeSwitchProps> = ({
  mode,
  onChange,
  errorMapDisabled = false,
  className = '',
}) => {
  return (
    <div
      className={`inline-flex items-center p-1 bg-[var(--surface)]/90 backdrop-blur-md border border-[var(--border)] rounded-full shadow-lg ${className}`}
    >
      <Segmented
        id="viewer-mode-switch"
        size="sm"
        options={[
          { value: 'slider', label: 'Split Slider' },
          { value: 'side-by-side', label: 'Side-by-Side' },
          { value: 'flicker', label: 'Flicker 2Hz' },
          {
            value: 'error-map',
            label: errorMapDisabled ? 'Error Map (GT only)' : 'Error Map',
          },
        ]}
        value={mode}
        onChange={(val) => {
          if (val === 'error-map' && errorMapDisabled) return;
          onChange(val as ViewerMode);
        }}
        className="border-0 bg-transparent p-0"
      />
    </div>
  );
};

