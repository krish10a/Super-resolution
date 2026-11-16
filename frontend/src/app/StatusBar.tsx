import React from 'react';

export interface StatusBarProps {
  device?: string;
  activeCheckpoint?: string;
  lastLatencyMs?: number;
}

export const StatusBar: React.FC<StatusBarProps> = ({
  device = 'CUDA',
  activeCheckpoint = 'espcn_4x_mse.pt',
  lastLatencyMs = 42,
}) => {
  return (
    <footer className="fixed bottom-0 left-0 right-0 z-50 h-7 bg-surface-container-low border-t border-outline-variant px-6 flex items-center font-mono-metric-sm text-mono-metric-sm text-on-surface-variant tracking-wide select-none">
      <div className="flex items-center gap-2">
        <span className="text-primary-container font-medium">{device}</span>
        <span>·</span>
        <span className="truncate max-w-[280px]" title={activeCheckpoint}>
          {activeCheckpoint}
        </span>
        <span>·</span>
        <span className="text-on-surface">{lastLatencyMs} ms</span>
      </div>
    </footer>
  );
};
