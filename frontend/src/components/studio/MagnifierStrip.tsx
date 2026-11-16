import React from 'react';
import { MagnifierState } from '../../hooks/useMagnifier';
import { Chip } from '../ui/Chip';

export interface MagnifierStripProps {
  inputSrc: string;
  bicubicSrc: string;
  modelSrc: string;
  groundTruthSrc?: string | null;
  lensState: MagnifierState;
  onToggleLock: () => void;
  className?: string;
}

export const MagnifierStrip: React.FC<MagnifierStripProps> = ({
  inputSrc,
  bicubicSrc,
  modelSrc,
  groundTruthSrc,
  lensState,
  onToggleLock,
  className = '',
}) => {
  // Zoom calculation: lens is 64x64 over source image, enlarged ~4x in the tile
  const zoomFactor = 4.0;
  const posX = `${lensState.x * 100}%`;
  const posY = `${lensState.y * 100}%`;

  const tiles = [
    {
      label: 'Input (Nearest)',
      src: inputSrc,
      pixelated: true,
      variant: 'muted' as const,
    },
    {
      label: 'Bicubic',
      src: bicubicSrc,
      pixelated: false,
      variant: 'bicubic' as const,
    },
    {
      label: 'Model (SR)',
      src: modelSrc,
      pixelated: false,
      variant: 'accent' as const,
    },
    {
      label: 'Ground Truth',
      src: groundTruthSrc,
      pixelated: false,
      variant: 'good' as const,
      isMissing: !groundTruthSrc,
    },
  ];

  return (
    <div
      className={`h-[176px] bg-[var(--surface)] border-t border-[var(--border)] p-3 flex flex-col justify-between select-none ${className}`}
    >
      {/* Top Header line */}
      <div className="flex items-center justify-between mb-2 px-1">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-[var(--muted)] font-mono">
            Synchronized Magnifier Lens
          </span>
          <button
            type="button"
            onClick={onToggleLock}
            className={`text-[10px] font-mono px-2 py-0.5 rounded border transition-colors ${
              lensState.locked
                ? 'bg-[var(--amber)]/10 text-[var(--amber)] border-[var(--amber)]/30'
                : 'bg-[var(--raised)] text-[var(--muted)] border-[var(--border)] hover:text-[var(--text)]'
            }`}
          >
            {lensState.locked ? '● Lens Locked (Click to Unlock)' : '○ Click Viewer to Lock'}
          </button>
        </div>
        <div className="text-[11px] font-mono text-[var(--muted)]">
          Crop: 64×64 px @ {Math.round(lensState.x * 100)}%, {Math.round(lensState.y * 100)}%
        </div>
      </div>

      {/* Four Equal Magnification Tiles */}
      <div className="grid grid-cols-4 gap-3 flex-1 h-[120px]">
        {tiles.map((t, idx) => (
          <div
            key={idx}
            className="relative h-full rounded-[var(--radius-control)] border border-[var(--border)] bg-[var(--bg-base)] overflow-hidden flex items-center justify-center"
          >
            {/* Top Chip Label */}
            <div className="absolute top-2 left-2 z-10">
              <Chip variant={t.variant} size="sm">
                {t.label}
              </Chip>
            </div>

            {t.isMissing ? (
              <div className="flex flex-col items-center justify-center text-center p-3">
                <span className="font-mono text-xs text-[var(--muted)]">No ground truth</span>
                <span className="text-[10px] text-[var(--muted)]/60 mt-0.5">Custom upload</span>
              </div>
            ) : (
              <div className="w-full h-full relative overflow-hidden">
                <div
                  className="w-full h-full"
                  style={{
                    backgroundImage: `url(${t.src})`,
                    backgroundPosition: `${posX} ${posY}`,
                    backgroundSize: `${zoomFactor * 100}%`,
                    backgroundRepeat: 'no-repeat',
                    imageRendering: t.pixelated ? 'pixelated' : 'auto',
                  }}
                />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

