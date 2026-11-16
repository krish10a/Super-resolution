import React from 'react';
import { ModelName, LossName, Scale, UpscaleResponse } from '../../lib/types';
import { Chip } from '../ui/Chip';

export interface Compare4GridProps {
  scale: Scale;
  results: Record<string, UpscaleResponse>;
  fallbackImage: string;
  onSelectTile: (model: ModelName, loss: LossName) => void;
  className?: string;
}

export const Compare4Grid: React.FC<Compare4GridProps> = ({
  scale,
  results,
  fallbackImage,
  onSelectTile,
  className = '',
}) => {
  const tiles: {
    model: ModelName;
    loss: LossName;
    key: string;
    variant: 'srcnn' | 'espcn';
    title: string;
  }[] = [
    {
      model: 'SRCNN',
      loss: 'MSE',
      key: 'srcnn_mse',
      variant: 'srcnn',
      title: 'SRCNN · MSE',
    },
    {
      model: 'SRCNN',
      loss: 'MSE + Perceptual',
      key: 'srcnn_perc',
      variant: 'srcnn',
      title: 'SRCNN · Perceptual',
    },
    {
      model: 'ESPCN',
      loss: 'MSE',
      key: 'espcn_mse',
      variant: 'espcn',
      title: 'ESPCN · MSE',
    },
    {
      model: 'ESPCN',
      loss: 'MSE + Perceptual',
      key: 'espcn_perc',
      variant: 'espcn',
      title: 'ESPCN · Perceptual',
    },
  ];

  return (
    <div className={`grid grid-cols-2 grid-rows-2 gap-4 h-full p-4 bg-[var(--bg-base)] ${className}`}>
      {tiles.map((t) => {
        const res = results[t.key];
        const imgUrl = res?.srUrl || fallbackImage;
        const metrics = res?.metrics;

        return (
          <div
            key={t.key}
            onClick={() => onSelectTile(t.model, t.loss)}
            className="group relative rounded-[var(--radius-card)] border border-[var(--border)] hover:border-[var(--accent)] bg-[var(--surface)] overflow-hidden cursor-pointer transition-all duration-200 flex flex-col justify-between shadow-md hover:shadow-xl hover:-translate-y-0.5"
          >
            {/* Top Bar with Badge & Scale */}
            <div className="absolute top-3 inset-x-3 z-10 flex items-center justify-between">
              <Chip variant={t.variant} size="sm">
                {t.title}
              </Chip>
              <Chip variant="raised" size="sm">
                {scale}× Scale
              </Chip>
            </div>

            {/* Image Preview */}
            <div className="flex-1 w-full flex items-center justify-center p-2 pt-10">
              <img
                src={imgUrl}
                alt={t.title}
                className="max-h-[30vh] max-w-full object-contain rounded"
              />
            </div>

            {/* Bottom Mini Metrics Readout */}
            <div className="px-4 py-2.5 bg-[var(--raised)] border-t border-[var(--border)] flex items-center justify-between font-mono text-xs">
              <div className="flex items-center gap-3">
                {metrics ? (
                  <>
                    <span>
                      <strong className="text-[var(--text)]">{metrics.psnr}</strong>{' '}
                      <span className="text-[var(--muted)] text-[10px]">dB</span>
                    </span>
                    <span>
                      <strong className="text-[var(--text)]">{metrics.ssim}</strong>{' '}
                      <span className="text-[var(--muted)] text-[10px]">SSIM</span>
                    </span>
                  </>
                ) : (
                  <span className="text-[var(--muted)] text-[11px]">Click to promote</span>
                )}
              </div>
              <span className="text-[10px] text-[var(--accent)] group-hover:underline flex items-center gap-1">
                Promote to Single →
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
};

