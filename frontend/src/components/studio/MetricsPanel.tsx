import React from 'react';
import { Metrics, UpscaleResponse, ExperimentId } from '../../lib/types';
import { Card } from '../ui/Card';
import { DeltaPill } from '../ui/DeltaPill';
import { StatNumber } from '../ui/StatNumber';
import { RunInfoCard } from './RunInfoCard';
import { ContextCard } from './ContextCard';

export interface MetricsPanelProps {
  result: UpscaleResponse | null;
  metrics?: Metrics | null;
  bicubicMetrics?: Metrics | null;
  hasGroundTruth: boolean;
  rankInfo?: { rank: number; total: number; deltaToBicubic: number } | null;
  datasetName?: string;
  scale?: number;
  checkpoint?: string;
  className?: string;
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({
  result,
  metrics,
  bicubicMetrics,
  hasGroundTruth,
  rankInfo,
  datasetName = 'Set5',
  scale = 2,
  checkpoint,
  className = '',
}) => {
  const isPsnrBelowBicubic =
    metrics && bicubicMetrics ? metrics.psnr < bicubicMetrics.psnr : false;

  return (
    <aside
      className={`flex flex-col h-full bg-[var(--surface)] border-l border-[var(--border)] overflow-y-auto p-5 space-y-5 select-none ${className}`}
    >
      {/* 1. Fidelity vs Ground Truth Card */}
      <Card variant="surface" className="space-y-4">
        <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">
          <span>Fidelity vs. Ground Truth</span>
          <span className="font-mono text-[10px] text-[var(--accent)]">Y-channel BT.601</span>
        </div>

        {!hasGroundTruth ? (
          <div className="py-6 px-4 rounded-[var(--radius-control)] bg-[var(--bg-base)] border border-[var(--border)] text-center space-y-1.5">
            <div className="text-xs font-mono font-medium text-[var(--muted)]">
              No ground truth — metrics unavailable
            </div>
            <p className="text-[11px] text-[var(--muted)]/70 max-w-[220px] mx-auto">
              Custom uploads lack reference HR frames. Select a Set5/Set14 preset to view PSNR, SSIM & LPIPS.
            </p>
          </div>
        ) : metrics ? (
          <div className="space-y-4">
            {/* PSNR Row */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-xs text-[var(--muted)] font-medium">PSNR (dB)</span>
                <StatNumber
                  value={metrics.psnr}
                  decimals={2}
                  className="text-xl text-[var(--text)]"
                />
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[11px] font-mono text-[var(--muted)]">
                  bicubic {bicubicMetrics ? bicubicMetrics.psnr.toFixed(2) : '—'}
                </span>
                {bicubicMetrics && (
                  <DeltaPill value={metrics.psnr - bicubicMetrics.psnr} suffix=" dB" />
                )}
              </div>
              {/* Mini Two-Bar Comparison */}
              <div className="space-y-1 pt-1">
                <div className="h-1.5 w-full bg-[var(--raised)] rounded-full overflow-hidden flex">
                  <div
                    className="bg-[var(--accent)] h-full transition-all duration-300"
                    style={{ width: `${Math.min(100, (metrics.psnr / 40) * 100)}%` }}
                  />
                </div>
                <div className="h-1 w-full bg-[var(--raised)] rounded-full overflow-hidden flex">
                  <div
                    className="bg-[var(--bicubic)] h-full transition-all duration-300"
                    style={{
                      width: `${
                        bicubicMetrics ? Math.min(100, (bicubicMetrics.psnr / 40) * 100) : 0
                      }%`,
                    }}
                  />
                </div>
              </div>
            </div>

            {/* SSIM Row */}
            <div className="space-y-1.5 pt-2 border-t border-[var(--border)]">
              <div className="flex items-center justify-between">
                <span className="text-xs text-[var(--muted)] font-medium">SSIM (Structural)</span>
                <StatNumber
                  value={metrics.ssim}
                  decimals={4}
                  className="text-xl text-[var(--text)]"
                />
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[11px] font-mono text-[var(--muted)]">
                  bicubic {bicubicMetrics ? bicubicMetrics.ssim.toFixed(4) : '—'}
                </span>
                {bicubicMetrics && (
                  <DeltaPill
                    value={metrics.ssim - bicubicMetrics.ssim}
                    suffix=""
                  />
                )}
              </div>
            </div>

            {/* LPIPS Row (Lower is Better) */}
            {typeof metrics.lpips === 'number' && (
              <div className="space-y-1.5 pt-2 border-t border-[var(--border)]">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs text-[var(--muted)] font-medium">LPIPS (Perceptual)</span>
                    <span className="text-[10px] font-mono text-[var(--muted)]">↓ lower is better</span>
                  </div>
                  <StatNumber
                    value={metrics.lpips}
                    decimals={3}
                    className="text-xl text-[var(--text)]"
                  />
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-[11px] font-mono text-[var(--muted)]">
                    bicubic {bicubicMetrics?.lpips ? bicubicMetrics.lpips.toFixed(3) : '—'}
                  </span>
                  {bicubicMetrics?.lpips && (
                    <DeltaPill
                      value={metrics.lpips - bicubicMetrics.lpips}
                      inverse={true}
                      suffix=""
                    />
                  )}
                </div>
              </div>
            )}

            {/* Calm Amber Note if below bicubic */}
            {isPsnrBelowBicubic && (
              <div className="p-2.5 rounded bg-[var(--amber)]/10 border border-[var(--amber)]/30 text-left">
                <div className="flex items-start gap-2">
                  <span className="text-[var(--amber)] text-xs">ℹ</span>
                  <p className="text-[11px] font-mono text-[var(--amber)] leading-tight">
                    Below bicubic on PSNR for this image.
                  </p>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="py-6 text-center font-mono text-xs text-[var(--muted)]">
            Run enhance to calculate metrics
          </div>
        )}
      </Card>

      {/* 2. Run Telemetry Card */}
      <RunInfoCard
        inputSize={result?.inputSize}
        outputSize={result?.outputSize}
        latencyMs={result?.ms}
        device={result?.device}
        checkpoint={checkpoint || result?.modelId}
      />

      {/* 3. In-Context Benchmark Card */}
      {hasGroundTruth && rankInfo && (
        <ContextCard
          rank={rankInfo.rank}
          total={rankInfo.total}
          dataset={datasetName}
          scale={scale}
          metric="PSNR"
          deltaToBicubic={rankInfo.deltaToBicubic}
        />
      )}
    </aside>
  );
};

