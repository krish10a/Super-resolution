import React from 'react';
import { Card } from '../ui/Card';

export const ProtocolCard: React.FC<{ className?: string }> = ({ className = '' }) => {
  return (
    <Card variant="surface" className={`space-y-4 ${className}`}>
      <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">
        <span>Evaluation Protocol</span>
        <span className="font-mono text-[var(--accent)]">Standard Benchmark Parity</span>
      </div>

      <div className="space-y-3 font-mono text-xs">
        <div className="p-3 rounded bg-[var(--raised)] border border-[var(--border)] space-y-1">
          <div className="text-[var(--text)] font-semibold">1. ITU-R BT.601 Y-Channel Conversion</div>
          <p className="text-[var(--muted)] text-[11px] font-sans">
            Human visual perception is primarily sensitive to luminance. Metrics (PSNR/SSIM) are computed strictly on the Y channel after RGB → YCbCr transformation:
            <code className="block mt-1 p-1 bg-[var(--bg-base)] text-[var(--accent)] rounded font-mono text-[10px]">
              Y = 16 + (65.481·R + 128.553·G + 24.966·B) / 255
            </code>
          </p>
        </div>

        <div className="p-3 rounded bg-[var(--raised)] border border-[var(--border)] space-y-1">
          <div className="text-[var(--text)] font-semibold">2. Border Shaving by Scale</div>
          <p className="text-[var(--muted)] text-[11px] font-sans">
            To eliminate boundary convolution padding artifacts, outer borders are shaved by <code>scale</code> pixels (2px for 2×, 4px for 4×) before calculating PSNR/SSIM.
          </p>
        </div>

        <div className="p-3 rounded bg-[var(--raised)] border border-[var(--border)] space-y-1">
          <div className="text-[var(--text)] font-semibold">3. Standard Evaluation Benchmarks</div>
          <p className="text-[var(--muted)] text-[11px] font-sans">
            Evaluated on Set5 (5 images), Set14 (14 images), and BSD100 (100 images). Perceptual loss measured via AlexNet LPIPS (lower distance = higher perceptual realism).
          </p>
        </div>
      </div>
    </Card>
  );
};

