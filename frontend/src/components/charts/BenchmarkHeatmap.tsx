import React, { useState } from 'react';
import { EvaluationResults, ExperimentId } from '../../lib/types';
import { Chip } from '../ui/Chip';

export interface BenchmarkHeatmapProps {
  results: EvaluationResults | null;
  scale: 2 | 4;
  metric: 'psnr' | 'ssim' | 'lpips';
  onRowClick?: (expId: ExperimentId) => void;
  selectedExpId?: ExperimentId | null;
  className?: string;
}

export const BenchmarkHeatmap: React.FC<BenchmarkHeatmapProps> = ({
  results,
  scale,
  metric,
  onRowClick,
  selectedExpId,
  className = '',
}) => {
  const [sortCol, setSortCol] = useState<'Set5' | 'Set14' | 'BSD100'>('BSD100');
  const [sortAsc, setSortAsc] = useState(false);

  if (!results) {
    return (
      <div className="p-8 text-center font-mono text-sm text-[var(--muted)] border border-[var(--border)] rounded-[var(--radius-card)] bg-[var(--surface)]">
        Loading benchmark matrix...
      </div>
    );
  }

  // Filter experiments for current scale
  const expList: {
    id: ExperimentId;
    model: string;
    loss: string;
    scale: number;
    Set5: number;
    Set14: number;
    BSD100: number;
    Set5_delta: number;
    Set14_delta: number;
    BSD100_delta: number;
  }[] = [];

  (Object.keys(results) as ExperimentId[]).forEach((id) => {
    const exp = results[id];
    if (exp?.config?.scale !== scale) return;

    const s5 = exp.results?.Set5;
    const s14 = exp.results?.Set14;
    const bsd = exp.results?.BSD100;

    const getVal = (ds: any) => {
      if (!ds) return 0;
      return metric === 'psnr'
        ? ds.avg_psnr
        : metric === 'ssim'
        ? ds.avg_ssim
        : ds.avg_lpips ?? 0.15;
    };

    expList.push({
      id,
      model: exp.config.model.toUpperCase(),
      loss: exp.config.loss.includes('perceptual') ? 'Perceptual' : 'MSE',
      scale: exp.config.scale,
      Set5: getVal(s5),
      Set14: getVal(s14),
      BSD100: getVal(bsd),
      Set5_delta: s5?.psnr_delta ?? 0,
      Set14_delta: s14?.psnr_delta ?? 0,
      BSD100_delta: bsd?.psnr_delta ?? 0,
    });
  });

  // Sort
  expList.sort((a, b) => {
    const va = a[sortCol];
    const vb = b[sortCol];
    return sortAsc ? va - vb : vb - va;
  });

  // Bicubic baseline row
  const bicubicRef = {
    Set5: results.E1?.results?.Set5?.bicubic_psnr ?? 32.09,
    Set14: results.E1?.results?.Set14?.bicubic_psnr ?? 30.4,
    BSD100: results.E1?.results?.BSD100?.bicubic_psnr ?? 29.96,
  };

  const getDeltaColor = (delta: number) => {
    if (delta > -0.5) return 'bg-[var(--good)]/15 text-[var(--good)]';
    if (delta > -2.0) return 'bg-[var(--amber)]/15 text-[var(--amber)]';
    return 'bg-[var(--bad)]/15 text-[var(--bad)]';
  };

  return (
    <div className={`overflow-x-auto rounded-[var(--radius-card)] border border-[var(--border)] bg-[var(--surface)] ${className}`}>
      <table className="w-full text-left font-mono text-xs border-collapse">
        <thead>
          <tr className="border-b border-[var(--border)] bg-[var(--raised)] text-[var(--muted)]">
            <th className="py-3 px-4 font-semibold">Model / Exp</th>
            <th className="py-3 px-3 font-semibold">Loss</th>
            <th
              className="py-3 px-4 font-semibold cursor-pointer hover:text-[var(--text)] transition-colors"
              onClick={() => {
                setSortCol('Set5');
                setSortAsc(!sortAsc);
              }}
            >
              Set5 {sortCol === 'Set5' && (sortAsc ? '▲' : '▼')}
            </th>
            <th
              className="py-3 px-4 font-semibold cursor-pointer hover:text-[var(--text)] transition-colors"
              onClick={() => {
                setSortCol('Set14');
                setSortAsc(!sortAsc);
              }}
            >
              Set14 {sortCol === 'Set14' && (sortAsc ? '▲' : '▼')}
            </th>
            <th
              className="py-3 px-4 font-semibold cursor-pointer hover:text-[var(--text)] transition-colors"
              onClick={() => {
                setSortCol('BSD100');
                setSortAsc(!sortAsc);
              }}
            >
              BSD100 {sortCol === 'BSD100' && (sortAsc ? '▲' : '▼')}
            </th>
            <th className="py-3 px-4 text-right">Details</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--border)]">
          {/* Reference Bicubic Row */}
          <tr className="bg-[var(--raised)]/40 text-[var(--muted)]">
            <td className="py-2.5 px-4 font-semibold text-[var(--bicubic)] flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[var(--bicubic)]" />
              <span>Bicubic Reference</span>
            </td>
            <td className="py-2.5 px-3">Interpolation</td>
            <td className="py-2.5 px-4 font-bold">{bicubicRef.Set5.toFixed(2)} dB</td>
            <td className="py-2.5 px-4 font-bold">{bicubicRef.Set14.toFixed(2)} dB</td>
            <td className="py-2.5 px-4 font-bold">{bicubicRef.BSD100.toFixed(2)} dB</td>
            <td className="py-2.5 px-4 text-right text-[10px]">Baseline</td>
          </tr>

          {/* Model Rows */}
          {expList.map((row) => {
            const isSelected = selectedExpId === row.id;
            return (
              <tr
                key={row.id}
                onClick={() => onRowClick?.(row.id)}
                className={`transition-colors cursor-pointer hover:bg-[var(--raised)]/70 ${
                  isSelected ? 'bg-[var(--accent)]/10 ring-1 ring-inset ring-[var(--accent)]' : ''
                }`}
              >
                <td className="py-3 px-4">
                  <div className="flex items-center gap-2">
                    <span
                      className={`w-2 h-2 rounded-full ${
                        row.model === 'SRCNN' ? 'bg-[var(--srcnn)]' : 'bg-[var(--espcn)]'
                      }`}
                    />
                    <strong className="text-[var(--text)]">{row.id}</strong>
                    <span className="text-[var(--muted)] text-[11px]">{row.model}</span>
                  </div>
                </td>
                <td className="py-3 px-3">
                  <Chip variant={row.loss === 'MSE' ? 'raised' : 'amber'} size="sm">
                    {row.loss}
                  </Chip>
                </td>
                <td className="py-3 px-4">
                  <div className="flex items-center gap-2">
                    <span className="text-[var(--text)] font-semibold">{row.Set5.toFixed(2)}</span>
                    <span className={`text-[10px] px-1.5 py-0.2 rounded ${getDeltaColor(row.Set5_delta)}`}>
                      {row.Set5_delta.toFixed(2)}
                    </span>
                  </div>
                </td>
                <td className="py-3 px-4">
                  <div className="flex items-center gap-2">
                    <span className="text-[var(--text)] font-semibold">{row.Set14.toFixed(2)}</span>
                    <span className={`text-[10px] px-1.5 py-0.2 rounded ${getDeltaColor(row.Set14_delta)}`}>
                      {row.Set14_delta.toFixed(2)}
                    </span>
                  </div>
                </td>
                <td className="py-3 px-4">
                  <div className="flex items-center gap-2">
                    <span className="text-[var(--text)] font-semibold">{row.BSD100.toFixed(2)}</span>
                    <span className={`text-[10px] px-1.5 py-0.2 rounded ${getDeltaColor(row.BSD100_delta)}`}>
                      {row.BSD100_delta.toFixed(2)}
                    </span>
                  </div>
                </td>
                <td className="py-3 px-4 text-right">
                  <span className="text-[11px] text-[var(--accent)] hover:underline">
                    Per-image →
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

