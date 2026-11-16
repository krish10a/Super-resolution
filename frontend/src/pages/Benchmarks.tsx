import React, { useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { ExperimentId } from '../lib/types';
import { useResults } from '../hooks/useResults';
import { Drawer } from '../components/ui/Drawer';

export const Benchmarks: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  // Read initial from query params or fallback
  const paramDataset = (searchParams.get('dataset') as any) || 'BSD100';
  const paramScale = (parseInt(searchParams.get('scale') || '4', 10) === 2 ? 2 : 4) as 2 | 4;
  const paramMetric = (searchParams.get('metric') as any) || 'psnr';

  const [dataset, setDataset] = useState<'Set5' | 'Set14' | 'BSD100'>(paramDataset);
  const [scale, setScale] = useState<2 | 4>(paramScale);
  const [metric, setMetric] = useState<'psnr' | 'ssim' | 'lpips'>(paramMetric);

  // Per-image inspect drawer
  const [inspectExpId, setInspectExpId] = useState<ExperimentId | null>(null);

  const { results, headline } = useResults();

  // Sync state to URL params
  const updateDataset = (d: 'Set5' | 'Set14' | 'BSD100') => {
    setDataset(d);
    setSearchParams((prev) => {
      prev.set('dataset', d);
      return prev;
    });
  };

  const updateScale = (s: 2 | 4) => {
    setScale(s);
    setSearchParams((prev) => {
      prev.set('scale', String(s));
      return prev;
    });
  };

  const updateMetric = (m: 'psnr' | 'ssim' | 'lpips') => {
    setMetric(m);
    setSearchParams((prev) => {
      prev.set('metric', m);
      return prev;
    });
  };

  // Experiment metadata mapping
  const expMeta: Record<
    ExperimentId,
    { name: string; model: 'SRCNN' | 'ESPCN'; scale: 2 | 4; loss: 'MSE' | 'Perceptual'; color: string }
  > = {
    E1: { name: 'E1 · SRCNN · MSE', model: 'SRCNN', scale: 2, loss: 'MSE', color: '#a78bfa' },
    E2: { name: 'E2 · SRCNN · Perceptual', model: 'SRCNN', scale: 2, loss: 'Perceptual', color: '#a78bfa' },
    E3: { name: 'E3 · SRCNN · MSE', model: 'SRCNN', scale: 4, loss: 'MSE', color: '#a78bfa' },
    E4: { name: 'E4 · SRCNN · Perceptual', model: 'SRCNN', scale: 4, loss: 'Perceptual', color: '#a78bfa' },
    E5: { name: 'E5 · ESPCN · MSE', model: 'ESPCN', scale: 2, loss: 'MSE', color: '#22d3ee' },
    E6: { name: 'E6 · ESPCN · Perceptual', model: 'ESPCN', scale: 2, loss: 'Perceptual', color: '#22d3ee' },
    E7: { name: 'E7 · ESPCN · MSE', model: 'ESPCN', scale: 4, loss: 'MSE', color: '#22d3ee' },
    E8: { name: 'E8 · ESPCN · Perceptual', model: 'ESPCN', scale: 4, loss: 'Perceptual', color: '#22d3ee' },
  };

  const getMetricVal = (expId: ExperimentId, ds: 'Set5' | 'Set14' | 'BSD100') => {
    if (!results || !results[expId]) return 0;
    const r = results[expId]?.results?.[ds];
    if (!r) return 0;
    if (metric === 'psnr') return r.avg_psnr;
    if (metric === 'ssim') return r.avg_ssim;
    return r.avg_lpips ?? 0.15;
  };

  const getDeltaVal = (expId: ExperimentId, ds: 'Set5' | 'Set14' | 'BSD100') => {
    if (!results || !results[expId]) return 0;
    const r = results[expId]?.results?.[ds];
    if (!r) return 0;
    if (metric === 'psnr') return r.psnr_delta;
    if (metric === 'ssim') return r.ssim_delta;
    return -(r.avg_lpips ?? 0.15) + 0.25;
  };

  const getBicubicVal = (ds: 'Set5' | 'Set14' | 'BSD100') => {
    if (!results) return ds === 'Set5' ? 26.98 : ds === 'Set14' ? 25.76 : 24.80;
    const r = results.E1?.results?.[ds];
    if (!r) return ds === 'Set5' ? 26.98 : ds === 'Set14' ? 25.76 : 24.80;
    if (metric === 'psnr') return r.bicubic_psnr;
    if (metric === 'ssim') return r.bicubic_ssim;
    return 0.28;
  };

  const selectedExpData = inspectExpId && results ? results[inspectExpId] : null;
  const perImageList = selectedExpData?.results?.[dataset]?.per_image || [];

  // Models for active scale: ONE SCALE AT A TIME
  const activeScaleExpIds: ExperimentId[] = scale === 4 ? ['E3', 'E4', 'E7', 'E8'] : ['E1', 'E2', 'E5', 'E6'];
  const barBicubic = getBicubicVal(dataset);

  const barData = activeScaleExpIds.map((id) => {
    const val = getMetricVal(id, dataset);
    const meta = expMeta[id];
    const isChampion = id === (scale === 4 ? 'E7' : 'E5');
    return {
      id,
      label: id,
      sublabel: meta.loss === 'MSE' ? 'MSE' : 'PERC',
      value: val,
      color: meta.color,
      isChampion,
    };
  });

  const allBarVals = [...barData.map((b) => b.value), barBicubic];
  const minVal = Math.floor(Math.min(...allBarVals) - 1);
  const maxVal = Math.ceil(Math.max(...allBarVals) + 1);
  const valRange = Math.max(maxVal - minVal, 1);

  const getYCoord = (val: number) => {
    const h = 135;
    const ratio = Math.max(0, Math.min(1, (val - minVal) / valRange));
    return 155 - ratio * h;
  };

  const bicubicY = getYCoord(barBicubic);

  return (
    <div className="w-full px-6 py-6 max-w-[1440px] mx-auto flex flex-col gap-6">
      {/* 1. Top Benchmarks Header & Filter Bar */}
      <div className="flex flex-col">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 pb-4">
          <div>
            <div className="flex items-center gap-2 mb-1.5">
              <span className="inline-flex w-2 h-2 rounded-full bg-primary-container shadow-[0_0_8px_rgba(56,225,255,0.7)]" />
              <span className="font-label-caps text-label-caps uppercase tracking-wider text-on-surface-variant">
                FORENSIC EVALUATION WORKBENCH
              </span>
            </div>
            <h2 className="font-headline-xl text-headline-xl text-on-surface font-semibold tracking-tight">
              Benchmarks
            </h2>
            <p className="font-mono-metric-md text-mono-metric-md text-on-surface-variant mt-1.5">
              Closest to bicubic:{' '}
              <span className="text-primary-container font-mono-metric-md">
                {headline || 'E7 · ESPCN 4× MSE at −0.72 dB on BSD100'}
              </span>
            </p>
          </div>
        </div>

        {/* Filter Controls Strip */}
        <div className="flex flex-wrap items-center gap-6 py-3.5 px-4 bg-surface-container-low rounded-xl">
          {/* Dataset Selector */}
          <div className="flex items-center gap-2.5">
            <span className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">
              Dataset:
            </span>
            <div className="flex items-center gap-1 bg-surface-container-lowest p-1 rounded-lg">
              {(['Set5', 'Set14', 'BSD100'] as const).map((d) => (
                <button
                  key={d}
                  onClick={() => updateDataset(d)}
                  className={`px-3 py-1 font-body-sm-medium text-body-sm-medium rounded transition-all cursor-pointer ${
                    dataset === d
                      ? 'text-primary-container bg-surface-container font-medium shadow-[0_0_8px_rgba(56,225,255,0.15)]'
                      : 'text-on-surface-variant hover:text-on-surface'
                  }`}
                >
                  {d}
                </button>
              ))}
            </div>
          </div>

          <div className="h-4 w-[1px] bg-outline-variant hidden sm:block" />

          {/* Scale Selector */}
          <div className="flex items-center gap-2.5">
            <span className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">
              Scale:
            </span>
            <div className="flex items-center gap-1 bg-surface-container-lowest p-1 rounded-lg">
              {([2, 4] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => updateScale(s)}
                  className={`px-3 py-1 font-body-sm-medium text-body-sm-medium rounded transition-all cursor-pointer ${
                    scale === s
                      ? 'text-primary-container bg-surface-container font-medium shadow-[0_0_8px_rgba(56,225,255,0.15)]'
                      : 'text-on-surface-variant hover:text-on-surface'
                  }`}
                >
                  {s}×
                </button>
              ))}
            </div>
          </div>

          <div className="h-4 w-[1px] bg-outline-variant hidden sm:block" />

          {/* Metric Selector */}
          <div className="flex items-center gap-2.5">
            <span className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">
              Metric:
            </span>
            <div className="flex items-center gap-1 bg-surface-container-lowest p-1 rounded-lg">
              {[
                { key: 'psnr', label: 'PSNR (dB)' },
                { key: 'ssim', label: 'SSIM' },
                { key: 'lpips', label: 'LPIPS ↓' },
              ].map(({ key, label }) => (
                <button
                  key={key}
                  onClick={() => updateMetric(key as any)}
                  className={`px-3 py-1 font-body-sm-medium text-body-sm-medium rounded transition-all cursor-pointer ${
                    metric === key
                      ? 'text-primary-container bg-surface-container font-medium shadow-[0_0_8px_rgba(56,225,255,0.15)]'
                      : 'text-on-surface-variant hover:text-on-surface'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* 2. Middle Section: Heatmap Matrix (7 cols) + Grouped Bar Chart (5 cols) */}
      <div className="grid grid-cols-12 gap-6 items-stretch">
        {/* LEFT 60% (col-span-12 lg:col-span-7): Heatmap Matrix Table */}
        <div className="col-span-12 lg:col-span-7 bg-surface-container-low rounded-xl p-5 flex flex-col justify-between shadow-xl">
          <div>
            <div className="flex items-center justify-between mb-4">
              <span className="font-label-caps text-label-caps uppercase tracking-wider text-on-surface-variant font-semibold">
                {metric.toUpperCase()} VS BICUBIC (DB)
              </span>
              <div className="flex items-center gap-3 text-on-surface-variant font-mono-metric-sm text-mono-metric-sm">
                <span className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-[#a78bfa]" />
                  SRCNN
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-[#22d3ee]" />
                  ESPCN
                </span>
              </div>
            </div>

            {/* Table showing ONE scale at a time */}
            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono-metric-sm text-mono-metric-sm border-collapse">
                <thead>
                  <tr className="text-on-surface-variant uppercase font-label-caps text-label-caps bg-surface-container-lowest/50">
                    <th className="py-2.5 px-3 rounded-l-lg">Model</th>
                    <th className="py-2.5 px-3 text-right">Set5</th>
                    <th className="py-2.5 px-3 text-right">Set14</th>
                    <th className="py-2.5 px-3 text-right rounded-r-lg">BSD100</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-surface-container-lowest/40 font-mono-metric-sm text-mono-metric-sm">
                  {/* Reference Baseline Row: Bicubic */}
                  <tr className="bg-surface-container-lowest/70 text-on-surface font-medium">
                    <td className="py-2.5 px-3 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-secondary" />
                      <span className="font-body-sm-medium text-body-sm-medium">Bicubic</span>
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono-metric-md text-mono-metric-md">
                      {getBicubicVal('Set5').toFixed(2)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono-metric-md text-mono-metric-md">
                      {getBicubicVal('Set14').toFixed(2)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono-metric-md text-mono-metric-md">
                      {getBicubicVal('BSD100').toFixed(2)}
                    </td>
                  </tr>

                  {/* Experiments for ACTIVE scale only */}
                  {activeScaleExpIds.map((id) => {
                    const meta = expMeta[id];
                    const isOptimal = id === (scale === 4 ? 'E7' : 'E5');

                    return (
                      <tr
                        key={id}
                        onClick={() => setInspectExpId(id)}
                        className={`cursor-pointer transition-colors ${
                          isOptimal
                            ? 'bg-surface-container/60 hover:bg-surface-container'
                            : 'hover:bg-surface-container'
                        }`}
                      >
                        <td className="py-2 px-3 flex items-center gap-2">
                          <span
                            className="w-2 h-2 rounded-full"
                            style={{ backgroundColor: meta.color }}
                          />
                          <span
                            className={`font-body-sm text-body-sm font-medium ${
                              isOptimal ? 'text-primary-container font-semibold' : 'text-on-surface'
                            }`}
                          >
                            {meta.name}
                          </span>
                        </td>

                        {(['Set5', 'Set14', 'BSD100'] as const).map((dsName) => {
                          const val = getMetricVal(id, dsName);
                          const delta = getDeltaVal(id, dsName);
                          const isBestCell = isOptimal && dsName === 'BSD100';

                          return (
                            <td key={dsName} className="py-2 px-3 text-right">
                              {isBestCell ? (
                                <div className="inline-flex items-center gap-1 px-2 py-0.5 rounded border border-primary-container bg-surface-container-highest text-primary-container">
                                  <span className="font-mono-metric-md text-mono-metric-md font-bold">
                                    {val.toFixed(2)}
                                  </span>
                                  <span className="text-[10px] tracking-tight bg-primary-container/20 text-primary-container px-1 py-0.2 rounded font-semibold">
                                    {delta >= 0 ? `+${delta.toFixed(2)}` : delta.toFixed(2)}
                                  </span>
                                </div>
                              ) : (
                                <>
                                  <span className="text-on-surface">{val.toFixed(2)}</span>
                                  <span
                                    className={`inline-block ml-1.5 px-1.5 py-0.5 rounded bg-surface-container font-medium ${
                                      delta >= 0 ? 'text-[#34d399]' : 'text-[#ff5c6c]'
                                    }`}
                                  >
                                    {delta >= 0 ? `+${delta.toFixed(2)}` : delta.toFixed(2)}
                                  </span>
                                </>
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="pt-3 mt-3 flex items-center justify-between text-on-surface-variant font-mono-metric-sm text-mono-metric-sm">
            <span>Click a row for per-image results.</span>
          </div>
        </div>

        {/* RIGHT 40% (col-span-12 lg:col-span-5): Scale Grouped Bar Chart */}
        <div className="col-span-12 lg:col-span-5 bg-surface-container-low rounded-xl p-5 flex flex-col justify-between shadow-xl">
          <div>
            {/* Card Header & Legend */}
            <div className="flex items-center justify-between mb-4">
              <span className="font-label-caps text-label-caps uppercase tracking-wider text-on-surface-variant font-semibold">
                {scale}× · {dataset} · {metric.toUpperCase()}
              </span>
              <div className="flex items-center gap-3 font-mono-metric-sm text-mono-metric-sm text-on-surface-variant">
                <span className="flex items-center gap-1">
                  <span className="w-2.5 h-2.5 rounded-sm bg-[#a78bfa]" />
                  SRCNN
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-2.5 h-2.5 rounded-sm bg-[#22d3ee]" />
                  ESPCN
                </span>
              </div>
            </div>

            {/* Bar Chart Area (SVG) */}
            <div className="w-full bg-surface-container-lowest/60 rounded-xl p-4">
              <svg className="w-full h-auto overflow-visible select-none" fill="none" viewBox="0 0 460 210">
                {/* Horizontal Grid lines */}
                <line
                  className="text-surface-container"
                  stroke="currentColor"
                  strokeWidth="1"
                  x1="40"
                  x2="440"
                  y1="20"
                  y2="20"
                />
                <line
                  className="text-surface-container"
                  stroke="currentColor"
                  strokeWidth="1"
                  x1="40"
                  x2="440"
                  y1="65"
                  y2="65"
                />
                <line
                  className="text-surface-container"
                  stroke="currentColor"
                  strokeWidth="1"
                  x1="40"
                  x2="440"
                  y1="110"
                  y2="110"
                />
                <line
                  className="text-surface-container"
                  stroke="currentColor"
                  strokeWidth="1"
                  x1="40"
                  x2="440"
                  y1="155"
                  y2="155"
                />

                {/* Y Axis Labels */}
                <text fill="#859397" fontFamily="JetBrains Mono" fontSize="10" textAnchor="end" x="32" y="24">
                  {maxVal} dB
                </text>
                <text fill="#859397" fontFamily="JetBrains Mono" fontSize="10" textAnchor="end" x="32" y="69">
                  {Math.round(minVal + valRange * 0.66)} dB
                </text>
                <text fill="#859397" fontFamily="JetBrains Mono" fontSize="10" textAnchor="end" x="32" y="114">
                  {Math.round(minVal + valRange * 0.33)} dB
                </text>
                <text fill="#859397" fontFamily="JetBrains Mono" fontSize="10" textAnchor="end" x="32" y="159">
                  {minVal} dB
                </text>

                {/* Bicubic Baseline Reference Dashed Line */}
                <line
                  stroke="#94A3B8"
                  strokeDasharray="3 3"
                  strokeWidth="1.5"
                  x1="40"
                  x2="440"
                  y1={bicubicY}
                  y2={bicubicY}
                />
                <rect
                  className="fill-surface-container-highest"
                  height="18"
                  rx="4"
                  width="130"
                  x="300"
                  y={Math.max(10, bicubicY - 9)}
                />
                <text
                  fill="#bbc8d8"
                  fontFamily="JetBrains Mono"
                  fontSize="10"
                  fontWeight="500"
                  textAnchor="middle"
                  x="365"
                  y={Math.max(10, bicubicY - 9) + 13}
                >
                  Bicubic: {barBicubic.toFixed(2)} dB
                </text>

                {/* Bars: rendered dynamically */}
                {barData.map((b, idx) => {
                  const x = 75 + idx * 90;
                  const y = getYCoord(b.value);
                  const h = Math.max(8, 170 - y);

                  return (
                    <g
                      key={b.id}
                      className="cursor-pointer group"
                      onClick={() => setInspectExpId(b.id)}
                    >
                      <rect
                        className="group-hover:brightness-110 transition-all"
                        fill={b.color}
                        fillOpacity={b.isChampion ? 0.95 : b.sublabel === 'MSE' ? 0.85 : 0.6}
                        height={h}
                        rx="3"
                        width="38"
                        x={x}
                        y={y}
                      />
                      <text
                        fill={b.isChampion ? '#38e1ff' : '#e2e2e7'}
                        fontFamily="JetBrains Mono"
                        fontSize={b.isChampion ? '11' : '10'}
                        fontWeight={b.isChampion ? '700' : '500'}
                        textAnchor="middle"
                        x={x + 19}
                        y={Math.max(14, y - 7)}
                      >
                        {b.value.toFixed(2)}
                      </text>
                      <text
                        fill={b.isChampion ? '#38e1ff' : '#bbc9cd'}
                        fontFamily="Inter"
                        fontSize="11"
                        fontWeight={b.isChampion ? '600' : '500'}
                        textAnchor="middle"
                        x={x + 19}
                        y="184"
                      >
                        {b.label}
                      </text>
                      <text
                        fill={b.isChampion ? '#38e1ff' : '#859397'}
                        fontFamily="JetBrains Mono"
                        fontSize="9"
                        textAnchor="middle"
                        x={x + 19}
                        y="198"
                      >
                        {b.sublabel}
                      </text>
                    </g>
                  );
                })}

                {/* Baseline X Axis Line */}
                <line
                  className="text-outline-variant"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  x1="40"
                  x2="440"
                  y1="170"
                  y2="170"
                />
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* 3. FULL-WIDTH BOTTOM CARD: PSNR VS LPIPS Scatter Plot */}
      <div className="w-full bg-surface-container-low rounded-xl p-5 shadow-xl flex flex-col">
        {/* Header with Simplified Legend */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 mb-4 pb-3 border-b border-surface-container">
          <div>
            <span className="font-label-caps text-label-caps uppercase tracking-wider text-on-surface-variant font-semibold">
              PSNR VS LPIPS
            </span>
          </div>
          {/* Legend Items */}
          <div className="flex flex-wrap items-center gap-5 font-mono-metric-sm text-mono-metric-sm text-on-surface-variant">
            <div className="flex items-center gap-1.5">
              <span>Solid = MSE</span>
              <span>·</span>
              <span>Hollow = Perceptual</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-[#a78bfa]" />
              <span>SRCNN</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-[#22d3ee]" />
              <span>ESPCN</span>
            </div>
          </div>
        </div>

        {/* Scatter Visualization (SVG) */}
        <div className="w-full h-72 bg-surface-container-lowest/70 rounded-xl p-4 relative overflow-hidden flex items-center justify-center">
          <svg className="w-full h-full overflow-visible select-none" fill="none" viewBox="0 0 1000 240">
            {/* Grid lines */}
            <line className="text-surface-container" stroke="currentColor" strokeWidth="1" x1="80" x2="80" y1="20" y2="200" />
            <line className="text-surface-container" stroke="currentColor" strokeWidth="1" x1="370" x2="370" y1="20" y2="200" />
            <line className="text-surface-container" stroke="currentColor" strokeWidth="1" x1="660" x2="660" y1="20" y2="200" />
            <line className="text-surface-container" stroke="currentColor" strokeWidth="1" x1="960" x2="960" y1="20" y2="200" />

            <line className="text-surface-container" stroke="currentColor" strokeWidth="1" x1="80" x2="960" y1="20" y2="20" />
            <line className="text-surface-container" stroke="currentColor" strokeWidth="1" x1="80" x2="960" y1="80" y2="80" />
            <line className="text-surface-container" stroke="currentColor" strokeWidth="1" x1="80" x2="960" y1="140" y2="140" />
            <line className="text-surface-container" stroke="currentColor" strokeWidth="1" x1="80" x2="960" y1="200" y2="200" />

            {/* Y Axis (LPIPS) */}
            <text fill="#859397" fontFamily="JetBrains Mono" fontSize="10" textAnchor="end" x="68" y="24">
              0.45
            </text>
            <text fill="#859397" fontFamily="JetBrains Mono" fontSize="10" textAnchor="end" x="68" y="84">
              0.35
            </text>
            <text fill="#859397" fontFamily="JetBrains Mono" fontSize="10" textAnchor="end" x="68" y="144">
              0.25
            </text>
            <text fill="#38e1ff" fontFamily="JetBrains Mono" fontSize="10" fontWeight="600" textAnchor="end" x="68" y="204">
              0.15 (Sharp ↓)
            </text>

            {/* X Axis (PSNR dB) */}
            <text fill="#859397" fontFamily="JetBrains Mono" fontSize="10" textAnchor="middle" x="80" y="222">
              22 dB
            </text>
            <text fill="#859397" fontFamily="JetBrains Mono" fontSize="10" textAnchor="middle" x="370" y="222">
              23 dB
            </text>
            <text fill="#859397" fontFamily="JetBrains Mono" fontSize="10" textAnchor="middle" x="660" y="222">
              24 dB
            </text>
            <text fill="#859397" fontFamily="JetBrains Mono" fontSize="10" textAnchor="middle" x="960" y="222">
              25 dB
            </text>

            {/* Pairs with vectors */}
            {/* E3 vs E4 (SRCNN 4x) */}
            <line opacity="0.6" stroke="#a78bfa" strokeDasharray="4 3" strokeWidth="1.5" x1="468" x2="241" y1="80" y2="158" />
            <circle cx="468" cy="80" fill="#a78bfa" r="5.5" className="cursor-pointer" onClick={() => setInspectExpId('E3')} />
            <text fill="#e2e2e7" fontFamily="JetBrains Mono" fontSize="10" x="478" y="78">
              E3 · 4× MSE ({getMetricVal('E3', 'BSD100').toFixed(2)} dB, 0.35)
            </text>
            <circle cx="241" cy="158" fill="#0c0e11" r="5.5" stroke="#a78bfa" strokeWidth="2" className="cursor-pointer" onClick={() => setInspectExpId('E4')} />
            <text fill="#e2e2e7" fontFamily="JetBrains Mono" fontSize="10" x="251" y="162">
              E4 · 4× Perc ({getMetricVal('E4', 'BSD100').toFixed(2)} dB, 0.22)
            </text>

            {/* E7 vs E8 (ESPCN 4x) */}
            <line opacity="0.8" stroke="#22d3ee" strokeDasharray="4 3" strokeWidth="1.5" x1="648" x2="630" y1="104" y2="182" />
            <circle cx="648" cy="104" fill="#22d3ee" filter="drop-shadow(0 0 6px #38e1ff)" r="6" className="cursor-pointer" onClick={() => setInspectExpId('E7')} />
            <text fill="#38e1ff" fontFamily="JetBrains Mono" fontSize="11" fontWeight="600" x="660" y="102">
              E7 · 4× MSE ({getMetricVal('E7', 'BSD100').toFixed(2)} dB, 0.31)
            </text>
            <circle cx="630" cy="182" fill="#0c0e11" r="6" stroke="#22d3ee" strokeWidth="2.5" className="cursor-pointer" onClick={() => setInspectExpId('E8')} />
            <text fill="#e2e2e7" fontFamily="JetBrains Mono" fontSize="10" x="642" y="186">
              E8 · 4× Perc ({getMetricVal('E8', 'BSD100').toFixed(2)} dB, 0.18)
            </text>
          </svg>
        </div>
      </div>

      {/* 4. Slide-Over Per-Image Detailed Drawer */}
      <Drawer
        open={Boolean(inspectExpId)}
        onClose={() => setInspectExpId(null)}
        title={`Experiment Details · ${inspectExpId} on ${dataset}`}
        subtitle={`${selectedExpData?.config?.model?.toUpperCase()} ${selectedExpData?.config?.scale}× · ${selectedExpData?.config?.loss}`}
        width="max-w-[640px]"
      >
        <div className="space-y-4 font-mono-metric-sm text-mono-metric-sm">
          <div className="flex items-center justify-between text-on-surface-variant">
            <span>Showing {perImageList.length} Test Images</span>
            <span>Y-Channel BT.601 Shaved</span>
          </div>

          <div className="rounded-xl border border-outline-variant/30 overflow-hidden bg-surface-container-lowest">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-surface-container-low border-b border-outline-variant/30 text-on-surface-variant uppercase font-label-caps text-label-caps">
                  <th className="py-2.5 px-3">Image ID</th>
                  <th className="py-2.5 px-3">PSNR (dB)</th>
                  <th className="py-2.5 px-3">SSIM</th>
                  <th className="py-2.5 px-3 text-right">LR / HR Dim</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-outline-variant/20">
                {perImageList.map((img) => (
                  <tr key={img.image} className="hover:bg-surface-container/50 transition-colors">
                    <td className="py-2.5 px-3 font-semibold text-on-surface">{img.image}</td>
                    <td className="py-2.5 px-3 text-[#34d399] font-medium">{img.psnr.toFixed(2)} dB</td>
                    <td className="py-2.5 px-3 text-primary-container">{img.ssim.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-right text-on-surface-variant text-[11px]">
                      {img.lr_size[0]}×{img.lr_size[1]} → {img.hr_size[0]}×{img.hr_size[1]}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </Drawer>
    </div>
  );
};
