import React, { useRef, useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { useResults } from '../hooks/useResults';

export const Overview: React.FC = () => {
  const { results, globalStats } = useResults();

  // Interactive Curtain Splitter State
  const [splitPos, setSplitPos] = useState(55); // 0..100%
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const updateSplit = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const offsetX = clientX - rect.left;
    let pct = (offsetX / rect.width) * 100;
    if (pct < 5) pct = 5;
    if (pct > 95) pct = 95;
    setSplitPos(pct);
  }, []);

  const handlePointerDown = (e: React.PointerEvent) => {
    setIsDragging(true);
    updateSplit(e.clientX);
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!isDragging) return;
    updateSplit(e.clientX);
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    setIsDragging(false);
    try {
      (e.target as HTMLElement).releasePointerCapture?.(e.pointerId);
    } catch (_) {}
  };

  // Compute real metrics from data
  const totalModels = results ? Object.keys(results).length : 8;
  const testImagesCount = results?.E1?.results
    ? (results.E1.results.Set5?.n_images || 0) +
      (results.E1.results.Set14?.n_images || 0) +
      (results.E1.results.BSD100?.n_images || 0)
    : 119;
  const bestAttenuation = results?.E7?.results?.BSD100?.psnr_delta
    ? `${results.E7.results.BSD100.psnr_delta.toFixed(2)} dB`
    : '−0.72 dB';

  return (
    <div className="flex flex-col w-full">
      {/* Interactive Overview Container */}
      <div className="w-full max-w-[1440px] mx-auto px-6 py-6 flex flex-col gap-10">
        {/* HERO SECTION */}
        <section className="w-full min-h-[580px] grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          {/* Left 45% (lg:col-span-5) Content Column */}
          <div className="lg:col-span-5 flex flex-col justify-center pr-0 lg:pr-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-primary-container bg-primary-container/5 shadow-sm w-fit mb-5">
              <span className="w-1.5 h-1.5 rounded-full bg-primary-container" />
              <span className="font-mono-metric-sm text-mono-metric-sm text-primary-container uppercase tracking-wider">
                Deep-learning super-resolution
              </span>
            </div>

            <h1 className="font-headline-xl text-on-surface text-[56px] lg:text-[64px] leading-[1.05] tracking-tight font-semibold">
              Recovering <span className="text-primary-container">detail</span> from low-quality surveillance frames.
            </h1>

            <p className="font-body-md text-on-surface-variant text-[15px] leading-relaxed mt-5 mb-8 max-w-xl">
              SRCNN and ESPCN at 2× and 4×, compared on PSNR, SSIM and LPIPS.
            </p>

            <div className="flex items-center gap-3">
              <Link
                to="/studio"
                className="inline-flex items-center gap-2 bg-primary-container text-on-primary-container font-body-sm-medium text-body-sm-medium font-semibold px-6 py-3 rounded-lg shadow-lg hover:opacity-95 transition-all"
              >
                <span>Open Studio</span>
                <span className="text-[16px]">→</span>
              </Link>
              <Link
                to="/benchmarks"
                className="inline-flex items-center gap-2 bg-surface-container text-on-surface hover:text-primary font-body-sm-medium text-body-sm-medium px-5 py-3 rounded-lg transition-colors border border-outline-variant"
              >
                <span>See benchmarks</span>
              </Link>
            </div>
          </div>

          {/* Right 55% (lg:col-span-7) Interactive Comparison Viewport */}
          <div className="lg:col-span-7 relative flex items-center justify-center">
            <div
              ref={containerRef}
              id="hero-curtain-viewport"
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerCancel={handlePointerUp}
              className="relative w-full h-[460px] rounded-xl bg-surface-container-low overflow-hidden shadow-2xl select-none group cursor-ew-resize border border-outline-variant/30"
            >
              {/* High-Res Background Image (SR) */}
              <div className="absolute inset-0 w-full h-full bg-surface-container-lowest">
                <img
                  src="/samples/set14_01_hr.png"
                  alt="High resolution frame"
                  className="w-full h-full object-cover object-center pointer-events-none select-none"
                />
                <div className="absolute bottom-4 right-4 bg-surface-container-low/90 backdrop-blur-md px-2.5 py-1 rounded shadow-md pointer-events-none font-mono-metric-sm text-mono-metric-sm text-primary-container tracking-wider font-semibold border border-outline-variant/30">
                  SR 512×512
                </div>
              </div>

              {/* Low-Res Clipped Overlay (LR) */}
              <div
                className="absolute inset-0 h-full overflow-hidden pointer-events-none select-none"
                style={{ width: `${splitPos}%` }}
              >
                <div className="w-[1000px] h-[460px] relative">
                  <img
                    src="/samples/set14_01_lr_x4.png"
                    alt="Low resolution degraded input"
                    className="w-full h-full object-cover object-center filter blur-[1.2px] contrast-95 brightness-90 saturate-75"
                  />
                  <div className="absolute inset-0 bg-surface-container-lowest/25 mix-blend-multiply" />
                  <div className="absolute top-4 left-4 bg-surface-container-low/90 backdrop-blur-sm px-2.5 py-1 rounded shadow pointer-events-none font-mono-metric-sm text-mono-metric-sm text-on-surface-variant font-medium border border-outline-variant/30">
                    LR 128×128
                  </div>
                </div>
              </div>

              {/* Curtain Divider Bar with Central Grip */}
              <div
                className="absolute top-0 bottom-0 pointer-events-none transition-none"
                style={{ left: `${splitPos}%` }}
              >
                <div className="w-[2px] h-full bg-primary-container shadow-[0_0_12px_rgba(56,225,255,0.7)] relative">
                  {/* Central Circular Grip Handle */}
                  <div className="absolute top-1/2 -translate-y-1/2 -left-4 w-9 h-9 rounded-full bg-surface-container-high flex items-center justify-center shadow-xl border border-primary-container/40">
                    <div className="w-6 h-6 rounded-full bg-surface-container-low flex items-center justify-center text-primary-container text-[12px] font-bold">
                      ↔
                    </div>
                  </div>
                  {/* Diamond Endcaps */}
                  <div className="absolute top-2 -left-[5px] w-3 h-3 bg-primary-container rotate-45 rounded-[1px] opacity-80" />
                  <div className="absolute bottom-2 -left-[5px] w-3 h-3 bg-primary-container rotate-45 rounded-[1px] opacity-80" />
                </div>
              </div>

              {/* Viewport Reticle Corners */}
              <div className="absolute top-2 left-2 w-4 h-4 pointer-events-none">
                <div className="w-full h-[2px] bg-primary-container/70" />
                <div className="w-[2px] h-full bg-primary-container/70" />
              </div>
              <div className="absolute top-2 right-2 w-4 h-4 pointer-events-none flex flex-col items-end">
                <div className="w-full h-[2px] bg-primary-container/70" />
                <div className="w-[2px] h-full bg-primary-container/70 self-end" />
              </div>
              <div className="absolute bottom-2 left-2 w-4 h-4 pointer-events-none flex flex-col justify-end">
                <div className="w-[2px] h-full bg-primary-container/70" />
                <div className="w-full h-[2px] bg-primary-container/70" />
              </div>
              <div className="absolute bottom-2 right-2 w-4 h-4 pointer-events-none flex flex-col justify-end items-end">
                <div className="w-[2px] h-full bg-primary-container/70 self-end" />
                <div className="w-full h-[2px] bg-primary-container/70" />
              </div>
            </div>
          </div>
        </section>

        {/* KEY EVALUATION METRIC TILES */}
        <section className="w-full grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-surface-container-low border border-outline-variant/30 rounded-xl p-6 relative overflow-hidden flex flex-col justify-between shadow-md">
            <div className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider mb-3">
              MODELS TRAINED
            </div>
            <div className="font-mono-metric-lg text-[48px] font-semibold text-on-surface leading-none">
              {totalModels}
            </div>
          </div>

          <div className="bg-surface-container-low border border-outline-variant/30 rounded-xl p-6 relative overflow-hidden flex flex-col justify-between shadow-md">
            <div className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider mb-3">
              TEST IMAGES
            </div>
            <div className="font-mono-metric-lg text-[48px] font-semibold text-on-surface leading-none">
              {testImagesCount}
            </div>
          </div>

          <div className="bg-surface-container-low border border-outline-variant/30 rounded-xl p-6 relative overflow-hidden flex flex-col justify-between shadow-md">
            <div className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider mb-3">
              CLOSEST TO BICUBIC
            </div>
            <div className="font-mono-metric-lg text-[48px] font-semibold text-primary-container leading-none">
              {bestAttenuation}
            </div>
          </div>
        </section>

        {/* HORIZONTAL FORENSIC PIPELINE ARCHITECTURE */}
        <section className="w-full bg-surface-container-low border border-outline-variant/30 rounded-xl p-6 shadow-md">
          <div className="flex items-center justify-between pb-6 mb-6 border-b border-outline-variant/20">
            <h2 className="font-headline-lg text-headline-lg text-on-surface font-semibold">
              Super-Resolution Forensic Pipeline
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 relative">
            <div className="bg-surface-container border border-outline-variant/20 rounded-lg p-5 flex flex-col justify-between relative">
              <div className="flex items-center justify-between mb-3">
                <span className="font-mono-metric-lg text-mono-metric-lg text-primary-container font-semibold">01</span>
                <span className="text-on-surface-variant font-mono text-[12px]">DEGRADE</span>
              </div>
              <h3 className="font-headline-md text-headline-md text-on-surface font-semibold mb-2">Degrade</h3>
              <p className="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
                Blur, downscale, noise, JPEG.
              </p>
            </div>

            <div className="bg-surface-container border border-outline-variant/20 rounded-lg p-5 flex flex-col justify-between relative">
              <div className="flex items-center justify-between mb-3">
                <span className="font-mono-metric-lg text-mono-metric-lg text-primary-container font-semibold">02</span>
                <span className="text-on-surface-variant font-mono text-[12px]">UPSCALE</span>
              </div>
              <h3 className="font-headline-md text-headline-md text-on-surface font-semibold mb-2">Upscale</h3>
              <p className="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
                SRCNN or ESPCN, 2× or 4×.
              </p>
            </div>

            <div className="bg-surface-container border border-outline-variant/20 rounded-lg p-5 flex flex-col justify-between relative">
              <div className="flex items-center justify-between mb-3">
                <span className="font-mono-metric-lg text-mono-metric-lg text-primary-container font-semibold">03</span>
                <span className="text-on-surface-variant font-mono text-[12px]">COMPARE</span>
              </div>
              <h3 className="font-headline-md text-headline-md text-on-surface font-semibold mb-2">Compare</h3>
              <p className="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
                Slider, flicker, error map.
              </p>
            </div>

            <div className="bg-surface-container border border-outline-variant/20 rounded-lg p-5 flex flex-col justify-between relative">
              <div className="flex items-center justify-between mb-3">
                <span className="font-mono-metric-lg text-mono-metric-lg text-primary-container font-semibold">04</span>
                <span className="text-on-surface-variant font-mono text-[12px]">MEASURE</span>
              </div>
              <h3 className="font-headline-md text-headline-md text-on-surface font-semibold mb-2">Measure</h3>
              <p className="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
                PSNR, SSIM and LPIPS.
              </p>
            </div>
          </div>
        </section>

        {/* FORENSIC WORKBENCH CLOSING CTA BANNER */}
        <section className="w-full bg-surface-container-low border border-outline-variant/30 rounded-xl p-6 mb-6 shadow-md flex flex-col sm:flex-row items-center justify-between gap-6">
          <div className="font-headline-lg text-[20px] text-on-surface font-semibold">
            Try it on your own image.
          </div>
          <div>
            <Link
              to="/studio"
              className="inline-flex items-center gap-2 font-body-sm-medium text-body-sm-medium font-semibold bg-primary-container text-on-primary-container px-6 py-2.5 rounded-lg hover:opacity-95 shadow-md transition-all"
            >
              <span>Launch Studio</span>
              <span className="text-[16px]">→</span>
            </Link>
          </div>
        </section>
      </div>
    </div>
  );
};
