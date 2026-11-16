import React, { useRef, useState, useEffect } from 'react';
import { motion } from 'framer-motion';

export interface ArchitectureDiagramProps {
  model: 'SRCNN' | 'ESPCN';
  scale: 2 | 4;
  className?: string;
}

export const ArchitectureDiagram: React.FC<ArchitectureDiagramProps> = ({
  model,
  scale,
  className = '',
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [isInView, setIsInView] = useState(true);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => setIsInView(entry.isIntersecting),
      { threshold: 0.1 }
    );
    if (containerRef.current) observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const isSRCNN = model === 'SRCNN';

  // Real layer specs from models/srcnn.py and models/espcn.py
  const srcnnLayers = [
    { name: 'Input LR', desc: '3 Channels', color: '#94A3B8' },
    { name: 'Bicubic Up', desc: `Pre-upscale (${scale}×)`, color: '#38E1FF' },
    { name: 'Conv1 (9×9)', desc: '64 ch, ReLU (Patch Extr.)', color: '#A78BFA' },
    { name: 'Conv2 (1×1)', desc: '32 ch, ReLU (Non-linear)', color: '#A78BFA' },
    { name: 'Conv3 (5×5)', desc: '3 ch, Tanh (Reconstruct)', color: '#34D399' },
    { name: 'Output HR', desc: 'Enhanced Frame', color: '#38E1FF' },
  ];

  const espcnLayers = [
    { name: 'Input LR', desc: '3 Channels (Raw)', color: '#94A3B8' },
    { name: 'Conv1 (5×5)', desc: '64 ch, ReLU (Feat Map)', color: '#22D3EE' },
    { name: 'Conv2 (3×3)', desc: '64 ch, ReLU (Hidden Rep)', color: '#22D3EE' },
    {
      name: 'Conv3 (3×3)',
      desc: `${3 * (scale ** 2)} ch (${scale}× sub-pixel)`,
      color: '#A78BFA',
    },
    {
      name: 'PixelShuffle',
      desc: `r = ${scale} Sub-pixel upscale`,
      color: '#34D399',
    },
    { name: 'Output HR', desc: 'Enhanced Frame', color: '#38E1FF' },
  ];

  const layers = isSRCNN ? srcnnLayers : espcnLayers;

  return (
    <div
      ref={containerRef}
      className={`p-4 rounded-[var(--radius-card)] border border-[var(--border)] bg-[var(--raised)] space-y-4 ${className}`}
    >
      <div className="flex items-center justify-between text-xs font-mono text-[var(--muted)]">
        <span className="font-semibold text-[var(--text)]">{model} Network Graph</span>
        <span className="text-[11px] text-[var(--accent)]">
          {isSRCNN ? '~20.1k Parameters' : scale === 2 ? '~60.9k Parameters' : '~94.5k Parameters'}
        </span>
      </div>

      {/* Layer Sequence Flow */}
      <div className="relative flex flex-col gap-2.5">
        {layers.map((layer, idx) => (
          <div key={idx} className="relative flex items-center gap-3">
            {/* Step Node */}
            <div
              className="w-8 h-8 rounded-full border-2 flex items-center justify-center font-mono text-xs font-bold shrink-0 bg-[var(--surface)] shadow-md z-10"
              style={{ borderColor: layer.color, color: layer.color }}
            >
              {idx + 1}
            </div>

            {/* Step Card */}
            <div className="flex-1 p-2.5 rounded-[var(--radius-control)] border border-[var(--border)] bg-[var(--surface)] flex items-center justify-between">
              <div>
                <div className="text-xs font-semibold text-[var(--text)]">{layer.name}</div>
                <div className="text-[11px] font-mono text-[var(--muted)]">{layer.desc}</div>
              </div>
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: layer.color }}
              />
            </div>

            {/* Connecting Vertical Track with Moving Particle */}
            {idx < layers.length - 1 && (
              <div className="absolute left-4 top-8 bottom-0 w-0.5 -translate-x-1/2 bg-[var(--border)] h-3 -z-0">
                {isInView && (
                  <motion.div
                    className="w-1.5 h-1.5 rounded-full bg-[var(--accent)] -translate-x-0.5"
                    animate={{ y: [0, 12] }}
                    transition={{
                      duration: 1.2,
                      repeat: Infinity,
                      ease: 'linear',
                      delay: idx * 0.2,
                    }}
                  />
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

