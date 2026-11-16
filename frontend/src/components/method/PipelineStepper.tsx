import React, { useState } from 'react';
import { Button } from '../ui/Button';
import { Chip } from '../ui/Chip';

export interface PipelineStepperProps {
  sampleImage?: string;
  className?: string;
}

export const PipelineStepper: React.FC<PipelineStepperProps> = ({
  sampleImage = '/samples/surveillance_face.png',
  className = '',
}) => {
  const [currentStep, setCurrentStep] = useState(0);

  const steps = [
    {
      title: 'Step 1: Optical Blur',
      desc: 'Gaussian kernel simulates lens defocus and low CCTV MTF (σ = 1.5).',
      filter: 'blur(3px)',
      style: { filter: 'blur(3px)' },
    },
    {
      title: 'Step 2: Sub-sampling',
      desc: 'Area downsampling drops native resolution by 4×, losing high frequencies.',
      filter: 'none',
      style: { imageRendering: 'pixelated', transform: 'scale(1.0)' },
    },
    {
      title: 'Step 3: Sensor Noise',
      desc: 'Additive Gaussian noise simulates low-light sensor ISO noise (σ = 15).',
      filter: 'contrast(1.4) brightness(0.85) grayscale(0.2)',
      style: { filter: 'contrast(1.4) brightness(0.85) grayscale(0.2)' },
    },
    {
      title: 'Step 4: JPEG Compression',
      desc: 'Discrete Cosine Transform (DCT) 8×8 block artifacts at quality Q = 50.',
      filter: 'contrast(1.2) blur(0.5px)',
      style: { filter: 'contrast(1.2) blur(0.5px)', imageRendering: 'pixelated' },
    },
  ];

  return (
    <div
      className={`p-6 rounded-[var(--radius-card)] border border-[var(--border)] bg-[var(--surface)] space-y-6 ${className}`}
    >
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-base font-heading font-semibold text-[var(--text)]">
            Surveillance Degradation Pipeline
          </h3>
          <p className="text-xs text-[var(--muted)] font-mono mt-0.5">
            Physics-based degradation model: x = (y ⊛ k)↓s + n
          </p>
        </div>
        <div className="flex items-center gap-1.5">
          {steps.map((_, idx) => (
            <button
              key={idx}
              onClick={() => setCurrentStep(idx)}
              className={`w-2.5 h-2.5 rounded-full transition-all ${
                idx === currentStep
                  ? 'bg-[var(--accent)] w-6'
                  : idx < currentStep
                  ? 'bg-[var(--good)]'
                  : 'bg-[var(--border)]'
              }`}
            />
          ))}
        </div>
      </div>

      {/* Stepper Content Stage */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
        {/* Degradation Preview Box */}
        <div className="relative h-60 rounded-[var(--radius-control)] border border-[var(--border)] bg-[var(--bg-base)] overflow-hidden flex items-center justify-center p-4">
          <div className="absolute top-3 left-3 z-10">
            <Chip variant="amber" size="sm">
              {steps[currentStep].title}
            </Chip>
          </div>
          <img
            src={sampleImage}
            alt="Degradation preview"
            className="max-h-full max-w-full object-contain transition-all duration-300"
            style={steps[currentStep].style as any}
          />
        </div>

        {/* Description & Controls */}
        <div className="space-y-4">
          <div className="space-y-2">
            <span className="font-mono text-xs text-[var(--accent)] font-semibold">
              STAGE {currentStep + 1} OF {steps.length}
            </span>
            <h4 className="text-lg font-heading font-bold text-[var(--text)]">
              {steps[currentStep].title}
            </h4>
            <p className="text-sm text-[var(--muted)] leading-relaxed">
              {steps[currentStep].desc}
            </p>
          </div>

          <div className="flex items-center gap-3 pt-2">
            <Button
              variant="secondary"
              size="sm"
              disabled={currentStep === 0}
              onClick={() => setCurrentStep((s) => Math.max(0, s - 1))}
            >
              ← Previous Step
            </Button>
            <Button
              variant="primary"
              size="sm"
              disabled={currentStep === steps.length - 1}
              onClick={() => setCurrentStep((s) => Math.min(steps.length - 1, s + 1))}
            >
              Next Step →
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

