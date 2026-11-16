import React, { useState } from 'react';

interface PipelineStepData {
  step: number;
  name: string;
  desc: string;
  filterClass: string;
}

const pipelineSteps: PipelineStepData[] = [
  {
    step: 1,
    name: 'Blur',
    desc: 'Isotropic Gaussian blur σ = 1.6.',
    filterClass: 'blur-[1.5px]',
  },
  {
    step: 2,
    name: 'Downscale',
    desc: 'Downscale ×4, bicubic.',
    filterClass: 'contrast-95',
  },
  {
    step: 3,
    name: 'Noise',
    desc: 'Sensor noise injection SNR = 28.4 dB.',
    filterClass: 'contrast-125 saturate-50',
  },
  {
    step: 4,
    name: 'JPEG',
    desc: 'DCT block quantization q = 45.',
    filterClass: 'contrast-110 brightness-95',
  },
];

export const Method: React.FC = () => {
  const [currentStep, setCurrentStep] = useState(2);

  const activeStep = pipelineSteps[currentStep - 1];

  const navigatePipeline = (dir: number) => {
    let next = currentStep + dir;
    if (next > 4) next = 1;
    if (next < 1) next = 4;
    setCurrentStep(next);
  };

  return (
    <div className="w-full max-w-[1440px] mx-auto p-6 flex flex-col gap-6 select-none">
      <div className="grid grid-cols-12 gap-6 items-stretch">
        {/* Left Column (7 cols): Degradation pipeline */}
        <section className="col-span-12 lg:col-span-7 bg-[#0E1217] border border-[#1F2832] rounded-xl p-6 flex flex-col justify-between shadow-sm relative overflow-hidden">
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h2 className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">
                Degradation pipeline
              </h2>
            </div>

            {/* Step buttons */}
            <div className="grid grid-cols-4 gap-2">
              {pipelineSteps.map((s) => {
                const isActive = s.step === currentStep;
                return (
                  <button
                    key={s.step}
                    type="button"
                    onClick={() => setCurrentStep(s.step)}
                    className={`px-3 py-2 rounded-lg flex flex-col gap-0.5 text-left transition-all cursor-pointer ${
                      isActive
                        ? 'bg-surface-container-highest shadow-[0_0_0_1px_rgba(56,225,255,0.4)]'
                        : 'bg-surface-container hover:bg-surface-container-high'
                    }`}
                  >
                    <span
                      className={`font-mono-metric-sm text-mono-metric-sm flex items-center gap-1 ${
                        isActive ? 'text-primary-container font-medium' : 'text-on-surface-variant'
                      }`}
                    >
                      {isActive && <span className="w-1 h-1 rounded-full bg-primary-container" />}
                      STEP 0{s.step}
                    </span>
                    <span
                      className={`font-body-sm-medium text-body-sm-medium truncate ${
                        isActive ? 'text-primary font-medium' : 'text-on-surface'
                      }`}
                    >
                      {s.name}
                    </span>
                  </button>
                );
              })}
            </div>

            {/* Viewport Preview */}
            <div className="relative w-full aspect-[4/3] rounded-lg overflow-hidden bg-surface-container-lowest flex items-center justify-center border border-outline-variant/30">
              <img
                src="/samples/set14_01_hr.png"
                alt="Degradation stage view"
                className={`w-full h-full object-cover transition-all duration-300 filter ${activeStep.filterClass}`}
              />
              <div className="absolute top-3 left-3 bg-surface-container-lowest/85 backdrop-blur-md px-2.5 py-1 rounded text-primary-fixed font-mono-metric-sm text-mono-metric-sm flex items-center gap-1.5 shadow-md border border-outline-variant/30">
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary-container" />
                <span>Step {currentStep} / 4</span>
              </div>
            </div>
          </div>

          {/* Stepper Footer */}
          <div className="pt-4 flex items-center justify-between border-t border-outline-variant/20 mt-2">
            <span className="font-mono-metric-sm text-mono-metric-sm text-on-surface-variant">
              {activeStep.desc}
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => navigatePipeline(-1)}
                className="bg-surface-container hover:bg-surface-container-high transition-colors px-3 py-1.5 rounded text-on-surface hover:text-primary font-mono-metric-sm text-mono-metric-sm flex items-center gap-1 cursor-pointer"
              >
                <span>← PREV</span>
              </button>
              <button
                type="button"
                onClick={() => navigatePipeline(1)}
                className="bg-surface-container-high hover:bg-surface-bright transition-colors px-3 py-1.5 rounded text-on-surface hover:text-primary font-mono-metric-sm text-mono-metric-sm flex items-center gap-1 cursor-pointer"
              >
                <span>NEXT →</span>
              </button>
            </div>
          </div>
        </section>

        {/* Right Column (5 cols): Evaluation, Loss, Data fix */}
        <div className="col-span-12 lg:col-span-5 flex flex-col gap-6 justify-between">
          {/* Card 1: Evaluation */}
          <section className="bg-[#0E1217] border border-[#1F2832] rounded-xl p-6 flex flex-col gap-3 shadow-sm">
            <div className="flex items-center justify-between">
              <h2 className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">
                Evaluation
              </h2>
            </div>
            <div className="flex flex-col gap-2">
              <div className="bg-surface-container p-2.5 rounded-lg flex items-center justify-between font-mono-metric-sm text-mono-metric-sm">
                <span className="text-on-surface-variant">Colour space</span>
                <span className="text-on-surface font-medium">ITU-R BT.601 Y</span>
              </div>
              <div className="bg-surface-container p-2.5 rounded-lg flex items-center justify-between font-mono-metric-sm text-mono-metric-sm">
                <span className="text-on-surface-variant">Border shave</span>
                <span className="text-on-surface font-medium">= scale</span>
              </div>
              <div className="bg-surface-container p-2.5 rounded-lg flex items-center justify-between font-mono-metric-sm text-mono-metric-sm">
                <span className="text-on-surface-variant">LPIPS</span>
                <span className="text-on-surface font-medium">AlexNet</span>
              </div>
            </div>
          </section>

          {/* Card 2: Loss */}
          <section className="bg-[#0E1217] border border-[#1F2832] rounded-xl p-6 flex flex-col gap-2 shadow-sm">
            <div className="flex items-center justify-between">
              <h2 className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">
                Loss
              </h2>
            </div>
            <div className="bg-surface-container px-4 py-3 rounded-lg flex items-center justify-between">
              <div className="font-mono-metric-lg text-mono-metric-lg text-on-surface tracking-wide">
                L = MSE + 0.05 · <span className="text-primary-container font-medium">L_perc</span>
              </div>
            </div>
            <p className="font-body-sm text-body-sm text-on-surface-variant pt-1 font-mono-metric-sm">
              3 epochs MSE-only warm-up
            </p>
          </section>

          {/* Card 3: Data fix */}
          <section className="bg-[#0E1217] border border-[#1F2832] rounded-xl p-6 flex flex-col gap-3 shadow-sm">
            <div className="flex items-center justify-between">
              <h2 className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">
                Data fix
              </h2>
            </div>
            <div className="relative pl-6 flex flex-col gap-3">
              <div className="absolute left-2 top-2 bottom-2 w-[1px] bg-outline-variant" />
              <div className="relative flex items-center gap-3">
                <div className="absolute -left-6 w-4 h-4 rounded-full bg-[#0E1217] flex items-center justify-center">
                  <span className="w-2 h-2 rounded-full bg-error" />
                </div>
                <div className="bg-surface-container px-3 py-1.5 rounded-md w-full flex items-center justify-between">
                  <span className="font-mono-metric-sm text-mono-metric-sm text-error">
                    76 misaligned pairs · T91 ×2
                  </span>
                </div>
              </div>
              <div className="relative flex items-center gap-3">
                <div className="absolute -left-6 w-4 h-4 rounded-full bg-[#0E1217] flex items-center justify-center">
                  <span className="w-2 h-2 rounded-full bg-secondary" />
                </div>
                <div className="bg-surface-container px-3 py-1.5 rounded-md w-full flex items-center justify-between">
                  <span className="font-mono-metric-sm text-mono-metric-sm text-on-surface-variant">
                    Regenerated
                  </span>
                </div>
              </div>
              <div className="relative flex items-center gap-3">
                <div className="absolute -left-6 w-4 h-4 rounded-full bg-[#0E1217] flex items-center justify-center">
                  <span className="w-2 h-2 rounded-full bg-primary-fixed-dim" />
                </div>
                <div className="bg-surface-container px-3 py-1.5 rounded-md w-full flex items-center justify-between">
                  <span className="font-mono-metric-sm text-mono-metric-sm text-primary">
                    0 mismatches
                  </span>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>

      {/* Limitations Section */}
      <section className="w-full bg-[#0E1217] border border-[#1F2832] rounded-xl p-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">
            Limitations
          </h2>
        </div>
        <div className="flex flex-col gap-2.5">
          <div className="flex items-center gap-3 font-mono-metric-sm text-mono-metric-sm text-on-surface">
            <span className="w-1.5 h-1.5 rounded-full bg-primary-container shrink-0" />
            <span>Every model scores below bicubic on PSNR.</span>
          </div>
          <div className="flex items-center gap-3 font-mono-metric-sm text-mono-metric-sm text-on-surface">
            <span className="w-1.5 h-1.5 rounded-full bg-primary-container shrink-0" />
            <span>Trained for 15 epochs on T91 only.</span>
          </div>
          <div className="flex items-center gap-3 font-mono-metric-sm text-mono-metric-sm text-on-surface">
            <span className="w-1.5 h-1.5 rounded-full bg-primary-container shrink-0" />
            <span>Perceptual loss trades PSNR for sharper detail.</span>
          </div>
        </div>
      </section>
    </div>
  );
};
