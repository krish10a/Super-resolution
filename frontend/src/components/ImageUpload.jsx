/**
 * Drag-and-drop + file picker image upload with preview and image metadata.
 * Emits a File object via onFile(file, imgDimensions) when a valid image is chosen.
 */
import React, { useRef, useState, useEffect } from 'react';
import { Upload, X, AlertCircle, Info } from 'lucide-react';

const ACCEPT = '.jpg,.jpeg,.png,.webp';
const MAX_MB = 10;

export default function ImageUpload({ image, onFile, onRemove, disabled }) {
  const [drag, setDrag] = useState(false);
  const [err, setErr] = useState(null);
  const [dimensions, setDimensions] = useState(null);
  const inputRef = useRef(null);

  const extOK = (name) =>
    ACCEPT.split(',').some((e) => name.toLowerCase().endsWith(e.trim()));

  useEffect(() => {
    if (image?.preview) {
      const img = new Image();
      img.onload = () => {
        setDimensions({ w: img.naturalWidth, h: img.naturalHeight });
      };
      img.src = image.preview;
    } else {
      setDimensions(null);
    }
  }, [image]);

  const onDrop = (e) => {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files[0];
    if (!f) return;
    handleFile(f);
  };

  const handleFile = (file) => {
    setErr(null);
    if (!extOK(file.name)) {
      setErr(`Unsupported type “${file.name.split('.').pop()}”. Allowed: JPG, JPEG, PNG, WEBP.`);
      return;
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      setErr(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max ${MAX_MB} MB.`);
      return;
    }
    onFile(file);
  };

  const onFileInput = (e) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const pickFile = () => {
    if (disabled) return;
    inputRef.current?.click();
  };

  const isHighRes = dimensions && (dimensions.w > 300 || dimensions.h > 300);

  return (
    <div
      className={`rounded-2xl border-2 border-dashed transition-all duration-200 p-6 ${
        drag
          ? 'bg-blue-50/50 border-blue-500'
          : 'bg-white border-slate-200 hover:border-slate-300'
      }`}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={onDrop}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && pickFile()}
      onClick={pickFile}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        onChange={onFileInput}
        disabled={disabled}
        className="sr-only"
      />
      <div className="flex flex-col items-center justify-center">
        {image ? (
          <>
            <div className="relative group max-h-48 max-w-full">
              <img
                src={image.preview}
                alt="Preview"
                className="max-h-48 max-w-full rounded-xl border border-slate-200 shadow-xs object-contain bg-slate-50"
              />
              <button
                onClick={(e) => { e.stopPropagation(); onRemove(null); }}
                className="absolute top-2 right-2 w-7 h-7 rounded-full bg-black/60 hover:bg-red-600 text-white flex items-center justify-center transition-colors shadow-sm"
                title="Remove image"
              >
                <X size={15} />
              </button>
            </div>

            <div className="mt-3 flex items-center flex-wrap justify-center gap-2 text-xs">
              <span className="text-slate-700 font-medium truncate max-w-xs" title={image.file.name}>
                {image.file.name}
              </span>
              <span className="text-slate-400 bg-slate-100 px-2 py-0.5 rounded-full font-mono text-[11px]">
                {(image.file.size / 1024).toFixed(1)} KB
              </span>
              {dimensions && (
                <span className={`px-2 py-0.5 rounded-full font-mono text-[11px] font-semibold ${
                  isHighRes ? 'bg-amber-100 text-amber-800' : 'bg-emerald-100 text-emerald-800'
                }`}>
                  {dimensions.w} × {dimensions.h} px {isHighRes ? '(Full-Res)' : '(Low-Res CCTV)'}
                </span>
              )}
            </div>

            {isHighRes && (
              <div className="mt-3 p-2.5 rounded-xl bg-blue-50/70 border border-blue-100 text-[11px] text-blue-800 flex items-center gap-2 max-w-xs text-left">
                <Info size={14} className="shrink-0 text-blue-600" />
                <span>
                  High-res image detected. Surveillance Simulation mode will downscale to CCTV resolution so enhancement is visibly noticeable!
                </span>
              </div>
            )}

            {err && <AlertError>{err}</AlertError>}
          </>
        ) : (
          <>
            <div className="w-12 h-12 rounded-2xl bg-blue-50 text-blue-600 flex items-center justify-center mb-3 shadow-xs">
              <Upload size={22} />
            </div>
            <p className="text-sm font-semibold text-slate-800">Drag & drop your custom image here</p>
            <p className="text-xs text-slate-500 mt-1">or click to browse from device</p>
            <p className="text-[11px] text-slate-400 mt-2">Any resolution JPG, PNG, WEBP supported</p>
          </>
        )}
      </div>
    </div>
  );
}

function AlertError({ children }) {
  return (
    <div className="mt-3 flex items-start gap-2 text-red-600 text-xs">
      <AlertCircle size={15} className="shrink-0" />
      <span>{children}</span>
    </div>
  );
}
