import React, { useRef, useState } from 'react';

export interface DropzoneProps {
  onFileSelect: (file: File, dataUrl: string) => void;
  selectedPreview?: string | null;
  dimensions?: [number, number] | null;
  onClear?: () => void;
  className?: string;
}

export const Dropzone: React.FC<DropzoneProps> = ({
  onFileSelect,
  selectedPreview,
  dimensions,
  onClear,
  className = '',
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file (PNG or JPEG)');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      onFileSelect(file, dataUrl);
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className={`space-y-2 ${className}`}>
      {selectedPreview ? (
        <div className="relative rounded-[var(--radius-control)] border border-[var(--border)] bg-[var(--surface)] p-2 overflow-hidden flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img
              src={selectedPreview}
              alt="Preview"
              className="w-12 h-12 object-cover rounded bg-[var(--bg-base)] border border-[var(--border)]"
            />
            <div>
              <div className="text-xs font-semibold text-[var(--text)]">Custom Image</div>
              {dimensions && (
                <div className="text-[11px] font-mono text-[var(--muted)]">
                  {dimensions[0]} × {dimensions[1]} px
                </div>
              )}
            </div>
          </div>
          {onClear && (
            <button
              onClick={onClear}
              className="p-1 text-[var(--muted)] hover:text-[var(--bad)] rounded transition-colors"
              title="Remove image"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      ) : (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragOver(true);
          }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragOver(false);
            handleFiles(e.dataTransfer.files);
          }}
          onClick={() => inputRef.current?.click()}
          className={`flex flex-col items-center justify-center p-4 border-2 border-dashed rounded-[var(--radius-control)] cursor-pointer transition-colors text-center ${
            isDragOver
              ? 'border-[var(--accent)] bg-[var(--accent)]/5'
              : 'border-[var(--border)] hover:border-[var(--accent)]/50 bg-[var(--surface)]/40 hover:bg-[var(--surface)]'
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/png, image/jpeg"
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
          <svg className="w-6 h-6 text-[var(--muted)] mb-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
            />
          </svg>
          <span className="text-xs text-[var(--text)] font-medium">
            Drag & drop surveillance frame
          </span>
          <span className="text-[10px] text-[var(--muted)] mt-0.5">PNG or JPG up to 10MB</span>
        </div>
      )}
    </div>
  );
};

