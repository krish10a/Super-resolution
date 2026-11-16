import { useState, useCallback, useRef } from 'react';

export interface MagnifierState {
  x: number; // 0..1 normalized
  y: number; // 0..1 normalized
  locked: boolean;
  active: boolean;
}

export function useMagnifier(defaultLensSize = 64) {
  const [state, setState] = useState<MagnifierState>({
    x: 0.5,
    y: 0.5,
    locked: false,
    active: true,
  });

  const containerRef = useRef<HTMLDivElement | null>(null);

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (state.locked || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const clientX = e.clientX;
      const clientY = e.clientY;

      const normX = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      const normY = Math.max(0, Math.min(1, (clientY - rect.top) / rect.height));

      setState((prev) => ({
        ...prev,
        x: normX,
        y: normY,
        active: true,
      }));
    },
    [state.locked]
  );

  const handlePointerLeave = useCallback(() => {
    if (!state.locked) {
      setState((prev) => ({ ...prev, active: false }));
    }
  }, [state.locked]);

  const toggleLock = useCallback(() => {
    setState((prev) => ({
      ...prev,
      locked: !prev.locked,
      active: true,
    }));
  }, []);

  const reset = useCallback(() => {
    setState({
      x: 0.5,
      y: 0.5,
      locked: false,
      active: false,
    });
  }, []);

  return {
    state,
    containerRef,
    handlePointerMove,
    handlePointerLeave,
    toggleLock,
    reset,
    lensSize: defaultLensSize,
  };
}

