/**
 * Client-side |model − groundTruth| heatmap on an HTML Canvas.
 * Computes per-pixel absolute difference in luminance/RGB and maps to Turbo/Inferno colormap.
 */

// Turbo-like colormap stops: 0.0 (dark blue) -> 0.25 (cyan) -> 0.5 (green) -> 0.75 (yellow) -> 1.0 (bright red/white)
function turboColormap(t: number): [number, number, number] {
  const clampT = Math.max(0, Math.min(1, t));
  // 5-point piecewise linear gradient
  if (clampT < 0.25) {
    const s = clampT / 0.25;
    return [Math.round(10 * (1 - s) + 20 * s), Math.round(20 * (1 - s) + 160 * s), Math.round(80 * (1 - s) + 240 * s)];
  } else if (clampT < 0.5) {
    const s = (clampT - 0.25) / 0.25;
    return [Math.round(20 * (1 - s) + 40 * s), Math.round(160 * (1 - s) + 220 * s), Math.round(240 * (1 - s) + 80 * s)];
  } else if (clampT < 0.75) {
    const s = (clampT - 0.5) / 0.25;
    return [Math.round(40 * (1 - s) + 240 * s), Math.round(220 * (1 - s) + 180 * s), Math.round(80 * (1 - s) + 20 * s)];
  } else {
    const s = (clampT - 0.75) / 0.25;
    return [Math.round(240 * (1 - s) + 255 * s), Math.round(180 * (1 - s) + 70 * s), Math.round(20 * (1 - s) + 70 * s)];
  }
}

export async function generateDiffMap(
  imgUrlA: string,
  imgUrlB: string,
  targetWidth?: number,
  targetHeight?: number
): Promise<{ dataUrl: string; maxDiff: number; meanDiff: number }> {
  const [imgA, imgB] = await Promise.all([loadImage(imgUrlA), loadImage(imgUrlB)]);

  const w = targetWidth || Math.min(imgA.width, imgB.width);
  const h = targetHeight || Math.min(imgA.height, imgB.height);

  const canvasA = document.createElement('canvas');
  canvasA.width = w;
  canvasA.height = h;
  const ctxA = canvasA.getContext('2d', { willReadFrequently: true });

  const canvasB = document.createElement('canvas');
  canvasB.width = w;
  canvasB.height = h;
  const ctxB = canvasB.getContext('2d', { willReadFrequently: true });

  if (!ctxA || !ctxB) {
    throw new Error('Canvas 2D context not available');
  }

  ctxA.drawImage(imgA, 0, 0, w, h);
  ctxB.drawImage(imgB, 0, 0, w, h);

  const dataA = ctxA.getImageData(0, 0, w, h).data;
  const dataB = ctxB.getImageData(0, 0, w, h).data;

  const outCanvas = document.createElement('canvas');
  outCanvas.width = w;
  outCanvas.height = h;
  const outCtx = outCanvas.getContext('2d');
  if (!outCtx) throw new Error('Out Canvas 2D context not available');

  const outImgData = outCtx.createImageData(w, h);
  const outData = outImgData.data;

  let totalDiff = 0;
  let maxDiff = 0;
  const pixelCount = w * h;

  for (let i = 0; i < dataA.length; i += 4) {
    // RGB absolute error
    const dr = Math.abs(dataA[i] - dataB[i]);
    const dg = Math.abs(dataA[i + 1] - dataB[i + 1]);
    const db = Math.abs(dataA[i + 2] - dataB[i + 2]);
    const diff = (dr + dg + db) / 3 / 255; // 0..1

    totalDiff += diff;
    if (diff > maxDiff) maxDiff = diff;

    // Amplify slightly for visualization (surveillance error typically in 0..0.3)
    const amplified = Math.min(1.0, diff * 3.5);
    const [cr, cg, cb] = turboColormap(amplified);

    outData[i] = cr;
    outData[i + 1] = cg;
    outData[i + 2] = cb;
    outData[i + 3] = 255;
  }

  outCtx.putImageData(outImgData, 0, 0);

  return {
    dataUrl: outCanvas.toDataURL('image/png'),
    maxDiff: Math.round(maxDiff * 100) / 100,
    meanDiff: Math.round((totalDiff / pixelCount) * 1000) / 1000,
  };
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = (e) => reject(new Error(`Failed to load image: ${src}`));
    img.src = src;
  });
}

