/**
 * API client for the SISR Super-Resolution Studio backend.
 *
 * URL resolution order for every call:
 *   1. Same-origin proxy:   /api/v1/...      (no CORS, works via Vite dev server)
 *   2. Direct backend:      http://localhost:8000/api/v1/...   (CORS-allowed)
 *   3. Direct backend:      http://127.0.0.1:8000/api/v1/...   (CORS-allowed)
 * The first successful response wins. This makes the frontend resilient to
 * proxy misconfiguration, IPv6/IPv4 localhost resolution issues, and CORS
 * origin mismatches (localhost vs 127.0.0.1 are distinct CORS origins).
 */
const API_BASE = '/api/v1';
const DIRECT_BASES = ['http://localhost:8001', 'http://127.0.0.1:8001', 'http://localhost:8000', 'http://127.0.0.1:8000'];

async function _json(res, url) {
  if (!res.ok) {
    let detail = `HTTP ${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      detail = body.error || body.detail || JSON.stringify(body);
    } catch (_) {
      // ignore
    }
    const err = new Error(`${url}: ${detail}`);
    err.status = res.status;
    err.url = url;
    throw err;
  }
  return res.json();
}

/** Try a list of candidate URLs in order, returning the first successful JSON. */
async function _tryUrls(path, init) {
  const urls = [`${API_BASE}${path}`, ...DIRECT_BASES.map((b) => `${b}${path}`)];
  let lastErr;
  for (const url of urls) {
    try {
      // eslint-disable-next-line no-console
      console.log(`[api] trying ${url}`);
      return _json(await fetch(url, init), url);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn(`[api] ${url} failed:`, err.message);
      lastErr = err;
    }
  }
  throw lastErr;
}

export const api = {
  /** GET /api/v1/health */
  async health() {
    return _tryUrls('/health');
  },

  /** GET /api/v1/models */
  async getModels() {
    return _tryUrls('/models');
  },

  /** GET /api/v1/models/{model_id} */
  async getModel(modelId) {
    return _tryUrls(`/models/${encodeURIComponent(modelId)}`);
  },

  /**
   * POST /api/v1/super-resolve
   * file: File, model: string (model id), simulateLr: boolean
   */
  async superResolve(file, modelId, simulateLr = false) {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('model', modelId);
    fd.append('simulate_lr', String(simulateLr));
    return _tryUrls('/super-resolve', { method: 'POST', body: fd });
  },

  /**
   * POST /api/v1/bicubic
   * file: File, scale: number (2 or 4)
   */
  async bicubic(file, scale) {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('scale', String(scale));
    return _tryUrls('/bicubic', { method: 'POST', body: fd });
  },

  /**
   * POST /api/v1/compare
   * file: File, model_ids: "id1,id2" (same scale)
   */
  async compare(file, modelIds) {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('model_ids', modelIds);
    return _tryUrls('/compare', { method: 'POST', body: fd });
  },

  /**
   * POST /api/v1/evaluate
   * lr_file: File, hr_file: File, model_id: string
   */
  async evaluate(lrFile, hrFile, modelId) {
    const fd = new FormData();
    fd.append('lr_file', lrFile);
    fd.append('hr_file', hrFile);
    fd.append('model_id', modelId);
    return _tryUrls('/evaluate', { method: 'POST', body: fd });
  },
};

export default api;
