// ============================================================================
// Agricultural Decision Support - Real API Service
// All calls proxy to the FastAPI backend at /api (vite proxy -> :8000).
// No mock, random, or fabricated prediction logic exists here.
// ============================================================================

const API_BASE = '/api';

// ─── Real dataset/model vocabulary (matches backend schemas & trained data) ─

export const REGIONS = [
  'Ayeyarwady', 'Bago', 'Chin', 'Kachin', 'Kayah',
  'Kayin', 'Magway', 'Mandalay', 'Mon', 'Nay Pyi Taw',
  'Rakhine', 'Sagaing', 'Shan', 'Tanintharyi', 'Yangon',
];

export const SOIL_TYPES = [
  'Acrisols / Ferralsols / Red Earth',
  'Acrisols / Red-Brown Lateritic',
  'Cambisols / Fluvisols / Loam',
  'Cambisols / Mountainous Soil',
  'Ferralsols / Gleysols / Lateritic',
  'Ferralsols / Plinthosols / Lateritic',
  'Fluvisols / Gleysols / Alluvial',
  'Gleysols / Alluvial / Mountainous',
  'Luvisols / Cambisols / Savanna',
  'Luvisols / Red Earth',
  'Regosols / Luvisols / Sandy Savanna',
];

export const WATER_SOURCES = ['Irrigation', 'Rainfed'];

export const SEEDING_SEASONS = ['Rainy', 'Summer', 'Winter'];

export const CROP_TYPES = [
  'Beteleaves', 'Betelnut', 'Bocatepe(Cow Pea)', 'Butter Bean',
  'Chillie', 'Coffee', 'Garlic', 'Gram(Chick pea)',
  'Groundnut(Rain)', 'Groundnut(Winter)', 'Maize', 'Matpe(Blackgram)',
  'Onion', 'Paddy', 'Panauk(Krishna mung)', 'Peboke(Soy bean)',
  'Pebyugale(Duffin bean)', 'Pedisein(Greengram)', 'Pegya(Lima bean)',
  'Pegyi(Lablab bean)', 'Pelun(Cow pea)', 'Pesingon(Pigeon pea)',
  'Peyin(Rice bean)', 'Plantain', 'Potato', 'Sadawpe(Garden pea)',
  'Sesamum(Early)', 'Sesamum(Late)', 'Sesamum(Summer)', 'Sugarcane',
  'Sunflower', 'Tea', 'Wheat',
];

// Yield level is binary: Low / High (as trained).
export const YIELD_LEVELS = ['Low', 'High'];

// ─── Model variants + availability (source of truth: backend registry) ──────
// The `available` flags are refreshed from GET /api/health at runtime. The
// defaults here mirror the trained artifacts and are only a fallback until the
// health call resolves.

export const MODEL_VARIANTS = {
  crop_type: [
    { id: 'baseline', name: 'Baseline Model', description: 'Raw unscaled data with original features', color: '#dc2626', bgColor: '#fef2f2' },
    { id: 'feature_engineering', name: 'Feature Engineering', description: 'Optimized selected features with scaling', color: '#2563eb', bgColor: '#eff6ff' },
    { id: 'advanced', name: 'Advanced-Association', description: 'Apriori rule-based interaction features (Soil×Water, etc.)', color: '#16a34a', bgColor: '#f0fdf4' },
  ],
  yield_level: [
    { id: 'baseline', name: 'Baseline Model', description: 'Requires target-leakage inputs (guarded)', color: '#dc2626', bgColor: '#fef2f2', guarded: true },
    { id: 'feature_engineering', name: 'Feature Engineering', description: 'Optimized selected features with scaling', color: '#2563eb', bgColor: '#eff6ff' },
    { id: 'advanced', name: 'Advanced-Association', description: 'Apriori + sequential interaction features', color: '#16a34a', bgColor: '#f0fdf4' },
  ],
  crop_yield: [
    { id: 'baseline', name: 'Baseline Model', description: 'Requires target-leakage inputs (guarded)', color: '#dc2626', bgColor: '#fef2f2', guarded: true },
    { id: 'feature_engineering', name: 'Feature Engineering', description: 'Optimized selected features with scaling', color: '#2563eb', bgColor: '#eff6ff' },
    { id: 'advanced', name: 'Advanced-Association', description: 'Apriori + sequential interaction features', color: '#16a34a', bgColor: '#f0fdf4' },
  ],
};

// ─── API error helper ───────────────────────────────────────────────────────

export function humanizeApiError(error) {
  if (!error) return 'Unknown error occurred.';
  if (error.__status) {
    const status = error.__status;
    if (status === 422) {
      // FastAPI validation error detail: [{ loc, msg }]
      const detail = error.detail;
      if (Array.isArray(detail)) {
        const msgs = detail.map((d) => `${d.loc?.join(' > ')}: ${d.msg}`).join('; ');
        return `Please fix the input: ${msgs}`;
      }
      return `Invalid input (${status}): ${String(detail)}`;
    }
    if (status === 503) return String(error.detail || error.message || 'Model unavailable / required data unavailable.');
    if (status >= 500) return 'Prediction service error. Please try again later.';
    return String(error.detail || error.message || `Request failed (${status}).`);
  }
  if (error.__network) return 'Cannot connect to ML backend.';
  return error.message || 'Request failed.';
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch (e) {
    const err = new Error('Cannot connect to ML backend.');
    err.__network = true;
    throw err;
  }

  let body = null;
  const raw = await response.text();
  if (raw) {
    try {
      body = JSON.parse(raw);
    } catch {
      body = raw;
    }
  }

  if (!response.ok) {
    const err = new Error(
      typeof body === 'string' ? body : (body && (body.detail || body.message)) || `Request failed (${response.status}).`
    );
    err.__status = response.status;
    err.detail = typeof body === 'object' && body !== null ? body.detail : body;
    err.status = response.status;
    throw err;
  }
  return body;
}

// ─── Frontend camelCase inputs -> backend snake_case request body ───────────

function toBackendRequest(inputs) {
  return {
    region: inputs.region,
    year: Number(inputs.year),
    crop_type: inputs.cropType,
    sown_acre: Number(inputs.sownAcre),
    soil_type: inputs.soilType,
    avg_temperature: Number(inputs.avgTemperature),
    total_rainfall: Number(inputs.totalRainfall),
    avg_humidity: Number(inputs.avgHumidity),
    water_source: inputs.waterSource,
    seeding_season: inputs.seedingSeason,
  };
}

// ─── Health / availability ──────────────────────────────────────────────────

export async function fetchHealth() {
  return request('/health');
}

// ─── Crop Type Prediction ───────────────────────────────────────────────────

export async function predictCropType(inputs, modelVariant = 'baseline') {
  const payload = { ...toBackendRequest(inputs), model_variant: modelVariant };
  return request('/predict/crop-type', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

// ─── Crop Yield Prediction ──────────────────────────────────────────────────

export async function predictCropYield(inputs, modelVariant = 'feature_engineering') {
  const payload = { ...toBackendRequest(inputs), model_variant: modelVariant };
  return request('/predict/crop-yield', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

// ─── Yield Level Prediction (binary Low/High) ───────────────────────────────

export async function predictYieldLevel(inputs, modelVariant = 'feature_engineering') {
  const payload = { ...toBackendRequest(inputs), model_variant: modelVariant };
  return request('/predict/yield-level', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

// ─── Read-only GET dedup cache ──────────────────────────────────────────────
// Predictions and health are always fresh. Read-only lookups (historical trends,
// model comparison) return stable data, so we reuse the in-flight/result promise
// to avoid redundant network calls (e.g. React StrictMode double-invocation and
// Dashboard remounts on navigation).

const _getCache = new Map();

function cachedGet(key, loader) {
  if (_getCache.has(key)) return _getCache.get(key);
  const promise = loader().catch(err => {
    _getCache.delete(key);
    throw err;
  });
  _getCache.set(key, promise);
  return promise;
}

// ─── Model Comparison ───────────────────────────────────────────────────────

export function getModelComparison(inputs, task = 'crop_type') {
  const params = new URLSearchParams({
    task,
    region: inputs.region,
    year: String(inputs.year),
    crop_type: inputs.cropType,
    sown_acre: String(inputs.sownAcre),
    soil_type: inputs.soilType,
    avg_temperature: String(inputs.avgTemperature),
    total_rainfall: String(inputs.totalRainfall),
    avg_humidity: String(inputs.avgHumidity),
    water_source: inputs.waterSource,
    seeding_season: inputs.seedingSeason,
  });
  const key = `/compare?${params.toString()}`;
  return cachedGet(key, () => request(key));
}

// ─── Historical Trends ──────────────────────────────────────────────────────
// Returns real historical values from the backend dataset. Surfaces errors and
// never fabricates values.

export function getHistoricalTrends(region, cropType) {
  const params = new URLSearchParams({ region, crop_type: cropType });
  const key = `/historical?${params.toString()}`;
  return cachedGet(key, () => request(`/historical?${params.toString()}`));
}

// ─── Historical Overview ─────────────────────────────────────────────────────
// Returns real per-crop historical yield series for a region plus the available
// year range. Surfaces errors and never fabricates values.

export function getHistoricalOverview(region) {
  const params = new URLSearchParams({ region });
  const key = `/historical/overview?${params.toString()}`;
  return cachedGet(key, () => request(`/historical/overview?${params.toString()}`));
}

// ─── Data Statistics ─────────────────────────────────────────────────────────
// Returns descriptive statistics computed from the real backend dataset. Stable
// read-only data, so we reuse the dedup cache.

export function getDataStatistics() {
  const key = '/stats';
  return cachedGet(key, () => request(key));
}

// ─── Descriptive Mining (Association Rules) ───────────────────────────────────
// Surfaces the repository's existing association-rule mining outputs (real CSV
// data) and the availability of other descriptive-mining components.

export function getDescriptiveMining() {
  const key = '/descriptive-mining';
  return cachedGet(key, () => request(key));
}

// ─── Chapter 4 ROC / AUC Evaluation (Crop Type) ──────────────────────────────
// Returns REAL per-class AUC + test support read from the Chapter 4 result files
// and the real one-vs-rest ROC curve points recomputed from the deployed model
// artifacts. `variant` is 'baseline' or 'feature_engineering'.

export function getRocEvaluation(variant = 'baseline') {
  const params = new URLSearchParams({ variant });
  const key = `/evaluation/roc?${params.toString()}`;
  return cachedGet(key, () => request(`/evaluation/roc?${params.toString()}`));
}
