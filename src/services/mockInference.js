// ============================================================================
// Agricultural Decision Support - Mock Inference Service
// Provides deterministic, realistic mock predictions for crop type & yield
// ============================================================================

// ─── Constants ──────────────────────────────────────────────────────────────

export const REGIONS = [
  'Ayeyarwady', 'Bago', 'Chin', 'Kachin', 'Kayah',
  'Kayin', 'Magway', 'Mandalay', 'Mon', 'Naypyitaw',
  'Rakhine', 'Sagaing', 'Shan', 'Tanintharyi', 'Yangon'
];

export const SOIL_TYPES = [
  'Alluvial', 'Clay', 'Loamy', 'Sandy', 'Silt',
  'Red Soil', 'Black Soil', 'Laterite'
];

export const WATER_SOURCES = [
  'Rainfed', 'Canal', 'Tubewell', 'River', 'Pond', 'Dam', 'Borewell'
];

export const CROP_TYPES = [
  'Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton',
  'Groundnut', 'Sesame', 'Pulses', 'Soybean', 'Sunflower',
  'Jute', 'Millet'
];

export const YIELD_LEVELS = ['Low', 'Medium', 'High'];

export const MODEL_VARIANTS = [
  {
    id: 'baseline',
    name: 'Baseline Model',
    description: 'Raw unscaled data with original features',
    color: '#dc2626',      // red
    bgColor: '#fef2f2',
  },
  {
    id: 'feature_eng',
    name: 'Feature Engineering',
    description: 'Optimized selected features with scaling',
    color: '#2563eb',      // blue
    bgColor: '#eff6ff',
  },
  {
    id: 'advanced',
    name: 'Advanced-Association',
    description: 'Apriori rule-based interaction features (Soil×Water, etc.)',
    color: '#16a34a',      // green
    bgColor: '#f0fdf4',
  },
];

// ─── Deterministic PRNG ─────────────────────────────────────────────────────

function hashString(str) {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash + str.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

function seededRandom(seed) {
  const h = hashString(String(seed));
  let s = h % 2147483647;
  if (s <= 0) s += 2147483646;
  s = (s * 16807) % 2147483647;
  return (s - 1) / 2147483646;
}

function seededRandomRange(seed, min, max) {
  return min + seededRandom(seed) * (max - min);
}

function generateInputSeed(inputs) {
  return Object.values(inputs).join('|');
}

// ─── Crop Affinity Lookup ───────────────────────────────────────────────────
// Realistic mappings of which crops suit which soil/water/region combos

const CROP_AFFINITY = {
  Rice:       { soils: ['Alluvial', 'Clay', 'Silt'], waters: ['Canal', 'River', 'Dam', 'Rainfed'], tempRange: [22, 35], rainRange: [100, 300] },
  Wheat:      { soils: ['Loamy', 'Clay', 'Alluvial'], waters: ['Canal', 'Tubewell', 'Borewell'], tempRange: [12, 25], rainRange: [40, 120] },
  Maize:      { soils: ['Loamy', 'Sandy', 'Red Soil'], waters: ['Rainfed', 'Canal', 'Tubewell'], tempRange: [20, 32], rainRange: [60, 180] },
  Sugarcane:  { soils: ['Alluvial', 'Loamy', 'Black Soil'], waters: ['Canal', 'River', 'Dam'], tempRange: [24, 38], rainRange: [80, 250] },
  Cotton:     { soils: ['Black Soil', 'Loamy', 'Red Soil'], waters: ['Rainfed', 'Canal', 'Borewell'], tempRange: [22, 35], rainRange: [50, 150] },
  Groundnut:  { soils: ['Sandy', 'Loamy', 'Red Soil'], waters: ['Rainfed', 'Tubewell', 'Borewell'], tempRange: [25, 35], rainRange: [50, 130] },
  Sesame:     { soils: ['Sandy', 'Loamy', 'Alluvial'], waters: ['Rainfed', 'Tubewell'], tempRange: [25, 38], rainRange: [30, 100] },
  Pulses:     { soils: ['Loamy', 'Clay', 'Red Soil', 'Black Soil'], waters: ['Rainfed', 'Canal', 'Tubewell'], tempRange: [18, 30], rainRange: [40, 120] },
  Soybean:    { soils: ['Loamy', 'Clay', 'Black Soil'], waters: ['Rainfed', 'Canal'], tempRange: [20, 30], rainRange: [60, 160] },
  Sunflower:  { soils: ['Loamy', 'Sandy', 'Black Soil'], waters: ['Rainfed', 'Tubewell', 'Borewell'], tempRange: [20, 30], rainRange: [40, 100] },
  Jute:       { soils: ['Alluvial', 'Clay', 'Silt'], waters: ['River', 'Canal', 'Rainfed'], tempRange: [24, 37], rainRange: [150, 300] },
  Millet:     { soils: ['Sandy', 'Loamy', 'Red Soil', 'Laterite'], waters: ['Rainfed', 'Tubewell'], tempRange: [25, 38], rainRange: [25, 80] },
};

function computeCropScore(crop, inputs) {
  const aff = CROP_AFFINITY[crop];
  let score = 0;
  if (aff.soils.includes(inputs.soilType)) score += 3;
  if (aff.waters.includes(inputs.waterSource)) score += 2;
  const temp = Number(inputs.avgTemperature);
  if (temp >= aff.tempRange[0] && temp <= aff.tempRange[1]) score += 2.5;
  else {
    const dist = Math.min(Math.abs(temp - aff.tempRange[0]), Math.abs(temp - aff.tempRange[1]));
    score += Math.max(0, 2.5 - dist * 0.15);
  }
  const rain = Number(inputs.totalRainfall);
  if (rain >= aff.rainRange[0] && rain <= aff.rainRange[1]) score += 2;
  else {
    const dist = Math.min(Math.abs(rain - aff.rainRange[0]), Math.abs(rain - aff.rainRange[1]));
    score += Math.max(0, 2 - dist * 0.01);
  }
  const humidity = Number(inputs.avgHumidity);
  if (humidity >= 50 && humidity <= 85) score += 0.5;
  return score;
}

// ─── Model Performance Scaling ──────────────────────────────────────────────

const MODEL_PERFORMANCE = {
  baseline: {
    accuracyRange: [0.72, 0.78],
    f1Range: [0.68, 0.75],
    confidenceRange: [0.55, 0.70],
    r2Range: [0.65, 0.72],
    rmseRange: [2.1, 3.5],
    maeRange: [1.6, 2.8],
    yieldAccRange: [0.65, 0.73],
    yieldF1Range: [0.62, 0.70],
  },
  feature_eng: {
    accuracyRange: [0.82, 0.88],
    f1Range: [0.79, 0.86],
    confidenceRange: [0.70, 0.82],
    r2Range: [0.78, 0.85],
    rmseRange: [1.4, 2.3],
    maeRange: [1.0, 1.8],
    yieldAccRange: [0.76, 0.84],
    yieldF1Range: [0.73, 0.82],
  },
  advanced: {
    accuracyRange: [0.88, 0.94],
    f1Range: [0.86, 0.93],
    confidenceRange: [0.82, 0.93],
    r2Range: [0.86, 0.93],
    rmseRange: [0.8, 1.5],
    maeRange: [0.5, 1.1],
    yieldAccRange: [0.84, 0.92],
    yieldF1Range: [0.82, 0.90],
  },
};

// ─── Feature Names ──────────────────────────────────────────────────────────

const CLASSIFICATION_FEATURES = [
  'Region', 'Soil Type', 'Water Source', 'Avg Temperature',
  'Total Rainfall', 'Avg Humidity', 'Year', 'Sown Acre'
];

const REGRESSION_FEATURES = [
  'Crop Type', 'Region', 'Soil Type', 'Water Source',
  'Avg Temperature', 'Total Rainfall', 'Avg Humidity', 'Year', 'Sown Acre'
];

// ─── Crop Type Prediction ───────────────────────────────────────────────────

export function predictCropType(inputs, modelVariant = 'baseline') {
  const seed = generateInputSeed(inputs) + modelVariant;
  const perf = MODEL_PERFORMANCE[modelVariant];

  // Compute crop scores
  const scores = CROP_TYPES.map(crop => ({
    crop,
    score: computeCropScore(crop, inputs) + seededRandomRange(seed + crop, -1.5, 1.5),
  }));

  // Model variant improvement: better models reduce noise
  const noiseReduction = modelVariant === 'advanced' ? 0.3 : modelVariant === 'feature_eng' ? 0.6 : 1.0;
  scores.forEach(s => {
    const noise = seededRandomRange(seed + s.crop + 'noise', -2, 2) * noiseReduction;
    s.score += noise;
  });

  // Sort by score descending
  scores.sort((a, b) => b.score - a.score);

  // Convert to probabilities using softmax-like approach
  const maxScore = scores[0].score;
  const expScores = scores.map(s => ({
    crop: s.crop,
    exp: Math.exp((s.score - maxScore) * 0.8),
  }));
  const sumExp = expScores.reduce((acc, s) => acc + s.exp, 0);
  const topPredictions = expScores.slice(0, 5).map(s => ({
    crop: s.crop,
    probability: Math.round((s.exp / sumExp) * 1000) / 1000,
  }));

  // Confidence for the top prediction
  const confidence = Math.round(
    seededRandomRange(seed + 'conf', perf.confidenceRange[0], perf.confidenceRange[1]) * 100
  ) / 100;

  // Feature importance (varies by model variant)
  const baseImportances = {
    baseline: [0.12, 0.18, 0.14, 0.15, 0.16, 0.10, 0.05, 0.10],
    feature_eng: [0.08, 0.22, 0.16, 0.17, 0.18, 0.08, 0.03, 0.08],
    advanced: [0.06, 0.25, 0.19, 0.14, 0.20, 0.07, 0.02, 0.07],
  };

  const importances = baseImportances[modelVariant].map((base, i) => {
    const jitter = seededRandomRange(seed + 'fi' + i, -0.02, 0.02);
    return Math.max(0.01, base + jitter);
  });
  const impSum = importances.reduce((a, b) => a + b, 0);

  const featureImportance = CLASSIFICATION_FEATURES.map((f, i) => ({
    feature: f,
    importance: Math.round((importances[i] / impSum) * 1000) / 1000,
  })).sort((a, b) => b.importance - a.importance);

  // Model metrics
  const accuracy = Math.round(seededRandomRange(seed + 'acc', ...perf.accuracyRange) * 1000) / 1000;
  const f1Score = Math.round(seededRandomRange(seed + 'f1', ...perf.f1Range) * 1000) / 1000;
  const precision = Math.round(seededRandomRange(seed + 'prec', accuracy - 0.03, accuracy + 0.02) * 1000) / 1000;
  const recall = Math.round(seededRandomRange(seed + 'rec', f1Score - 0.02, f1Score + 0.03) * 1000) / 1000;

  return {
    predictedCrop: scores[0].crop,
    confidence,
    topPredictions,
    featureImportance,
    modelMetrics: {
      accuracy,
      f1Score,
      precision,
      recall,
    },
  };
}

// ─── Crop Yield Prediction ──────────────────────────────────────────────────

export function predictCropYield(inputs, modelVariant = 'baseline') {
  const seed = generateInputSeed(inputs) + modelVariant;
  const perf = MODEL_PERFORMANCE[modelVariant];

  // Base yield from crop type
  const cropBaseYield = {
    Rice: 4.5, Wheat: 3.2, Maize: 5.0, Sugarcane: 18.0, Cotton: 2.0,
    Groundnut: 1.8, Sesame: 0.8, Pulses: 1.5, Soybean: 2.2,
    Sunflower: 1.6, Jute: 2.5, Millet: 1.2,
  };

  const baseYield = cropBaseYield[inputs.cropType] || 3.0;

  // Modifiers based on environmental factors
  const temp = Number(inputs.avgTemperature);
  const rain = Number(inputs.totalRainfall);
  const humidity = Number(inputs.avgHumidity);
  const acre = Number(inputs.sownAcre);

  // Temperature modifier (optimal range varies by crop)
  const tempMod = temp >= 20 && temp <= 32 ? 1.0 + (temp - 26) * 0.01 : 0.75 + seededRandom(seed + 'tmod') * 0.15;

  // Rainfall modifier
  const rainMod = rain >= 50 && rain <= 250 ? 0.9 + (rain / 300) * 0.2 : 0.7 + seededRandom(seed + 'rmod') * 0.15;

  // Humidity modifier
  const humMod = humidity >= 40 && humidity <= 80 ? 0.95 + humidity / 1000 : 0.85;

  // Acre scaling (not linear - diminishing returns)
  const acreScale = Math.pow(acre / 100, 0.85);

  // Noise based on model quality
  const noiseScale = modelVariant === 'advanced' ? 0.05 : modelVariant === 'feature_eng' ? 0.12 : 0.20;
  const noise = seededRandomRange(seed + 'ynoise', -noiseScale, noiseScale);

  let predictedYield = baseYield * tempMod * rainMod * humMod * acreScale * (1 + noise);
  predictedYield = Math.round(Math.max(0.1, predictedYield) * 100) / 100;

  // Yield level classification
  const yieldPerAcre = predictedYield / Math.max(acre / 100, 0.01);
  let yieldLevel;
  if (yieldPerAcre < 2.0) yieldLevel = 'Low';
  else if (yieldPerAcre < 5.0) yieldLevel = 'Medium';
  else yieldLevel = 'High';

  // Confidence interval
  const ciWidth = modelVariant === 'advanced' ? 0.08 : modelVariant === 'feature_eng' ? 0.15 : 0.22;
  const confidenceInterval = {
    lower: Math.round(Math.max(0, predictedYield * (1 - ciWidth)) * 100) / 100,
    upper: Math.round(predictedYield * (1 + ciWidth) * 100) / 100,
  };

  // Feature importance for regression
  const regImportances = {
    baseline: [0.15, 0.08, 0.12, 0.10, 0.14, 0.16, 0.09, 0.04, 0.12],
    feature_eng: [0.18, 0.06, 0.14, 0.11, 0.16, 0.18, 0.07, 0.02, 0.08],
    advanced: [0.20, 0.05, 0.16, 0.13, 0.14, 0.19, 0.06, 0.02, 0.05],
  };

  const rImps = regImportances[modelVariant].map((base, i) => {
    const jitter = seededRandomRange(seed + 'rfi' + i, -0.015, 0.015);
    return Math.max(0.01, base + jitter);
  });
  const rSum = rImps.reduce((a, b) => a + b, 0);

  const featureImportance = REGRESSION_FEATURES.map((f, i) => ({
    feature: f,
    importance: Math.round((rImps[i] / rSum) * 1000) / 1000,
  })).sort((a, b) => b.importance - a.importance);

  // Model metrics
  const r2Score = Math.round(seededRandomRange(seed + 'r2', ...perf.r2Range) * 1000) / 1000;
  const rmse = Math.round(seededRandomRange(seed + 'rmse', ...perf.rmseRange) * 100) / 100;
  const mae = Math.round(seededRandomRange(seed + 'mae', ...perf.maeRange) * 100) / 100;
  const classificationAccuracy = Math.round(
    seededRandomRange(seed + 'yacc', ...perf.yieldAccRange) * 1000
  ) / 1000;
  const classificationF1 = Math.round(
    seededRandomRange(seed + 'yf1', ...perf.yieldF1Range) * 1000
  ) / 1000;

  return {
    predictedYield,
    yieldLevel,
    confidenceInterval,
    featureImportance,
    modelMetrics: {
      r2Score,
      rmse,
      mae,
      classificationAccuracy,
      classificationF1,
    },
  };
}

// ─── Model Comparison ───────────────────────────────────────────────────────

export function getModelComparison(inputs, task = 'crop_type') {
  const variants = ['baseline', 'feature_eng', 'advanced'];
  const predictFn = task === 'crop_type' ? predictCropType : predictCropYield;

  const models = variants.map(variant => {
    const prediction = predictFn(inputs, variant);
    const variantInfo = MODEL_VARIANTS.find(m => m.id === variant);
    return {
      variant,
      name: variantInfo.name,
      color: variantInfo.color,
      bgColor: variantInfo.bgColor,
      prediction,
      metrics: prediction.modelMetrics,
    };
  });

  // Determine best model
  let bestModel;
  if (task === 'crop_type') {
    bestModel = models.reduce((best, m) =>
      m.metrics.accuracy > best.metrics.accuracy ? m : best
    ).variant;
  } else {
    bestModel = models.reduce((best, m) =>
      m.metrics.r2Score > best.metrics.r2Score ? m : best
    ).variant;
  }

  // Build comparison metrics for charts
  let comparisonMetrics;
  if (task === 'crop_type') {
    comparisonMetrics = [
      {
        metric: 'Accuracy',
        ...Object.fromEntries(models.map(m => [m.variant, m.metrics.accuracy])),
      },
      {
        metric: 'F1-Score',
        ...Object.fromEntries(models.map(m => [m.variant, m.metrics.f1Score])),
      },
      {
        metric: 'Precision',
        ...Object.fromEntries(models.map(m => [m.variant, m.metrics.precision])),
      },
      {
        metric: 'Recall',
        ...Object.fromEntries(models.map(m => [m.variant, m.metrics.recall])),
      },
    ];
  } else {
    comparisonMetrics = [
      {
        metric: 'R² Score',
        ...Object.fromEntries(models.map(m => [m.variant, m.metrics.r2Score])),
      },
      {
        metric: 'RMSE',
        ...Object.fromEntries(models.map(m => [m.variant, m.metrics.rmse])),
        lowerIsBetter: true,
      },
      {
        metric: 'MAE',
        ...Object.fromEntries(models.map(m => [m.variant, m.metrics.mae])),
        lowerIsBetter: true,
      },
      {
        metric: 'Yield Accuracy',
        ...Object.fromEntries(models.map(m => [m.variant, m.metrics.classificationAccuracy])),
      },
      {
        metric: 'Yield F1',
        ...Object.fromEntries(models.map(m => [m.variant, m.metrics.classificationF1])),
      },
    ];
  }

  return { models, bestModel, comparisonMetrics };
}

// ─── Historical Trends ──────────────────────────────────────────────────────

export function getHistoricalTrends(region, cropType) {
  const seed = region + cropType;
  const baseYield = {
    Rice: 4.2, Wheat: 3.0, Maize: 4.5, Sugarcane: 16.0, Cotton: 1.8,
    Groundnut: 1.5, Sesame: 0.7, Pulses: 1.3, Soybean: 2.0,
    Sunflower: 1.4, Jute: 2.3, Millet: 1.0,
  }[cropType] || 3.0;

  return Array.from({ length: 11 }, (_, i) => {
    const year = 2015 + i;
    const s = seed + year;
    const trend = 1 + i * 0.018; // gradual improvement over years
    const variation = seededRandomRange(s + 'v', 0.85, 1.15);

    return {
      year,
      yield: Math.round(baseYield * trend * variation * 100) / 100,
      rainfall: Math.round(seededRandomRange(s + 'r', 60, 250)),
      temperature: Math.round(seededRandomRange(s + 't', 20, 35) * 10) / 10,
      area: Math.round(seededRandomRange(s + 'a', 50, 500)),
    };
  });
}
