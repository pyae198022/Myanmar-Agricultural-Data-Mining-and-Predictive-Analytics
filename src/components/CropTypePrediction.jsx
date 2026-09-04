import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Sprout, Zap, AlertCircle } from 'lucide-react';
import PredictionForm, { DEFAULT_INPUTS } from './PredictionForm';
import CropTypeResults from './CropTypeResults';
import PageHeader, { InfoBanner, ResultsPlaceholder } from './PageHeader';
import { predictCropType, fetchHealth, humanizeApiError } from '../services/apiService';

export default function CropTypePrediction() {
  const [inputs, setInputs] = useState({ ...DEFAULT_INPUTS });
  const [modelVariant, setModelVariant] = useState('feature_engineering');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [availability, setAvailability] = useState(null);
  const resultsRef = useRef(null);

  useEffect(() => {
    fetchHealth()
      .then(h => setAvailability(h.models_loaded?.crop_type || null))
      .catch(() => setAvailability(null));
  }, []);

  useEffect(() => {
    if (result && resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [result]);

  const handlePredict = useCallback(async () => {
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const prediction = await predictCropType(inputs, modelVariant);
      setResult(prediction);
    } catch (e) {
      setError(humanizeApiError(e));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [inputs, modelVariant]);

  return (
    <div className="page-shell-wide">
      <PageHeader
        icon={Sprout}
        title="Crop Type Prediction"
        subtitle="Recommend the most suitable crop from soil, weather, and water conditions."
        gradient="from-forest-500 to-earth-500"
      />

      <InfoBanner icon={Zap} title="How it works" tone="forest">
        Enter regional conditions. The model uses soil type, water source, temperature,
        rainfall, and humidity to recommend a crop with confidence scores and feature importance.
      </InfoBanner>

      {error && (
        <InfoBanner icon={AlertCircle} title="Prediction failed" tone="red">
          {error}
        </InfoBanner>
      )}

      <PredictionForm
        inputs={inputs}
        setInputs={setInputs}
        modelVariant={modelVariant}
        setModelVariant={setModelVariant}
        onPredict={handlePredict}
        loading={loading}
        taskLabel="Predict Crop Type"
        task="crop_type"
        availability={availability}
      />

      <div ref={resultsRef} aria-live="polite">
        {loading && (
          <div className="glass-card p-10 flex flex-col items-center justify-center min-h-[200px]">
            <div className="w-12 h-12 border-4 border-forest-200 border-t-forest-600 rounded-full animate-spin mb-4" />
            <p className="text-sm font-medium text-gray-600">Analyzing conditions...</p>
            <p className="text-xs text-gray-400 mt-1">Running {modelVariant.replaceAll('_', ' ')} model</p>
          </div>
        )}
        {result && !loading && <CropTypeResults result={result} />}
        {!result && !loading && (
          <ResultsPlaceholder
            title="No prediction yet"
            hint="Set conditions above, then run Predict Crop Type. Charts and recommendations will use the full width of this page."
          />
        )}
      </div>
    </div>
  );
}
