import React, { useState, useCallback, useEffect, useRef } from 'react';
import { BarChart3, Zap, AlertCircle } from 'lucide-react';
import PredictionForm, { DEFAULT_INPUTS } from './PredictionForm';
import CropYieldResults from './CropYieldResults';
import PageHeader, { InfoBanner, ResultsPlaceholder } from './PageHeader';
import {
  predictCropYield, predictYieldLevel, fetchHealth, humanizeApiError,
} from '../services/apiService';

export default function CropYieldPrediction() {
  const [inputs, setInputs] = useState({ ...DEFAULT_INPUTS });
  const [modelVariant, setModelVariant] = useState('feature_engineering');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [availability, setAvailability] = useState(null);
  const resultsRef = useRef(null);

  useEffect(() => {
    fetchHealth()
      .then(h => setAvailability(h.models_loaded?.crop_yield || null))
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
      const regression = await predictCropYield(inputs, modelVariant);
      let level = null;
      try {
        level = await predictYieldLevel(inputs, modelVariant);
      } catch {
        level = null;
      }
      setResult({ ...regression, yieldLevelInfo: level });
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
        icon={BarChart3}
        title="Crop Yield Prediction"
        subtitle="Estimate production tonnage and whether yield is likely Low or High."
        gradient="from-harvest-500 to-harvest-600"
      />

      <InfoBanner icon={Zap} title="How it works" tone="harvest">
        Select a crop and enter environmental conditions. Regression predicts yield in tons;
        classification labels it Low or High. All values come from the backend.
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
        showCropType={true}
        taskLabel="Predict Yield"
        task="crop_yield"
        availability={availability}
      />

      <div ref={resultsRef} aria-live="polite">
        {loading && (
          <div className="glass-card p-10 flex flex-col items-center justify-center min-h-[200px]">
            <div className="w-12 h-12 border-4 border-harvest-200 border-t-harvest-600 rounded-full animate-spin mb-4" />
            <p className="text-sm font-medium text-gray-600">Estimating yield...</p>
            <p className="text-xs text-gray-400 mt-1">Running regression + classification</p>
          </div>
        )}
        {result && !loading && <CropYieldResults result={result} inputs={inputs} />}
        {!result && !loading && (
          <ResultsPlaceholder
            title="No yield estimate yet"
            hint="Choose a crop and conditions above, then run Predict Yield. Production and yield level will show at full width below."
          />
        )}
      </div>
    </div>
  );
}
