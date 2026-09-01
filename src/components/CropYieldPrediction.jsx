import React, { useState, useCallback, useEffect } from 'react';
import { BarChart3, Zap, AlertCircle } from 'lucide-react';
import PredictionForm, { DEFAULT_INPUTS } from './PredictionForm';
import CropYieldResults from './CropYieldResults';
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

  useEffect(() => {
    fetchHealth()
      .then(h => setAvailability(h.models_loaded?.crop_yield || null))
      .catch(() => setAvailability(null));
  }, []);

  const handlePredict = useCallback(async () => {
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const regression = await predictCropYield(inputs, modelVariant);
      // Yield level (binary Low/High) comes from the real yield-level model.
      let level = null;
      try {
        level = await predictYieldLevel(inputs, modelVariant);
      } catch {
        // If the yield-level variant is unavailable, still show regression output.
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
    <div className="p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto space-y-6">
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-harvest-500 to-harvest-600 flex items-center justify-center">
            <BarChart3 size={22} className="text-white" />
          </div>
          <div>
            <h1 className="text-xl sm:text-2xl font-bold text-gray-900">
              Crop Yield Prediction
            </h1>
            <p className="text-sm text-gray-500">
              Regression + Classification · Predict production tonnage &amp; yield level
            </p>
          </div>
        </div>
      </div>

      {/* Info Banner */}
      <div className="flex items-start gap-3 p-4 rounded-xl bg-harvest-50 border border-harvest-100">
        <Zap size={18} className="text-harvest-600 mt-0.5 shrink-0" />
        <div>
          <p className="text-sm font-medium text-harvest-800">How it works</p>
          <p className="text-xs text-harvest-600 mt-0.5">
            Select a crop type and enter environmental conditions. The model performs
            regression (predicting yield in tons) and classification (categorizing yield
            as Low or High). The backend supplies all values — nothing is fabricated.
          </p>
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="flex items-start gap-3 p-4 rounded-xl bg-red-50 border border-red-200">
          <AlertCircle size={18} className="text-red-500 mt-0.5 shrink-0" />
          <div>
            <p className="text-sm font-medium text-red-700">Prediction failed</p>
            <p className="text-xs text-red-600 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Form */}
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

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-16">
          <div className="text-center">
            <div className="w-12 h-12 border-4 border-harvest-200 border-t-harvest-600 rounded-full animate-spin mx-auto mb-4" />
            <p className="text-sm font-medium text-gray-600">Estimating yield...</p>
            <p className="text-xs text-gray-400 mt-1">Running regression + classification</p>
          </div>
        </div>
      )}

      {/* Results */}
      {result && !loading && <CropYieldResults result={result} inputs={inputs} />}
    </div>
  );
}
