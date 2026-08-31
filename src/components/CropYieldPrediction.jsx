import React, { useState, useCallback } from 'react';
import { BarChart3, Zap } from 'lucide-react';
import PredictionForm, { DEFAULT_INPUTS } from './PredictionForm';
import CropYieldResults from './CropYieldResults';
import { predictCropYield } from '../services/mockInference';

export default function CropYieldPrediction() {
  const [inputs, setInputs] = useState({ ...DEFAULT_INPUTS });
  const [modelVariant, setModelVariant] = useState('advanced');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = useCallback(() => {
    setLoading(true);
    setResult(null);
    setTimeout(() => {
      const prediction = predictCropYield(inputs, modelVariant);
      setResult(prediction);
      setLoading(false);
    }, 1000);
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
              Regression + Classification · Predict production tonnage & yield level
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
            Select a crop type and enter environmental conditions. The model performs both
            regression (predicting production in tons) and classification (categorizing yield
            as Low/Medium/High) with confidence intervals and performance metrics.
          </p>
        </div>
      </div>

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
