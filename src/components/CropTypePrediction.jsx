import React, { useState, useCallback } from 'react';
import { Sprout, Zap } from 'lucide-react';
import PredictionForm, { DEFAULT_INPUTS } from './PredictionForm';
import CropTypeResults from './CropTypeResults';
import { predictCropType } from '../services/mockInference';

export default function CropTypePrediction() {
  const [inputs, setInputs] = useState({ ...DEFAULT_INPUTS });
  const [modelVariant, setModelVariant] = useState('advanced');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = useCallback(() => {
    setLoading(true);
    setResult(null);
    // Simulate processing delay for realism
    setTimeout(() => {
      const prediction = predictCropType(inputs, modelVariant);
      setResult(prediction);
      setLoading(false);
    }, 800);
  }, [inputs, modelVariant]);

  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto space-y-6">
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-forest-500 to-earth-500 flex items-center justify-center">
            <Sprout size={22} className="text-white" />
          </div>
          <div>
            <h1 className="text-xl sm:text-2xl font-bold text-gray-900">
              Crop Type Prediction
            </h1>
            <p className="text-sm text-gray-500">
              Classification · Predict the most suitable crop before planting
            </p>
          </div>
        </div>
      </div>

      {/* How it works banner */}
      <div className="flex items-start gap-3 p-4 rounded-xl bg-forest-50 border border-forest-100">
        <Zap size={18} className="text-forest-600 mt-0.5 shrink-0" />
        <div>
          <p className="text-sm font-medium text-forest-800">How it works</p>
          <p className="text-xs text-forest-600 mt-0.5">
            Enter your region's environmental conditions below. The model analyzes soil type,
            water source, temperature, rainfall, and humidity to recommend the optimal crop
            type with confidence scores and feature importance analysis.
          </p>
        </div>
      </div>

      {/* Prediction Form */}
      <PredictionForm
        inputs={inputs}
        setInputs={setInputs}
        modelVariant={modelVariant}
        setModelVariant={setModelVariant}
        onPredict={handlePredict}
        loading={loading}
        taskLabel="Predict Crop Type"
      />

      {/* Results */}
      {loading && (
        <div className="flex items-center justify-center py-16">
          <div className="text-center">
            <div className="w-12 h-12 border-4 border-forest-200 border-t-forest-600 rounded-full animate-spin mx-auto mb-4" />
            <p className="text-sm font-medium text-gray-600">Analyzing conditions...</p>
            <p className="text-xs text-gray-400 mt-1">Running {modelVariant} model</p>
          </div>
        </div>
      )}

      {result && !loading && <CropTypeResults result={result} />}
    </div>
  );
}
