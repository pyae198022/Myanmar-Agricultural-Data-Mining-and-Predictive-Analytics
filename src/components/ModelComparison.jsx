import React, { useState, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Legend,
} from 'recharts';
import {
  GitCompare, Zap, Trophy, CheckCircle2, AlertCircle,
  ArrowRight, Star, TrendingUp, TrendingDown,
} from 'lucide-react';
import PredictionForm, { DEFAULT_INPUTS } from './PredictionForm';
import { getModelComparison, MODEL_VARIANTS } from '../services/mockInference';

function ComparisonTable({ comparison, task }) {
  const { models, bestModel, comparisonMetrics } = comparison;

  return (
    <div className="space-y-6">
      {/* Best Model Banner */}
      <div className="flex items-center gap-3 p-4 rounded-xl bg-gradient-to-r from-forest-50 to-earth-50 border border-forest-200">
        <Trophy size={22} className="text-harvest-500" />
        <div>
          <p className="text-sm font-bold text-forest-800">
            🏆 Best Performing Model: {MODEL_VARIANTS.find(v => v.id === bestModel)?.name}
          </p>
          <p className="text-xs text-forest-600">
            Based on {task === 'crop_type' ? 'Accuracy & F1-Score' : 'R² Score & RMSE'} evaluation
          </p>
        </div>
      </div>

      {/* Side-by-Side Results */}
      <div className="grid md:grid-cols-3 gap-4">
        {models.map(model => {
          const isBest = model.variant === bestModel;
          return (
            <div
              key={model.variant}
              className={`glass-card p-5 transition-all duration-300 ${
                isBest ? 'ring-2 ring-forest-400 shadow-lg' : ''
              }`}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: model.color }}
                  />
                  <span className="text-sm font-bold text-gray-800">{model.name}</span>
                </div>
                {isBest && (
                  <span className="flex items-center gap-1 text-xs font-bold text-forest-600 bg-forest-50 px-2 py-1 rounded-full">
                    <Star size={12} className="fill-forest-500" /> Best
                  </span>
                )}
              </div>

              {/* Prediction Result */}
              <div className="p-3 rounded-lg bg-gray-50 mb-4">
                {task === 'crop_type' ? (
                  <>
                    <p className="text-xs text-gray-500">Predicted Crop</p>
                    <p className="text-lg font-bold text-gray-900">
                      🌾 {model.prediction.predictedCrop}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Confidence: {(model.prediction.confidence * 100).toFixed(1)}%
                    </p>
                  </>
                ) : (
                  <>
                    <p className="text-xs text-gray-500">Predicted Yield</p>
                    <p className="text-lg font-bold text-gray-900">
                      {model.prediction.predictedYield.toFixed(2)} tons
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
                        model.prediction.yieldLevel === 'High' ? 'bg-green-100 text-green-700' :
                        model.prediction.yieldLevel === 'Medium' ? 'bg-yellow-100 text-yellow-700' :
                        'bg-red-100 text-red-700'
                      }`}>
                        {model.prediction.yieldLevel}
                      </span>
                    </div>
                  </>
                )}
              </div>

              {/* Metrics */}
              <div className="space-y-2">
                {task === 'crop_type' ? (
                  <>
                    <MetricRow label="Accuracy" value={model.metrics.accuracy} isPercent />
                    <MetricRow label="F1-Score" value={model.metrics.f1Score} isPercent />
                    <MetricRow label="Precision" value={model.metrics.precision} isPercent />
                    <MetricRow label="Recall" value={model.metrics.recall} isPercent />
                  </>
                ) : (
                  <>
                    <MetricRow label="R² Score" value={model.metrics.r2Score} isPercent={false} higherIsBetter />
                    <MetricRow label="RMSE" value={model.metrics.rmse} isPercent={false} higherIsBetter={false} />
                    <MetricRow label="MAE" value={model.metrics.mae} isPercent={false} higherIsBetter={false} />
                    <MetricRow label="Yield Acc." value={model.metrics.classificationAccuracy} isPercent />
                    <MetricRow label="Yield F1" value={model.metrics.classificationF1} isPercent />
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Comparison Charts */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Bar Chart Comparison */}
        <div className="glass-card p-5">
          <h4 className="text-sm font-semibold text-gray-800 mb-4">
            Metrics Comparison (Bar Chart)
          </h4>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonMetrics} barCategoryGap="20%">
                <XAxis
                  dataKey="metric"
                  tick={{ fontSize: 10, fill: '#374151' }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: '#9ca3af' }}
                  axisLine={false}
                  tickLine={false}
                  domain={[0, 'auto']}
                />
                <Tooltip
                  contentStyle={{
                    borderRadius: '12px',
                    border: '1px solid #e5e7eb',
                    fontSize: '12px',
                  }}
                />
                <Legend
                  wrapperStyle={{ fontSize: '11px' }}
                  formatter={(value) => {
                    const v = MODEL_VARIANTS.find(m => m.id === value);
                    return v ? v.name : value;
                  }}
                />
                <Bar dataKey="baseline" fill="#dc2626" radius={[4, 4, 0, 0]} />
                <Bar dataKey="feature_eng" fill="#2563eb" radius={[4, 4, 0, 0]} />
                <Bar dataKey="advanced" fill="#16a34a" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Radar Chart Comparison */}
        <div className="glass-card p-5">
          <h4 className="text-sm font-semibold text-gray-800 mb-4">
            Performance Radar
          </h4>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart
                data={comparisonMetrics.filter(m => !m.lowerIsBetter).map(m => ({
                  metric: m.metric,
                  baseline: m.baseline,
                  feature_eng: m.feature_eng,
                  advanced: m.advanced,
                }))}
                cx="50%" cy="50%" outerRadius="70%"
              >
                <PolarGrid stroke="#e5e7eb" />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: '#6b7280' }} />
                <PolarRadiusAxis angle={90} domain={[0, 1]} tick={{ fontSize: 9, fill: '#9ca3af' }} />
                <Radar name="Baseline" dataKey="baseline" stroke="#dc2626" fill="#dc2626" fillOpacity={0.1} strokeWidth={2} />
                <Radar name="Feature Eng." dataKey="feature_eng" stroke="#2563eb" fill="#2563eb" fillOpacity={0.1} strokeWidth={2} />
                <Radar name="Advanced" dataKey="advanced" stroke="#16a34a" fill="#16a34a" fillOpacity={0.15} strokeWidth={2} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Detailed Comparison Table */}
      <div className="glass-card p-5 overflow-x-auto">
        <h4 className="text-sm font-semibold text-gray-800 mb-4">
          Detailed Metrics Comparison Table
        </h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200">
              <th className="text-left py-3 px-4 font-semibold text-gray-700">Metric</th>
              {models.map(m => (
                <th key={m.variant} className="text-center py-3 px-4">
                  <div className="flex items-center justify-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: m.color }} />
                    <span className="font-semibold text-gray-700">{m.name}</span>
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {comparisonMetrics.map((row, i) => {
              const values = models.map(m => row[m.variant]);
              const bestValue = row.lowerIsBetter
                ? Math.min(...values)
                : Math.max(...values);

              return (
                <tr key={i} className="border-b border-gray-50 hover:bg-gray-50 transition-colors">
                  <td className="py-3 px-4 font-medium text-gray-700">
                    {row.metric}
                    {row.lowerIsBetter && (
                      <span className="text-xs text-gray-400 ml-1">(↓ lower = better)</span>
                    )}
                  </td>
                  {models.map(m => {
                    const val = row[m.variant];
                    const isBestVal = val === bestValue;
                    return (
                      <td key={m.variant} className="py-3 px-4 text-center">
                        <span className={`font-mono font-semibold ${
                          isBestVal ? 'text-forest-600 bg-forest-50 px-2 py-1 rounded-md' : 'text-gray-600'
                        }`}>
                          {typeof val === 'number' ? val.toFixed(3) : val}
                        </span>
                        {isBestVal && <Star size={10} className="inline ml-1 text-harvest-500 fill-harvest-400" />}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function MetricRow({ label, value, isPercent = false, higherIsBetter = true }) {
  const displayValue = isPercent ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-gray-50">
      <span className="text-xs text-gray-600">{label}</span>
      <span className="text-xs font-mono font-bold text-gray-800">{displayValue}</span>
    </div>
  );
}

export default function ModelComparison() {
  const [inputs, setInputs] = useState({ ...DEFAULT_INPUTS });
  const [modelVariant, setModelVariant] = useState('advanced');
  const [task, setTask] = useState('crop_type');
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleCompare = useCallback(() => {
    setLoading(true);
    setComparison(null);
    setTimeout(() => {
      const result = getModelComparison(
        task === 'crop_yield' ? inputs : { ...inputs },
        task
      );
      setComparison(result);
      setLoading(false);
    }, 1200);
  }, [inputs, task]);

  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto space-y-6">
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-500 flex items-center justify-center">
            <GitCompare size={22} className="text-white" />
          </div>
          <div>
            <h1 className="text-xl sm:text-2xl font-bold text-gray-900">
              Model Comparison
            </h1>
            <p className="text-sm text-gray-500">
              Side-by-Side · Compare Baseline, Feature Engineering & Advanced models
            </p>
          </div>
        </div>
      </div>

      {/* Info */}
      <div className="flex items-start gap-3 p-4 rounded-xl bg-blue-50 border border-blue-100">
        <Zap size={18} className="text-blue-600 mt-0.5 shrink-0" />
        <div>
          <p className="text-sm font-medium text-blue-800">Compare All Models</p>
          <p className="text-xs text-blue-600 mt-0.5">
            Run predictions across all three model variants simultaneously and compare their
            performance metrics (R², RMSE, MAE for regression; Accuracy, F1-Score for classification)
            in structured comparison tables and charts.
          </p>
        </div>
      </div>

      {/* Task Selector */}
      <div className="flex gap-3">
        <button
          onClick={() => { setTask('crop_type'); setComparison(null); }}
          className={`px-5 py-2.5 rounded-xl text-sm font-semibold transition-all ${
            task === 'crop_type'
              ? 'bg-forest-600 text-white shadow-md'
              : 'bg-white text-gray-600 border border-gray-200 hover:border-forest-300'
          }`}
        >
          🌱 Crop Type (Classification)
        </button>
        <button
          onClick={() => { setTask('crop_yield'); setComparison(null); }}
          className={`px-5 py-2.5 rounded-xl text-sm font-semibold transition-all ${
            task === 'crop_yield'
              ? 'bg-harvest-600 text-white shadow-md'
              : 'bg-white text-gray-600 border border-gray-200 hover:border-harvest-300'
          }`}
        >
          📊 Crop Yield (Regression)
        </button>
      </div>

      {/* Form - reuse but hide model variant selector since we compare all */}
      <PredictionForm
        inputs={inputs}
        setInputs={setInputs}
        modelVariant={modelVariant}
        setModelVariant={setModelVariant}
        onPredict={handleCompare}
        loading={loading}
        showCropType={task === 'crop_yield'}
        taskLabel="Compare All Models"
      />

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-16">
          <div className="text-center">
            <div className="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4" />
            <p className="text-sm font-medium text-gray-600">Running all models...</p>
            <p className="text-xs text-gray-400 mt-1">Baseline → Feature Eng → Advanced</p>
          </div>
        </div>
      )}

      {/* Comparison Results */}
      {comparison && !loading && <ComparisonTable comparison={comparison} task={task} />}
    </div>
  );
}
