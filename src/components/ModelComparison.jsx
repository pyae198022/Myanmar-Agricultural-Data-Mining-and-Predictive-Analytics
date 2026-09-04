import React, { useState, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Legend,
} from 'recharts';
import {
  GitCompare, Zap, Trophy, AlertCircle,
  Star, XCircle,
} from 'lucide-react';
import PredictionForm, { DEFAULT_INPUTS } from './PredictionForm';
import PageHeader, { InfoBanner, ResultsPlaceholder } from './PageHeader';
import { getModelComparison, humanizeApiError } from '../services/apiService';
import RocEvaluationSection from './RocEvaluationSection';

const VARIANT_STYLE = {
  baseline: { color: '#dc2626', name: 'Baseline' },
  feature_engineering: { color: '#2563eb', name: 'Feature Eng.' },
  advanced: { color: '#16a34a', name: 'Advanced' },
};

function PredictedValue({ model, task }) {
  if (!model.prediction) {
    return (
      <div className="flex items-center gap-1.5 text-xs text-amber-600">
        <XCircle size={13} />
        {model.error || (model.variant === 'advanced' && task === 'crop_type'
          ? 'Not available'
          : 'Unavailable / guarded')}
      </div>
    );
  }
  const pred = model.prediction;
  if (task === 'crop_type') {
    return (
      <>
        <p className="text-xs text-gray-500">Predicted Crop</p>
        <p className="text-lg font-bold text-gray-900">🌾 {pred.predictedCrop}</p>
        <p className="text-xs text-gray-500 mt-1">
          Confidence: {(pred.confidence * 100).toFixed(1)}%
        </p>
      </>
    );
  }
  if (task === 'yield_level') {
    return (
      <>
        <p className="text-xs text-gray-500">Predicted Yield Level</p>
        <p className="text-lg font-bold text-gray-900">{pred.predictedLevel}</p>
        <p className="text-xs text-gray-500 mt-1">
          P(High) {(pred.probabilityHigh * 100).toFixed(1)}% · P(Low) {(pred.probabilityLow * 100).toFixed(1)}%
        </p>
      </>
    );
  }
  return (
    <>
      <p className="text-xs text-gray-500">Predicted Yield</p>
      <p className="text-lg font-bold text-gray-900">{Number(pred.predictedYield).toFixed(2)} tons</p>
    </>
  );
}

function MetricValue({ task, metricKey, value }) {
  if (value == null) return <span className="text-xs text-gray-400">—</span>;
  let display;
  if (task === 'crop_type' && ['accuracy', 'f1', 'precision', 'recall'].includes(metricKey)) {
    display = `${(value * 100).toFixed(1)}%`;
  } else if (task === 'yield_level' && ['accuracy', 'f1', 'precision', 'recall'].includes(metricKey)) {
    display = `${(value * 100).toFixed(1)}%`;
  } else {
    display = Number(value).toFixed(3);
  }
  return <span className="text-xs font-mono font-bold text-gray-800">{display}</span>;
}

function ComparisonTable({ comparison, task }) {
  const { models, best_model: bestModel, comparison_metrics: comparisonMetrics } = comparison;
  const available = models.filter(m => m.available);
  const bestName = bestModel ? (VARIANT_STYLE[bestModel]?.name || bestModel) : null;

  const metricRows = (comparisonMetrics || []);

  // For radar: only classification metrics (0..1) or use available metric rows that are percentages.
  const radarData = comparisonMetrics
    ?.filter(row => !row.metric.includes('RMSE') && !row.metric.includes('MAE'))
    .map(row => {
      const out = { metric: row.metric };
      for (const v of ['baseline', 'feature_engineering', 'advanced']) {
        if (row[v] != null) out[v] = row[v];
      }
      return out;
    }) || [];

  return (
    <div className="space-y-6">
      {/* Best Model Banner */}
      {bestModel && (
        <div className="flex items-center gap-3 p-4 rounded-xl bg-gradient-to-r from-forest-50 to-earth-50 border border-forest-200">
          <Trophy size={22} className="text-harvest-500" />
          <div>
            <p className="text-sm font-bold text-forest-800">🏆 Best Performing Model: {bestName}</p>
            <p className="text-xs text-forest-600">
              Determined from real backend metrics (Accuracy/F1 or R²).
            </p>
          </div>
        </div>
      )}

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
                    style={{ backgroundColor: VARIANT_STYLE[model.variant]?.color || '#999' }}
                  />
                  <span className="text-sm font-bold text-gray-800">{model.name}</span>
                </div>
                {model.available === false ? (
                  <span className="flex items-center gap-1 text-xs font-bold text-amber-600 bg-amber-50 px-2 py-1 rounded-full">
                    <XCircle size={12} /> Unavailable
                  </span>
                ) : isBest ? (
                  <span className="flex items-center gap-1 text-xs font-bold text-forest-600 bg-forest-50 px-2 py-1 rounded-full">
                    <Star size={12} className="fill-forest-500" /> Best
                  </span>
                ) : null}
              </div>

              {/* Prediction Result */}
              <div className="p-3 rounded-lg bg-gray-50 mb-4">
                <PredictedValue model={model} task={task} />
              </div>

              {/* Metrics */}
              <div className="space-y-2">
                {model.metrics ? (
                  task === 'crop_type' || task === 'yield_level' ? (
                    <>
                      <MetricRow label="Accuracy" value={model.metrics.accuracy} task={task} metricKey="accuracy" />
                      <MetricRow label="F1-Score" value={model.metrics.f1} task={task} metricKey="f1" />
                      <MetricRow label="Precision" value={model.metrics.precision} task={task} metricKey="precision" />
                      <MetricRow label="Recall" value={model.metrics.recall} task={task} metricKey="recall" />
                    </>
                  ) : (
                    <>
                      <MetricRow label="R² Score" value={model.metrics.r2} task={task} metricKey="r2" />
                      <MetricRow label="RMSE" value={model.metrics.rmse} task={task} metricKey="rmse" />
                      <MetricRow label="MAE" value={model.metrics.mae} task={task} metricKey="mae" />
                    </>
                  )
                ) : (
                  <p className="text-xs text-gray-400">No metrics available.</p>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Comparison Charts */}
      {available.length > 1 && (
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Bar Chart Comparison */}
          <div className="glass-card p-5">
            <h4 className="text-sm font-semibold text-gray-800 mb-4">
              Metrics Comparison (Bar Chart)
            </h4>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={metricRows} barCategoryGap="20%">
                  <XAxis dataKey="metric" tick={{ fontSize: 10, fill: '#374151' }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 10, fill: '#9ca3af' }} axisLine={false} tickLine={false} domain={[0, 'auto']} />
                  <Tooltip contentStyle={{ borderRadius: '12px', border: '1px solid #e5e7eb', fontSize: '12px' }} />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  {available.map(m => (
                    <Bar key={m.variant} dataKey={m.variant} name={VARIANT_STYLE[m.variant]?.name || m.variant}
                      fill={VARIANT_STYLE[m.variant]?.color || '#999'} radius={[4, 4, 0, 0]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Radar Chart Comparison */}
          <div className="glass-card p-5">
            <h4 className="text-sm font-semibold text-gray-800 mb-4">Performance Radar</h4>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="78%">
                  <PolarGrid stroke="#e5e7eb" />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: '#6b7280' }} />
                  <PolarRadiusAxis angle={90} domain={[0, 1]} tick={{ fontSize: 9, fill: '#9ca3af' }} />
                  {available.map(m => (
                    <Radar key={m.variant} name={VARIANT_STYLE[m.variant]?.name || m.variant}
                      dataKey={m.variant} stroke={VARIANT_STYLE[m.variant]?.color || '#999'}
                      fill={VARIANT_STYLE[m.variant]?.color || '#999'} fillOpacity={0.1} strokeWidth={2} />
                  ))}
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Detailed Comparison Table */}
      <div className="glass-card p-5 overflow-x-auto">
        <h4 className="text-sm font-semibold text-gray-800 mb-4">Detailed Metrics Comparison Table</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200">
              <th className="text-left py-3 px-4 font-semibold text-gray-700">Metric</th>
              {models.map(m => (
                <th key={m.variant} className="text-center py-3 px-4">
                  <div className="flex items-center justify-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: VARIANT_STYLE[m.variant]?.color || '#999' }} />
                    <span className="font-semibold text-gray-700">{m.name}</span>
                    {m.available === false && <XCircle size={12} className="text-amber-500" />}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metricRows.map((row, i) => (
              <tr key={i} className="border-b border-gray-50 hover:bg-gray-50 transition-colors">
                <td className="py-3 px-4 font-medium text-gray-700">{row.metric}</td>
                {models.map(m => (
                  <td key={m.variant} className="py-3 px-4 text-center">
                    {row[m.variant] == null ? (
                      <span className="text-xs text-gray-400">—</span>
                    ) : (
                      <span className={`font-mono font-semibold ${
                        Number(row[m.variant]) === bestCellValue(metricRows, row, m)
                          ? 'text-forest-600 bg-forest-50 px-2 py-1 rounded-md' : 'text-gray-600'
                      }`}>
                        {formatMetricCell(row.metric, row[m.variant])}
                      </span>
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function bestCellValue(rows, row, model) {
  const vals = rows.map(r => r[model.variant]).filter(v => v != null);
  if (vals.length === 0) return null;
  const lowerIsBetter = row.metric === 'RMSE' || row.metric === 'MAE';
  return lowerIsBetter ? Math.min(...vals) : Math.max(...vals);
}

function formatMetricCell(metric, value) {
  if (metric === 'RMSE' || metric === 'MAE') return Number(value).toFixed(3);
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function MetricRow({ label, value, task, metricKey }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-gray-50">
      <span className="text-xs text-gray-600">{label}</span>
      <MetricValue task={task} metricKey={metricKey} value={value} />
    </div>
  );
}

const TASKS = [
  { id: 'crop_type', label: 'Crop Type' },
  { id: 'yield_level', label: 'Yield Level' },
  { id: 'crop_yield', label: 'Crop Yield' },
];

export default function ModelComparison() {
  const [inputs, setInputs] = useState({ ...DEFAULT_INPUTS });
  const [modelVariant, setModelVariant] = useState('feature_engineering');
  const [task, setTask] = useState('crop_type');
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleCompare = useCallback(async () => {
    setLoading(true);
    setComparison(null);
    setError(null);
    try {
      const result = await getModelComparison(inputs, task);
      setComparison(result);
    } catch (e) {
      setError(humanizeApiError(e));
      setComparison(null);
    } finally {
      setLoading(false);
    }
  }, [inputs, task]);

  return (
    <div className="page-shell-wide">
      <PageHeader
        icon={GitCompare}
        title="Model Comparison"
        subtitle="Run Baseline, Feature Engineering, and Advanced models on the same inputs."
        gradient="from-blue-500 to-indigo-500"
      />

      <InfoBanner icon={Zap} title="Compare all models" tone="blue">
        Predictions and metrics come from the backend. Unavailable variants are marked so you
        can see which pipelines actually ran.
      </InfoBanner>

      {error && (
        <InfoBanner icon={AlertCircle} title="Comparison failed" tone="red">
          {error}
        </InfoBanner>
      )}

      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Comparison task">
        {TASKS.map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            aria-selected={task === t.id}
            onClick={() => { setTask(t.id); setComparison(null); }}
            className={`px-5 py-2.5 rounded-xl text-sm font-semibold transition-all ${
              task === t.id
                ? 'bg-forest-600 text-white shadow-sm'
                : 'bg-white text-gray-600 border border-gray-200 hover:bg-gray-50'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <PredictionForm
        inputs={inputs}
        setInputs={setInputs}
        modelVariant={modelVariant}
        setModelVariant={setModelVariant}
        onPredict={handleCompare}
        loading={loading}
        showCropType={task !== 'crop_type'}
        hideVariants
        taskLabel="Compare All Models"
        task={task}
      />

      <div className="space-y-6" aria-live="polite">
        {loading && (
          <div className="glass-card p-10 flex flex-col items-center justify-center min-h-[200px]">
            <div className="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mb-4" />
            <p className="text-sm font-medium text-gray-600">Running all models...</p>
          </div>
        )}
        {comparison && !loading && <ComparisonTable comparison={comparison} task={task} />}
        {!comparison && !loading && (
          <ResultsPlaceholder
            title="No comparison yet"
            hint="Choose a task and conditions above, then compare all models. Side-by-side cards, charts, and ROC curves will use the full page width."
          />
        )}
        {task === 'crop_type' && !loading && <RocEvaluationSection />}
      </div>
    </div>
  );
}
