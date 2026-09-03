import React, { useEffect, useMemo, useState } from 'react';
import {
  LineChart, Line, ResponsiveContainer, XAxis, YAxis, Tooltip,
  CartesianGrid, Legend,
} from 'recharts';
import { AlertCircle, Download, Loader2 } from 'lucide-react';
import { getRocEvaluation, humanizeApiError } from '../services/apiService';

const VARIANT_COLORS = {
  baseline: '#dc2626',
  feature_engineering: '#2563eb',
};

const VARIANT_NAMES = {
  baseline: 'Baseline',
  feature_engineering: 'Feature Eng.',
};

export default function RocEvaluationSection() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedClass, setSelectedClass] = useState(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    Promise.all([getRocEvaluation('baseline'), getRocEvaluation('feature_engineering')])
      .then(([baseline, fe]) => {
        if (!active) return;
        setData({ baseline, fe });
        const classes = baseline.report?.class_names || [];
        setSelectedClass((prev) => prev || classes[0] || null);
        setLoading(false);
      })
      .catch((e) => {
        if (!active) return;
        setError(humanizeApiError(e));
        setLoading(false);
      });
    return () => { active = false; };
  }, []);

  const models = useMemo(() => {
    if (!data) return [];
    return (data.baseline.summary?.models || []).map((m) => ({
      variant: m.variant,
      name: VARIANT_NAMES[m.variant] || m.variant,
      macro: m.macro_roc_auc,
      weighted: m.weighted_roc_auc,
    }));
  }, [data]);

  const perClassRows = useMemo(() => {
    if (!data) return [];
    const base = data.baseline.report?.class_auc || {};
    const fe = data.fe.report?.class_auc || {};
    const names = Object.keys(base).length ? Object.keys(base) : Object.keys(fe);
    if (!names.length) return [];
    return names.map((cls) => ({
      cls,
      baselineAuc: base[cls]?.auc ?? null,
      feAuc: fe[cls]?.auc ?? null,
      support: base[cls]?.test_support ?? null,
    }));
  }, [data]);

  const chartData = useMemo(() => {
    if (!data || !selectedClass) return [];
    const points = data.baseline.report?.curves?.[selectedClass];
    if (!points || !points.fpr.length) return [];
    return points.fpr.map((fpr, i) => ({
      fpr,
      baseline_tpr: points.tpr[i] ?? null,
      fe_tpr: data.fe.report?.curves?.[selectedClass]?.tpr?.[i] ?? null,
    }));
  }, [data, selectedClass]);

  if (loading) {
    return (
      <div className="glass-card p-8 flex flex-col items-center justify-center gap-3">
        <Loader2 size={26} className="text-blue-500 animate-spin" />
        <p className="text-sm text-gray-500">Loading evaluation...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-start gap-3 p-4 rounded-xl bg-red-50 border border-red-200">
        <AlertCircle size={18} className="text-red-500 mt-0.5 shrink-0" />
        <div>
          <p className="text-sm font-medium text-red-700">ROC / AUC evaluation unavailable</p>
          <p className="text-xs text-red-600 mt-0.5">{error}</p>
        </div>
      </div>
    );
  }

  const selected = perClassRows.find((r) => r.cls === selectedClass) || {};

  return (
    <div className="space-y-5">
      {/* Header */}
      <div>
        <h3 className="text-lg font-bold text-gray-900">ROC Curve &amp; AUC</h3>
        <p className="text-sm font-medium text-gray-500">Crop Type</p>
        <p className="text-xs text-gray-400 mt-1">990 test samples · 2022–2023 · One-vs-Rest</p>
      </div>

      {/* Evaluation note */}
      <p className="text-xs text-gray-500 leading-relaxed">
        <span className="font-semibold text-gray-600">Evaluation note:</span> ROC curves are
        generated from the deployed model artifacts. Chapter 4 AUC values are based on a
        separate model run with the same configuration, so minor differences may occur.
      </p>

      {/* Summary comparison */}
      <div className="grid sm:grid-cols-2 gap-4">
        {models.map((m) => (
          <div key={m.variant} className="glass-card p-4">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: VARIANT_COLORS[m.variant] }} />
              <span className="text-sm font-semibold text-gray-800">{m.name}</span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Metric label="Macro AUC" value={m.macro} />
              <Metric label="Weighted AUC" value={m.weighted} />
            </div>
          </div>
        ))}
      </div>

      {/* ROC chart with class selector */}
      <div className="glass-card p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h4 className="text-sm font-semibold text-gray-800">
            ROC Curve — <span className="text-blue-600">{selectedClass || '—'}</span>
          </h4>
          <select
            value={selectedClass || ''}
            onChange={(e) => setSelectedClass(e.target.value)}
            className="px-3 py-2 border border-gray-200 rounded-lg text-sm bg-white focus:ring-2 focus:ring-blue-300 focus:outline-none"
          >
            {Object.keys(data?.baseline?.report?.curves || {}).map((cls) => (
              <option key={cls} value={cls}>{cls}</option>
            ))}
          </select>
        </div>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="fpr" type="number" domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)}
                tick={{ fontSize: 11, fill: '#6b7280' }} label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5, fontSize: 11, fill: '#6b7280' }} />
              <YAxis type="number" domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)}
                tick={{ fontSize: 11, fill: '#6b7280' }} label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', offset: 10, fontSize: 11, fill: '#6b7280' }} />
              <Tooltip contentStyle={{ borderRadius: '12px', border: '1px solid #e5e7eb', fontSize: '12px' }}
                formatter={(value, name) => [Number(value).toFixed(3), name === 'baseline_tpr' ? 'Baseline TPR' : 'FE TPR']} />
              <Legend wrapperStyle={{ fontSize: '11px' }} />
              <Line type="monotone" dataKey="baseline_tpr" name="Baseline" stroke={VARIANT_COLORS.baseline}
                strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="fe_tpr" name="Feature Engineering" stroke={VARIANT_COLORS.feature_engineering}
                strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <p className="text-xs text-gray-500 mt-3">
          Class AUC — Baseline: {fmtAuc(selected.baselineAuc)} · Feature Eng.: {fmtAuc(selected.feAuc)}
        </p>
      </div>

      {/* Per-class AUC table */}
      <div className="glass-card p-5 overflow-x-auto">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-sm font-semibold text-gray-800">Per-Class AUC</h4>
          <button
            onClick={() => exportCsv(perClassRows)}
            className="flex items-center gap-1.5 text-xs font-semibold text-blue-600 px-3 py-1.5 rounded-lg border border-blue-200 hover:bg-blue-50 transition-colors"
          >
            <Download size={13} /> Export CSV
          </button>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200">
              <th className="text-left py-3 px-4 font-semibold text-gray-700">Crop Type</th>
              <th className="text-center py-3 px-4 font-semibold text-gray-700">Baseline AUC</th>
              <th className="text-center py-3 px-4 font-semibold text-gray-700">Feature Eng. AUC</th>
              <th className="text-center py-3 px-4 font-semibold text-gray-700">Test Support</th>
            </tr>
          </thead>
          <tbody>
            {perClassRows.map((r) => {
              const best = bestAuc(r.baselineAuc, r.feAuc);
              return (
                <tr key={r.cls} className="border-b border-gray-50 hover:bg-gray-50 transition-colors">
                  <td className="py-2.5 px-4 font-medium text-gray-700">{r.cls}</td>
                  <td className="py-2.5 px-4 text-center">
                    <AucCell value={r.baselineAuc} isBest={best === 'baseline'} />
                  </td>
                  <td className="py-2.5 px-4 text-center">
                    <AucCell value={r.feAuc} isBest={best === 'fe'} />
                  </td>
                  <td className="py-2.5 px-4 text-center text-gray-500">{r.support ?? '—'}</td>
                </tr>
              );
            })}
            {perClassRows.length === 0 && (
              <tr><td colSpan="4" className="py-6 text-center text-sm text-gray-400">No per-class AUC data available.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="p-2 rounded-lg bg-gray-50">
      <p className="text-[11px] text-gray-500">{label}</p>
      <p className="text-base font-bold font-mono text-gray-900">
        {value == null ? '—' : Number(value).toFixed(4)}
      </p>
    </div>
  );
}

function AucCell({ value, isBest }) {
  if (value == null) return <span className="text-xs text-gray-400">—</span>;
  return (
    <span className={`font-mono font-semibold text-xs ${
      isBest ? 'bg-forest-50 text-forest-700 px-2 py-1 rounded-md' : 'text-gray-600'
    }`}>
      {Number(value).toFixed(4)}
    </span>
  );
}

function bestAuc(a, b) {
  if (a == null && b == null) return null;
  if (a == null) return 'fe';
  if (b == null) return 'baseline';
  if (a === b) return null;
  return a > b ? 'baseline' : 'fe';
}

function fmtAuc(v) {
  return v == null ? '—' : Number(v).toFixed(4);
}

function exportCsv(rows) {
  const header = 'Crop Type,Baseline AUC,Feature Engineering AUC,Test Support';
  const lines = rows.map((r) => [
    r.cls,
    r.baselineAuc == null ? '' : Number(r.baselineAuc).toFixed(6),
    r.feAuc == null ? '' : Number(r.feAuc).toFixed(6),
    r.support ?? '',
  ].join(','));
  const blob = new Blob([[header, ...lines].join('\n')], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'crop_type_roc_auc_comparison.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}