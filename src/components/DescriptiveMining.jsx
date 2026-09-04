import React, { useState, useEffect } from 'react';
import {
  GitBranch, TrendingUp, Activity, AlertCircle,
  Loader2, GitFork, Database, ListTree, ArrowRight,
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';
import { getDescriptiveMining, humanizeApiError } from '../services/apiService';
import PageHeader, { InfoBanner } from './PageHeader';

const CHART_COLORS = ['#2d9f63', '#e5a03c', '#2563eb', '#8b5cf6', '#06b6d4', '#dc2626'];

const COMPONENT_TABS = [
  { id: 'correlation', label: 'Correlation Analysis', icon: Activity },
  { id: 'association', label: 'Association Rule Mining', icon: GitBranch },
  { id: 'frequent', label: 'Frequent Pattern Mining', icon: ListTree },
  { id: 'sequential', label: 'Sequential Pattern Mining', icon: TrendingUp },
];

function fmt(v, digits = 3) {
  if (v == null) return '—';
  return Number(v).toFixed(digits);
}

function pct(v, digits = 2) {
  if (v == null) return '—';
  return `${(Number(v) * 100).toFixed(digits)}%`;
}

// ─── Correlation ───────────────────────────────────────────────────────────

function CorrelationSection({ correlation }) {
  const { columns, matrix } = correlation;
  if (!columns || !matrix) {
    return <p className="text-sm text-gray-500">Correlation data is not available.</p>;
  }

  const heatColor = r => {
    const a = Math.abs(r);
    if (a > 0.85) return r > 0 ? 'bg-forest-700' : 'bg-red-700';
    if (a > 0.6) return r > 0 ? 'bg-forest-500' : 'bg-red-500';
    if (a > 0.3) return r > 0 ? 'bg-forest-300' : 'bg-red-300';
    return r > 0 ? 'bg-forest-100' : 'bg-red-100';
  };

  return (
    <div className="glass-card p-5">
      <div className="flex items-center gap-2 mb-3">
        <Activity size={16} className="text-forest-500" />
        <h4 className="text-sm font-semibold text-gray-800">Crop Correlation Analysis</h4>
      </div>

      <div className="overflow-x-auto">
        <table className="border-collapse text-center text-xs">
          <thead>
            <tr>
              <th className="p-2 text-gray-400 font-normal"></th>
              {columns.map(c => (
                <th key={c} className="p-1.5 font-medium text-gray-600 text-[10px] max-w-[70px] truncate">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map(row => (
              <tr key={row.feature}>
                <th className="p-1.5 pr-2 font-medium text-gray-600 text-[10px] text-left max-w-[110px] truncate">
                  {row.feature}
                </th>
                {row.values.map(v => {
                  const abs = Math.abs(v.r);
                  return (
                    <td key={v.other} className="p-1.5">
                      <div
                        className={`w-12 h-7 rounded flex items-center justify-center text-[9px] font-semibold text-white ${heatColor(v.r)}`}
                        title={`${row.feature} ↔ ${v.other} = ${v.r.toFixed(3)}`}
                      >
                        {abs >= 0.3 ? v.r.toFixed(2) : ''}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-4 rounded-xl bg-gray-50 p-3">
        <p className="text-xs font-semibold text-gray-700 mb-1">Heatmap scale</p>
        <div className="flex items-center gap-1 text-[10px] text-gray-500 flex-wrap">
          <span className="inline-block w-3 h-3 rounded bg-forest-700" /> +0.85+
          <span className="inline-block w-3 h-3 rounded bg-forest-500" /> +0.60
          <span className="inline-block w-3 h-3 rounded bg-forest-300" /> +0.30
          <span className="inline-block w-3 h-3 rounded bg-forest-100" /> low
          <span className="inline-block w-3 h-3 rounded bg-red-100" /> −low
          <span className="inline-block w-3 h-3 rounded bg-red-300" /> −0.30
          <span className="inline-block w-3 h-3 rounded bg-red-500" /> −0.60
          <span className="inline-block w-3 h-3 rounded bg-red-700" /> −0.85
        </div>
      </div>
    </div>
  );
}

// ─── Association Rules ─────────────────────────────────────────────────────

function RuleTable({ rules }) {
  if (!rules || rules.length === 0) return <p className="text-sm text-gray-500">No rules to display.</p>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="text-xs text-gray-500 border-b border-gray-200">
            <th className="py-2 pr-3 font-medium">#</th>
            <th className="py-2 pr-3 font-medium">Antecedent</th>
            <th className="py-2 pr-3 font-medium">Consequent</th>
            <th className="py-2 pr-3 font-medium text-right">Support</th>
            <th className="py-2 pr-3 font-medium text-right">Confidence</th>
            <th className="py-2 pr-3 font-medium text-right">Lift</th>
          </tr>
        </thead>
        <tbody>
          {rules.map((r, i) => (
            <tr key={i} className="border-b border-gray-50 last:border-0">
              <td className="py-2 pr-3 text-gray-400">{i + 1}</td>
              <td className="py-2 pr-3 font-medium text-gray-700">
                <div className="flex flex-wrap gap-1">
                  {(r.antecedent || []).map((a, j) => (
                    <span key={j} className="inline-flex items-center gap-1 text-[11px] bg-gray-100 px-1.5 py-0.5 rounded">
                      <Database size={10} className="text-gray-400" /> {a}
                    </span>
                  ))}
                </div>
              </td>
              <td className="py-2 pr-3 text-gray-600">
                <div className="flex flex-wrap gap-1">
                  {(r.consequent || []).map((c, j) => (
                    <span key={j} className="inline-flex items-center gap-1 text-[11px] bg-forest-50 text-forest-700 px-1.5 py-0.5 rounded">
                      {c}
                    </span>
                  ))}
                </div>
              </td>
              <td className="py-2 pr-3 text-right text-gray-600">{pct(r.support)}</td>
              <td className="py-2 pr-3 text-right text-gray-600">{pct(r.confidence, 1)}</td>
              <td className="py-2 pr-3 text-right font-semibold text-gray-700">{fmt(r.lift)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RulesCharts({ rules }) {
  if (!rules || rules.length === 0) return null;
  const compare = rules.map(r => ({
    name: (r.consequent || []).map(c => c.split(':')[0]).join('+'),
    confidence: r.confidence,
    lift: r.lift,
    support: r.support,
  })).slice(0, 8);

  return (
    <div className="grid lg:grid-cols-3 gap-4">
      {[
        { key: 'confidence', label: 'Confidence', color: '#2d9f63', unit: 'pct' },
        { key: 'lift', label: 'Lift', color: '#2563eb', unit: 'raw' },
        { key: 'support', label: 'Support', color: '#e5a03c', unit: 'pct' },
      ].map(({ key, label, color, unit }) => (
        <div key={key} className="glass-card p-5">
          <h5 className="text-sm font-semibold text-gray-800 mb-3">Rule {label} (top 8)</h5>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={compare} layout="vertical" margin={{ left: 8 }}>
                <XAxis
                  type="number"
                  domain={[0, unit === 'raw' ? 'auto' : 1]}
                  tick={{ fontSize: 10, fill: '#9ca3af' }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={v => (unit === 'raw' ? v : `${(v * 100).toFixed(0)}%`)}
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  tick={{ fontSize: 10, fill: '#374151' }}
                  axisLine={false}
                  tickLine={false}
                  width={80}
                />
                <Tooltip
                  formatter={v => [unit === 'raw' ? Number(v).toFixed(2) : `${(v * 100).toFixed(2)}%`, label]}
                  contentStyle={{ borderRadius: '12px', border: '1px solid #e5e7eb', fontSize: '12px' }}
                />
                <Bar dataKey={key} radius={[0, 6, 6, 0]} fill={color} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      ))}
    </div>
  );
}

function AssociationSection({ association }) {
  const { yield_level, crop_type } = association;
  return (
    <div className="space-y-4">
      <div className="glass-card p-5">
        <div className="flex items-center gap-2 mb-2">
          <GitBranch size={16} className="text-forest-500" />
          <h4 className="text-sm font-semibold text-gray-800">Yield Level Rules</h4>
          <span className="ml-auto text-[11px] font-medium text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
            min support {pct(yield_level.min_support ?? 0.05, 1)} · min conf {pct(yield_level.min_confidence ?? 0.6, 0)} · {yield_level.rule_count} rules
          </span>
        </div>
        <RuleTable rules={yield_level.rules} />
      </div>

      <div className="glass-card p-5">
        <div className="flex items-center gap-2 mb-2">
          <ArrowRight size={16} className="text-forest-500" />
          <h4 className="text-sm font-semibold text-gray-800">Crop Type Rules</h4>
          <span className="ml-auto text-[11px] font-medium text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
            min support {pct(crop_type.min_support ?? 0.03, 1)} · min conf {pct(crop_type.min_confidence ?? 0.6, 0)} · {crop_type.rule_count} rules
          </span>
        </div>
        <RuleTable rules={crop_type.rules} />
      </div>

      {yield_level.rules && yield_level.rules.length > 0 && (
        <RulesCharts rules={yield_level.rules} />
      )}
    </div>
  );
}

// ─── Frequent Patterns ─────────────────────────────────────────────────────

function FrequentCard({ report }) {
  const { analysis, min_support, candidate_counts, frequent_counts, total_candidates, total_frequent, by_size } = report;

  const orderStats = [
    { order: '1-itemset', candidates: candidate_counts?.['1'], frequent: frequent_counts?.['1'] },
    { order: '2-itemset', candidates: candidate_counts?.['2'], frequent: frequent_counts?.['2'] },
    { order: '3-itemset', candidates: candidate_counts?.['3'], frequent: frequent_counts?.['3'] },
  ];

  const chartData = ['1', '2', '3'].map(k => ({
    order: `${k}-itemset`,
    candidates: candidate_counts?.[k] || 0,
    frequent: frequent_counts?.[k] || 0,
  }));

  return (
    <div className="glass-card p-5">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <GitFork size={16} className="text-forest-500" />
          <h4 className="text-sm font-semibold text-gray-800">{analysis}</h4>
        </div>
        <span className="text-[11px] font-medium text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
          min support {pct(min_support, 1)}
        </span>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-4">
        {orderStats.map(({ order, candidates, frequent }) => (
          <div key={order} className="rounded-xl bg-gray-50 p-3 text-center">
            <p className="text-[11px] text-gray-500 font-medium">{order}</p>
            <p className="text-lg font-bold text-gray-800 mt-1">{frequent}</p>
            <p className="text-[11px] text-gray-400">frequent / {candidates} candidates</p>
          </div>
        ))}
      </div>

      <div className="stats-inline mb-4 text-xs text-gray-600">
        Total candidates: <strong>{total_candidates}</strong> &nbsp;·&nbsp; Total frequent: <strong>{total_frequent}</strong>
      </div>

      <div className="h-48 mb-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData}>
            <XAxis dataKey="order" tick={{ fontSize: 11, fill: '#374151' }} axisLine={false} tickLine={false} />
            <YAxis allowDecimals={false} tick={{ fontSize: 11, fill: '#9ca3af' }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ borderRadius: '12px', border: '1px solid #e5e7eb', fontSize: '12px' }} />
            <Bar dataKey="candidates" fill="#e5a03c" radius={[4, 4, 0, 0]} name="Candidates" />
            <Bar dataKey="frequent" fill="#2d9f63" radius={[4, 4, 0, 0]} name="Frequent" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="text-xs font-semibold text-gray-700 mb-2">Top frequent itemsets by size</p>
      {(['1', '2', '3']).map(k => (
        <div key={k} className="mb-3">
          <p className="text-[11px] text-gray-500 font-medium mb-1">{k}-itemsets (top 8)</p>
          <div className="space-y-1">
            {(by_size?.[k] || []).slice(0, 8).map((it, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className="text-gray-700 flex-1 truncate">{it.itemset.join(' · ')}</span>
                <span className="text-gray-500 shrink-0">{pct(it.support)}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function FrequentSection({ frequent_patterns }) {
  if (!frequent_patterns) return null;
  return (
    <div className="space-y-4">
      <FrequentCard report={frequent_patterns.crop_yield} />
      <FrequentCard report={frequent_patterns.crop_type} />
    </div>
  );
}

// ─── Sequential Patterns ───────────────────────────────────────────────────

function SequentialSection({ sequential_patterns }) {
  const { num_sequences, order_2, order_3 } = sequential_patterns;
  const chartData = (order_2?.patterns || []).map(p => ({
    name: p.sequence.join('→'),
    support: p.support,
  }));

  return (
    <div className="space-y-4">
      <div className="glass-card p-5">
        <div className="flex items-center gap-2 mb-3">
          <Activity size={16} className="text-forest-500" />
          <h4 className="text-sm font-semibold text-gray-800">Order-2 patterns</h4>
          <span className="ml-auto text-[11px] font-medium text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
            {num_sequences} sequences
          </span>
        </div>
        {order_2?.patterns?.length ? (
          <div className="grid sm:grid-cols-2 gap-3">
            {order_2.patterns.map((p, i) => (
              <div key={i} className="rounded-xl bg-gray-50 p-3 flex items-center justify-between">
                <span className="text-sm font-semibold text-gray-800">{p.sequence.join(' → ')}</span>
                <div className="text-right">
                  <p className="text-sm font-bold text-forest-700">{pct(p.support)}</p>
                  <p className="text-[10px] text-gray-400">{p.count} of {order_2.total_patterns}</p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-500">No order-2 patterns available.</p>
        )}
      </div>

      <div className="glass-card p-5">
        <div className="flex items-center gap-2 mb-3">
          <TrendingUp size={16} className="text-forest-500" />
          <h4 className="text-sm font-semibold text-gray-800">Order-3 patterns</h4>
        </div>
        {order_3?.patterns?.length ? (
          <div className="grid sm:grid-cols-2 gap-3">
            {order_3.patterns.map((p, i) => (
              <div key={i} className="rounded-xl bg-gray-50 p-3 flex items-center justify-between">
                <span className="text-sm font-semibold text-gray-800">{p.sequence.join(' → ')}</span>
                <div className="text-right">
                  <p className="text-sm font-bold text-forest-700">{pct(p.support)}</p>
                  <p className="text-[10px] text-gray-400">{p.count} of {order_3.total_patterns}</p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-500">No order-3 patterns available.</p>
        )}
      </div>

      {order_2?.patterns?.length ? (
        <div className="glass-card p-5">
          <h5 className="text-sm font-semibold text-gray-800 mb-3">Order-2 support comparison</h5>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#374151' }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 'auto']} tick={{ fontSize: 10, fill: '#9ca3af' }} axisLine={false} tickLine={false} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                <Tooltip formatter={v => [`${(v * 100).toFixed(2)}%`, 'Support']} contentStyle={{ borderRadius: '12px', border: '1px solid #e5e7eb', fontSize: '12px' }} />
                <Bar dataKey="support" radius={[4, 4, 0, 0]}>
                  {chartData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function Cell({ fill: _fill }) {
  return null;
}

// ─── Page ──────────────────────────────────────────────────────────────────

export default function DescriptiveMining() {
  const [data, setData] = useState(null);
  const [state, setState] = useState('loading'); // loading | loaded | error
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('correlation');

  useEffect(() => {
    getDescriptiveMining()
      .then(d => {
        setData(d);
        setState('loaded');
      })
      .catch(err => {
        setError(humanizeApiError(err));
        setState('error');
      });
  }, []);

  return (
    <div className="page-shell">
      <PageHeader
        icon={GitBranch}
        title="Descriptive Mining"
        subtitle="Correlation, association rules, frequent patterns, and sequential patterns."
        gradient="from-amber-500 to-orange-500"
      />

      {/* Loading */}
      {state === 'loading' && (
        <div className="flex flex-col items-center justify-center py-20 text-gray-500">
          <Loader2 size={32} className="animate-spin text-indigo-500 mb-4" />
          <p className="text-sm font-medium">Computing descriptive-mining results...</p>
        </div>
      )}

      {/* Error */}
      {state === 'error' && (
        <InfoBanner icon={AlertCircle} title="Failed to load descriptive-mining results" tone="red">
          {error}
        </InfoBanner>
      )}

      {/* Empty (defensive) */}
      {state === 'loaded' && (!data || !data.methodology) && (
        <div className="flex items-start gap-3 p-4 rounded-xl bg-amber-50 border border-amber-200">
          <AlertCircle size={18} className="text-amber-500 mt-0.5 shrink-0" />
          <p className="text-sm text-amber-700">No descriptive-mining results are currently available.</p>
        </div>
      )}

      {/* Loaded content */}
      {state === 'loaded' && data && data.methodology && (
        <>
          {/* Tabs */}
          <div className="flex gap-2 overflow-x-auto pb-1 -mx-1 px-1" role="tablist" aria-label="Mining methods">
            {COMPONENT_TABS.map(tab => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  role="tab"
                  aria-selected={isActive}
                  onClick={() => setActiveTab(tab.id)}
                  className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium whitespace-nowrap transition-colors ${
                    isActive
                      ? 'bg-forest-600 text-white shadow-sm'
                      : 'bg-white text-gray-600 border border-gray-200 hover:bg-gray-50'
                  }`}
                >
                  <Icon size={16} /> {tab.label}
                </button>
              );
            })}
          </div>

          {/* Tab content */}
          {activeTab === 'correlation' && <CorrelationSection correlation={data.correlation} />}
          {activeTab === 'association' && <AssociationSection association={data.association_rules} />}
          {activeTab === 'frequent' && <FrequentSection frequent_patterns={data.frequent_patterns} />}
          {activeTab === 'sequential' && <SequentialSection sequential_patterns={data.sequential_patterns} />}
        </>
      )}
    </div>
  );
}
