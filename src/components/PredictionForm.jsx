import React, { useId } from 'react';
import {
  MapPin, Layers, Droplets, Thermometer, CloudRain,
  Wind, Calendar, Maximize2, RotateCcw, AlertTriangle, Sprout, ChevronDown,
} from 'lucide-react';
import {
  REGIONS, SOIL_TYPES, WATER_SOURCES, CROP_TYPES,
  SEEDING_SEASONS, MODEL_VARIANTS,
} from '../services/apiService';

const DATASET_RANGES = {
  avgTemperature: { min: 21.06, max: 29.19, unit: '°C' },
  totalRainfall: { min: 270.44, max: 10036.74, unit: 'mm' },
  avgHumidity: { min: 70.05, max: 93, unit: '%' },
  year: { min: 2012, max: 2023, unit: '' },
};

const DEFAULT_INPUTS = {
  region: 'Mandalay',
  soilType: 'Acrisols / Ferralsols / Red Earth',
  waterSource: 'Rainfed',
  seedingSeason: 'Rainy',
  avgTemperature: 28,
  totalRainfall: 1000,
  avgHumidity: 80,
  year: 2023,
  sownAcre: 200,
  cropType: 'Paddy',
};

function SelectField({ id, label, icon: Icon, value, onChange, options }) {
  return (
    <div className="space-y-1.5 min-w-0">
      <label htmlFor={id} className="flex items-center gap-1.5 text-xs font-semibold text-gray-500 uppercase tracking-wide">
        <Icon size={13} className="text-forest-600" aria-hidden="true" />
        {label}
      </label>
      <div className="relative">
        <select
          id={id}
          value={value}
          onChange={e => onChange(e.target.value)}
          className="select-field"
          title={value}
        >
          <option value="">Select {label}</option>
          {options.map(opt => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
        <ChevronDown size={16} className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-gray-400" aria-hidden="true" />
      </div>
    </div>
  );
}

function ChipGroup({ label, icon: Icon, value, onChange, options }) {
  return (
    <div className="space-y-1.5 min-w-0">
      <p className="flex items-center gap-1.5 text-xs font-semibold text-gray-500 uppercase tracking-wide">
        <Icon size={13} className="text-forest-600" aria-hidden="true" />
        {label}
      </p>
      <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label={label}>
        {options.map(opt => {
          const on = value === opt;
          return (
            <button
              key={opt}
              type="button"
              role="radio"
              aria-checked={on}
              onClick={() => onChange(opt)}
              className={`chip-toggle ${on ? 'chip-toggle-on' : 'chip-toggle-off'}`}
            >
              {opt}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function ClimateTile({
  id, label, icon: Icon, value, onChange, min, max, step, unit, typicalKey, formatValue,
}) {
  const typical = typicalKey ? DATASET_RANGES[typicalKey] : null;
  const outside = typical && (value < typical.min || value > typical.max);
  const display = formatValue ? formatValue(value) : value;

  return (
    <div className={`rounded-xl border p-4 space-y-3 min-w-0 ${
      outside ? 'border-amber-200 bg-amber-50/40' : 'border-gray-100 bg-gray-50/70'
    }`}>
      <div className="flex items-start justify-between gap-2">
        <label htmlFor={id} className="flex items-center gap-1.5 text-sm font-medium text-gray-700">
          <Icon size={15} className="text-forest-600" aria-hidden="true" />
          {label}
        </label>
        <div className="flex items-center gap-1 shrink-0">
          <input
            id={`${id}-num`}
            type="number"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={e => onChange(Number(e.target.value))}
            className="w-[5.5rem] px-2 py-1 text-right text-sm font-mono font-semibold text-forest-700 bg-white border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-forest-400"
            aria-label={`${label} value`}
          />
          <span className="text-xs text-gray-500 w-7">{unit}</span>
        </div>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={Number.isFinite(value) ? value : min}
        onChange={e => onChange(Number(e.target.value))}
        className="form-range"
      />
      <div className="flex justify-between text-[11px] text-gray-400">
        <span>{min}{unit}</span>
        {typical && !outside && (
          <span>Typical {typical.min}–{typical.max}{typical.unit}</span>
        )}
        {outside && (
          <span className="text-amber-700 font-medium">Outside typical {typical.min}–{typical.max}{typical.unit}</span>
        )}
        <span>{max}{unit}</span>
      </div>
      <span className="sr-only">{display}{unit}</span>
    </div>
  );
}

export default function PredictionForm({
  inputs,
  setInputs,
  modelVariant,
  setModelVariant,
  onPredict,
  loading,
  showCropType = false,
  hideVariants = false,
  taskLabel = 'Predict',
  task = 'crop_type',
  availability = null,
}) {
  const uid = useId();
  const handleReset = () => setInputs({ ...DEFAULT_INPUTS });

  const updateField = (field, value) => {
    setInputs(prev => ({ ...prev, [field]: value }));
  };

  const variantOptions = MODEL_VARIANTS[task] || MODEL_VARIANTS.crop_type;
  const chosenVariantUnavailable =
    availability && modelVariant in availability && availability[modelVariant] === false;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (loading || chosenVariantUnavailable) return;
    onPredict();
  };

  const summary = [
    inputs.region,
    inputs.seedingSeason,
    inputs.waterSource,
    showCropType ? inputs.cropType : null,
    inputs.year,
    `${inputs.avgTemperature}°C`,
    `${inputs.totalRainfall} mm`,
    `${inputs.sownAcre} ac`,
  ].filter(Boolean).join(' · ');

  return (
    <form onSubmit={handleSubmit} className="glass-card overflow-hidden">
      <div className="p-5 sm:p-6 space-y-6">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">Field conditions</h3>
            <p className="text-sm text-gray-500 mt-0.5">
              Set location and climate, then run the model. Values can go beyond the training range.
            </p>
          </div>
          <button
            type="button"
            onClick={handleReset}
            className="flex items-center gap-1.5 text-xs font-medium text-gray-500 hover:text-forest-600 border border-gray-200 hover:border-forest-200 rounded-lg px-3 py-1.5 transition-colors shrink-0"
          >
            <RotateCcw size={13} />
            Reset
          </button>
        </div>

        {!hideVariants && (
          <div className="space-y-2">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Model variant</p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3" role="radiogroup" aria-label="Model variant">
              {variantOptions.map(variant => {
                const available = availability ? availability[variant.id] !== false : !variant.guarded;
                const selected = modelVariant === variant.id;
                const disabled = available === false;
                return (
                  <button
                    key={variant.id}
                    type="button"
                    role="radio"
                    aria-checked={selected}
                    disabled={disabled}
                    onClick={() => setModelVariant(variant.id)}
                    className={`p-4 rounded-xl border text-left transition-all duration-200 ${
                      selected
                        ? 'border-forest-500 bg-forest-50 ring-1 ring-forest-200'
                        : disabled
                          ? 'border-gray-100 bg-gray-50 opacity-60 cursor-not-allowed'
                          : 'border-gray-200 bg-white hover:border-forest-200'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: variant.color }} />
                      <span className="text-sm font-semibold text-gray-800">{variant.name}</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1.5 leading-relaxed">{variant.description}</p>
                    {disabled && (
                      <p className="flex items-center gap-1 text-[11px] font-medium text-amber-600 mt-2">
                        <AlertTriangle size={11} /> Unavailable
                      </p>
                    )}
                  </button>
                );
              })}
            </div>
            {chosenVariantUnavailable && (
              <p className="flex items-center gap-1.5 text-xs font-medium text-amber-700 bg-amber-50 border border-amber-100 rounded-lg px-3 py-2">
                <AlertTriangle size={13} />
                This model variant is not available. Choose another variant to continue.
              </p>
            )}
          </div>
        )}

        <div className="space-y-3">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Location</p>
          <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-4">
            <SelectField
              id={`${uid}-region`}
              label="Region"
              icon={MapPin}
              value={inputs.region}
              onChange={v => updateField('region', v)}
              options={REGIONS}
            />
            <SelectField
              id={`${uid}-soil`}
              label="Soil type"
              icon={Layers}
              value={inputs.soilType}
              onChange={v => updateField('soilType', v)}
              options={SOIL_TYPES}
            />
            {showCropType && (
              <SelectField
                id={`${uid}-crop`}
                label="Crop type"
                icon={Sprout}
                value={inputs.cropType}
                onChange={v => updateField('cropType', v)}
                options={CROP_TYPES}
              />
            )}
          </div>
          <div className="grid sm:grid-cols-2 gap-4 pt-1">
            <ChipGroup
              label="Water source"
              icon={Droplets}
              value={inputs.waterSource}
              onChange={v => updateField('waterSource', v)}
              options={WATER_SOURCES}
            />
            <ChipGroup
              label="Seeding season"
              icon={Calendar}
              value={inputs.seedingSeason}
              onChange={v => updateField('seedingSeason', v)}
              options={SEEDING_SEASONS}
            />
          </div>
        </div>

        <div className="space-y-3">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Climate and field</p>
          <div className="grid sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-5 gap-3">
            <ClimateTile
              id={`${uid}-temp`}
              label="Temperature"
              icon={Thermometer}
              value={inputs.avgTemperature}
              onChange={v => updateField('avgTemperature', v)}
              min={10} max={45} step={0.1} unit="°C"
              typicalKey="avgTemperature"
            />
            <ClimateTile
              id={`${uid}-rain`}
              label="Rainfall"
              icon={CloudRain}
              value={inputs.totalRainfall}
              onChange={v => updateField('totalRainfall', v)}
              min={0} max={15000} step={10} unit="mm"
              typicalKey="totalRainfall"
            />
            <ClimateTile
              id={`${uid}-humidity`}
              label="Humidity"
              icon={Wind}
              value={inputs.avgHumidity}
              onChange={v => updateField('avgHumidity', v)}
              min={30} max={100} step={1} unit="%"
              typicalKey="avgHumidity"
            />
            <ClimateTile
              id={`${uid}-year`}
              label="Year"
              icon={Calendar}
              value={inputs.year}
              onChange={v => updateField('year', Math.round(v))}
              min={2000} max={2030} step={1} unit=""
              typicalKey="year"
            />
            <ClimateTile
              id={`${uid}-acre`}
              label="Sown area"
              icon={Maximize2}
              value={inputs.sownAcre}
              onChange={v => updateField('sownAcre', Math.round(v))}
              min={0} max={1000000} step={50} unit="ac"
            />
          </div>
        </div>
      </div>

      <div className="px-5 sm:px-6 py-4 bg-gray-50/80 border-t border-gray-100 flex flex-col lg:flex-row lg:items-center gap-3">
        <button
          type="submit"
          disabled={loading || !inputs.region || !inputs.soilType || !inputs.waterSource || !inputs.seedingSeason || chosenVariantUnavailable}
          className="btn-primary w-full sm:w-auto flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <Sprout size={18} />
              {taskLabel}
            </>
          )}
        </button>
        <p className="text-xs text-gray-500 leading-relaxed min-w-0 lg:flex-1">
          {summary}
        </p>
      </div>
    </form>
  );
}

export { DEFAULT_INPUTS };
