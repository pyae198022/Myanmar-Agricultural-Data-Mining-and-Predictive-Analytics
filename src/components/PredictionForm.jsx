import React from 'react';
import {
  MapPin, Layers, Droplets, Thermometer, CloudRain,
  Wind, Calendar, Maximize2, RotateCcw, AlertTriangle,
} from 'lucide-react';
import {
  REGIONS, SOIL_TYPES, WATER_SOURCES, CROP_TYPES,
  SEEDING_SEASONS, MODEL_VARIANTS,
} from '../services/apiService';

const ICON_COLORS = {
  forest: 'text-green-600',
  blue: 'text-blue-500',
  red: 'text-red-500',
  cyan: 'text-cyan-500',
  harvest: 'text-amber-500',
};

const SelectField = ({ label, icon: Icon, value, onChange, options, color = 'forest' }) => (
  <div className="space-y-1.5">
    <label className="flex items-center gap-1.5 text-sm font-medium text-gray-700">
      <Icon size={14} className={ICON_COLORS[color] || ICON_COLORS.forest} />
      {label}
    </label>
    <select value={value} onChange={e => onChange(e.target.value)} className="input-field">
      <option value="">Select {label}</option>
      {options.map(opt => (
        <option key={opt} value={opt}>{opt}</option>
      ))}
    </select>
  </div>
);

const SliderField = ({ label, icon: Icon, value, onChange, min, max, step, unit, color = 'forest' }) => (
  <div className="space-y-1.5">
    <label className="flex items-center justify-between text-sm font-medium text-gray-700">
      <span className="flex items-center gap-1.5">
        <Icon size={14} className={ICON_COLORS[color] || ICON_COLORS.forest} />
        {label}
      </span>
      <span className="font-mono text-forest-600 bg-forest-50 px-2 py-0.5 rounded-md text-xs">
        {value}{unit}
      </span>
    </label>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={e => onChange(Number(e.target.value))}
      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-forest-500"
    />
    <div className="flex justify-between text-xs text-gray-400">
      <span>{min}{unit}</span>
      <span>{max}{unit}</span>
    </div>
  </div>
);

const NumberField = ({ label, icon: Icon, value, onChange, min, max, step, unit, color = 'forest' }) => (
  <div className="space-y-1.5">
    <label className="flex items-center justify-between text-sm font-medium text-gray-700">
      <span className="flex items-center gap-1.5">
        <Icon size={14} className={ICON_COLORS[color] || ICON_COLORS.forest} />
        {label}
      </span>
      <span className="font-mono text-forest-600 bg-forest-50 px-2 py-0.5 rounded-md text-xs">
        {value}{unit}
      </span>
    </label>
    <input
      type="number"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={e => onChange(Number(e.target.value))}
      className="input-field"
    />
    <div className="flex justify-between text-xs text-gray-400">
      <span>Range: {min}{unit} – {max}{unit}</span>
    </div>
  </div>
);

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
  availability = null, // { baseline, feature_engineering, advanced } booleans
}) {
  const handleReset = () => setInputs({ ...DEFAULT_INPUTS });

  const updateField = (field, value) => {
    setInputs(prev => ({ ...prev, [field]: value }));
  };

  const variantOptions = MODEL_VARIANTS[task] || MODEL_VARIANTS.crop_type;
  const chosenVariantUnavailable =
    availability && modelVariant in availability && availability[modelVariant] === false;

  return (
    <div className="glass-card p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">Input Parameters</h3>
        <button
          onClick={handleReset}
          className="flex items-center gap-1.5 text-xs font-medium text-gray-500 hover:text-forest-600 transition-colors"
        >
          <RotateCcw size={13} />
          Reset
        </button>
      </div>

      {/* Model Variant Selector */}
      {!hideVariants && (
        <div className="space-y-2">
        <p className="text-sm font-medium text-gray-700">Model Variant</p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          {variantOptions.map(variant => {
            const available = availability ? availability[variant.id] !== false : !variant.guarded;
            const selected = modelVariant === variant.id;
            const disabled = available === false;
            return (
              <button
                key={variant.id}
                type="button"
                disabled={disabled}
                onClick={() => setModelVariant(variant.id)}
                className={`p-3 rounded-xl border-2 text-left transition-all duration-200 ${
                  selected
                    ? 'border-forest-500 bg-forest-50 shadow-sm'
                    : disabled
                      ? 'border-gray-100 bg-gray-50 opacity-60 cursor-not-allowed'
                      : 'border-gray-100 bg-white hover:border-gray-200'
                }`}
              >
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: variant.color }}
                  />
                  <span className="text-sm font-semibold text-gray-800">{variant.name}</span>
                </div>
                <p className="text-xs text-gray-500 mt-1 ml-5">{variant.description}</p>
                {disabled && (
                  <p className="flex items-center gap-1 text-[11px] font-medium text-amber-600 mt-1 ml-5">
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
            This model variant is not available. Prediction cannot be run for it.
          </p>
        )}
      </div>
      )}

      {/* Categorical Inputs */}
      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <SelectField
          label="Region"
          icon={MapPin}
          value={inputs.region}
          onChange={v => updateField('region', v)}
          options={REGIONS}
        />
        <SelectField
          label="Soil Type"
          icon={Layers}
          value={inputs.soilType}
          onChange={v => updateField('soilType', v)}
          options={SOIL_TYPES}
        />
        <SelectField
          label="Water Source"
          icon={Droplets}
          value={inputs.waterSource}
          onChange={v => updateField('waterSource', v)}
          options={WATER_SOURCES}
          color="blue"
        />
        <SelectField
          label="Seeding Season"
          icon={Calendar}
          value={inputs.seedingSeason}
          onChange={v => updateField('seedingSeason', v)}
          options={SEEDING_SEASONS}
          color="cyan"
        />
        {showCropType && (
          <SelectField
            label="Crop Type"
            icon={Layers}
            value={inputs.cropType}
            onChange={v => updateField('cropType', v)}
            options={CROP_TYPES}
            color="harvest"
          />
        )}
      </div>

      {/* Numerical Inputs */}
      <div className="grid sm:grid-cols-2 gap-x-6 gap-y-5">
        <SliderField
          label="Avg Temperature"
          icon={Thermometer}
          value={inputs.avgTemperature}
          onChange={v => updateField('avgTemperature', v)}
          min={21.06} max={29.19} step={0.5} unit="°C"
          color="red"
        />
        <NumberField
          label="Total Rainfall"
          icon={CloudRain}
          value={inputs.totalRainfall}
          onChange={v => updateField('totalRainfall', v)}
          min={270.44} max={10036.74} step={0.01} unit="mm"
          color="blue"
        />
        <SliderField
          label="Avg Humidity"
          icon={Wind}
          value={inputs.avgHumidity}
          onChange={v => updateField('avgHumidity', v)}
          min={70.05} max={93} step={1} unit="%"
          color="cyan"
        />
        <NumberField
          label="Sown Acre"
          icon={Maximize2}
          value={inputs.sownAcre}
          onChange={v => updateField('sownAcre', v)}
          min={0} max={5225220} step={1} unit=" acre"
        />
      </div>

      {/* Year Input */}
      <div className="max-w-xs">
        <SliderField
          label="Year"
          icon={Calendar}
          value={inputs.year}
          onChange={v => updateField('year', v)}
          min={2012} max={2023} step={1} unit=""
        />
      </div>

      {/* Submit */}
      <button
        onClick={onPredict}
        disabled={loading || !inputs.region || !inputs.soilType || !inputs.waterSource || !inputs.seedingSeason || chosenVariantUnavailable}
        className="btn-primary w-full sm:w-auto flex items-center justify-center gap-2"
      >
        {loading ? (
          <>
            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            Processing...
          </>
        ) : (
          <>🌾 {taskLabel}</>
        )}
      </button>
    </div>
  );
}

export { DEFAULT_INPUTS };
