"""
Pydantic schemas for agricultural input data and prediction responses.

All input fields use raw/human-readable values. Prediction responses follow the
existing frontend (camelCase) contract so the frontend can consume them directly.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ─── Valid categorical values ────────────────────────────────────────────────
# These match the ACTUAL project dataset vocabulary (the trained artifacts were
# fit on these values). The OneHotEncoders use handle_unknown='ignore', so the
# API is tolerant of unknown values, but the canonical categories are here.

class Region(str, Enum):
    AYEYARWADY = "Ayeyarwady"
    BAGO = "Bago"
    CHIN = "Chin"
    KACHIN = "Kachin"
    KAYAH = "Kayah"
    KAYIN = "Kayin"
    MAGWAY = "Magway"
    MANDALAY = "Mandalay"
    MON = "Mon"
    NAYPYITAW = "Nay Pyi Taw"
    RAKHINE = "Rakhine"
    SAGAING = "Sagaing"
    SHAN = "Shan"
    TANINTHARYI = "Tanintharyi"
    YANGON = "Yangon"


class SoilType(str, Enum):
    FERRALSOLS_RED = "Acrisols / Ferralsols / Red Earth"
    RED_BROWN_LATERITIC = "Acrisols / Red-Brown Lateritic"
    CAMBISOLS_FLUVISOLS = "Cambisols / Fluvisols / Loam"
    CAMBISOLS_MOUNTAIN = "Cambisols / Mountainous Soil"
    FERRALSOLS_GLY = "Ferralsols / Gleysols / Lateritic"
    FERRALSOLS_PLIN = "Ferralsols / Plinthosols / Lateritic"
    FLUVISOLS_GLY = "Fluvisols / Gleysols / Alluvial"
    GLEYSOLS_ALLUVIAL = "Gleysols / Alluvial / Mountainous"
    LUVISOLS_CAMBISOLS = "Luvisols / Cambisols / Savanna"
    LUVISOLS_RED = "Luvisols / Red Earth"
    REGOSOLS_LUVISOLS = "Regosols / Luvisols / Sandy Savanna"


class WaterSource(str, Enum):
    IRRIGATION = "Irrigation"
    RAINFED = "Rainfed"


class CropType(str, Enum):
    BETELEAVES = "Beteleaves"
    BETELNUT = "Betelnut"
    BOCATEPE = "Bocatepe(Cow Pea)"
    BUTTER_BEAN = "Butter Bean"
    CHILLIE = "Chillie"
    COFFEE = "Coffee"
    GARLIC = "Garlic"
    GRAM = "Gram(Chick pea)"
    GROUNDNUT_RAIN = "Groundnut(Rain)"
    GROUNDNUT_WINTER = "Groundnut(Winter)"
    MAIZE = "Maize"
    MATPE = "Matpe(Blackgram)"
    ONION = "Onion"
    PADDY = "Paddy"
    PANAUK = "Panauk(Krishna mung)"
    PEBOKE = "Peboke(Soy bean)"
    PEBYUGALE = "Pebyugale(Duffin bean)"
    PEDISEIN = "Pedisein(Greengram)"
    PEGYA = "Pegya(Lima bean)"
    PEGYI = "Pegyi(Lablab bean)"
    PELUN = "Pelun(Cow pea)"
    PESINGON = "Pesingon(Pigeon pea)"
    PEYIN = "Peyin(Rice bean)"
    PLANTAIN = "Plantain"
    POTATO = "Potato"
    SADAWPE = "Sadawpe(Garden pea)"
    SESAMUM_EARLY = "Sesamum(Early)"
    SESAMUM_LATE = "Sesamum(Late)"
    SESAMUM_SUMMER = "Sesamum(Summer)"
    SUGARCANE = "Sugarcane"
    SUNFLOWER = "Sunflower"
    TEA = "Tea"
    WHEAT = "Wheat"


class SeedingSeason(str, Enum):
    RAINY = "Rainy"
    SUMMER = "Summer"
    WINTER = "Winter"


class ModelVariant(str, Enum):
    BASELINE = "baseline"
    FEATURE_ENGINEERING = "feature_engineering"
    ADVANCED = "advanced"


class ComparisonTask(str, Enum):
    CROP_TYPE = "crop_type"
    CROP_YIELD = "crop_yield"
    YIELD_LEVEL = "yield_level"


# ─── Input schemas ───────────────────────────────────────────────────────────

class AgriInputBase(BaseModel):
    """Shared agricultural input fields (human-readable)."""
    region: Region
    year: int = Field(..., ge=2010, le=2030, description="Crop year")
    crop_type: CropType
    sown_acre: float = Field(..., ge=0, description="Sown acreage")
    soil_type: SoilType
    avg_temperature: float = Field(..., ge=-10, le=60, description="Average temperature (°C)")
    total_rainfall: float = Field(..., ge=0, le=20000, description="Total rainfall (mm)")
    avg_humidity: float = Field(..., ge=0, le=100, description="Average humidity (%)")
    water_source: WaterSource
    seeding_season: SeedingSeason


class CropTypePredictionRequest(AgriInputBase):
    model_variant: ModelVariant = ModelVariant.BASELINE


class YieldLevelPredictionRequest(AgriInputBase):
    model_variant: ModelVariant = ModelVariant.BASELINE


class CropYieldPredictionRequest(AgriInputBase):
    model_variant: ModelVariant = ModelVariant.BASELINE


# ─── Response schemas (frontend camelCase contract) ─────────────────────────

class FeatureImportance(BaseModel):
    feature: str
    importance: float


class CropTypePredictionResponse(BaseModel):
    predictedCrop: str
    confidence: float
    topPredictions: list[dict]
    featureImportance: list[FeatureImportance]
    modelMetrics: dict


class YieldLevelPredictionResponse(BaseModel):
    predictedLevel: str
    probabilityLow: float
    probabilityHigh: float
    featureImportance: list[FeatureImportance]
    modelMetrics: dict


class CropYieldPredictionResponse(BaseModel):
    predictedYield: float
    featureImportance: list[FeatureImportance]
    modelMetrics: dict


class ModelInfo(BaseModel):
    variant: str
    name: str
    prediction: Optional[dict] = None
    metrics: Optional[dict] = None
    error: Optional[str] = None
    available: bool = True


class ComparisonResponse(BaseModel):
    models: list[ModelInfo]
    best_model: str
    comparison_metrics: list[dict]


class HistoricalDataPoint(BaseModel):
    year: int
    yield_value: float
    rainfall: float
    temperature: float
    area: float


class HistoricalTrendsResponse(BaseModel):
    data: list[HistoricalDataPoint]


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: dict[str, dict[str, bool]]
