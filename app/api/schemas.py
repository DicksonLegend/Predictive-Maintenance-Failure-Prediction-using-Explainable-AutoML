"""Pydantic schemas for the Predictive Maintenance API."""
import math
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator


class SensorInput(BaseModel):
    """Sensor telemetry reading for a machine instance."""
    type: Literal["L", "M", "H"] = Field(
        ...,
        description="Machine quality variant: L (50%), M (30%), or H (20%)",
        examples=["L"]
    )
    air_temp: float = Field(
        ...,
        description="Air temperature in Kelvin [K] (typical range: 290.0 - 310.0)",
        examples=[298.1]
    )
    process_temp: float = Field(
        ...,
        description="Process temperature in Kelvin [K] (typical range: 300.0 - 320.0)",
        examples=[308.6]
    )
    rotational_speed: float = Field(
        ...,
        description="Rotational speed in RPM [rpm] (typical range: 1000 - 3000)",
        examples=[1551.0]
    )
    torque: float = Field(
        ...,
        description="Torque in Newton-meters [Nm] (typical range: 0.0 - 100.0)",
        examples=[42.8]
    )
    tool_wear: float = Field(
        ...,
        description="Tool wear in minutes [min] (typical range: 0 - 300)",
        examples=[0.0]
    )

    @field_validator("air_temp", "process_temp", "rotational_speed", "torque", "tool_wear")
    @classmethod
    def check_finite(cls, v: float, info) -> float:
        if not math.isfinite(v):
            raise ValueError(f"Field '{info.field_name}' must be a finite numeric value.")
        return float(v)

    @field_validator("air_temp", "process_temp")
    @classmethod
    def check_temp_plausible(cls, v: float, info) -> float:
        if v < 200.0 or v > 400.0:
            raise ValueError(f"Field '{info.field_name}' temperature {v}K is outside physically plausible limits (200K - 400K).")
        return v

    @field_validator("rotational_speed")
    @classmethod
    def check_speed_plausible(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Rotational speed cannot be negative: {v}")
        return v

    @field_validator("torque")
    @classmethod
    def check_torque_plausible(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Torque cannot be negative: {v}")
        return v

    @field_validator("tool_wear")
    @classmethod
    def check_tool_wear_plausible(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Tool wear cannot be negative: {v}")
        return v


class PredictionDetails(BaseModel):
    failure_probability: float
    predicted_class: int
    risk_level: Literal["Low", "Medium", "High"]
    threshold: float = 0.5


class ModelInfo(BaseModel):
    name: str
    source: str


class PredictionResponse(BaseModel):
    input: Dict[str, Any]
    prediction: PredictionDetails
    model: ModelInfo


class FeatureImpact(BaseModel):
    feature: str
    impact: str
    attribution: Optional[float] = None


class ExplanationResponse(BaseModel):
    input: Dict[str, Any]
    prediction: PredictionDetails
    top_reasons: List[FeatureImpact]
    recommendation: str
    local_shap_plot: Optional[str] = None
    lime_explanation_html: Optional[str] = None
    explanation_method: str


class WhatIfRequest(BaseModel):
    original_input: SensorInput
    modified_input: SensorInput


class WhatIfResponse(BaseModel):
    original_probability: float
    modified_probability: float
    probability_change: float
    original_risk_level: str
    modified_risk_level: str
    interpretation: str
    recommendation: str


class TextExplainRequest(BaseModel):
    text: str = Field(..., min_length=2, description="Technician maintenance notes")


class WordImpact(BaseModel):
    word: str
    impact: str
    weight: Optional[float] = None


class TextExplainResponse(BaseModel):
    text: str
    predicted_label: Literal["urgent", "normal"]
    probability: float
    important_words: List[WordImpact]
    lime_html: Optional[str] = None


class ImageDemoItem(BaseModel):
    title: str
    original: str
    heatmap: str
    overlay: str
    predicted_class: str
    explanation: str


class ImageDemoResponse(BaseModel):
    image_demo_mode: str
    items: List[ImageDemoItem]
