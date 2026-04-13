"""Go/No-Go prediction schemas."""
from __future__ import annotations

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class GoNoGoRequest(BaseModel):
    icao: str = Field(..., description="Aerodrome ICAO code")
    wind_speed_kt: float = Field(0.0, ge=0)
    wind_gust_kt: Optional[float] = Field(None, ge=0)
    visibility_sm: float = Field(10.0, ge=0)
    ceiling_ft: Optional[float] = Field(None, ge=0, description="Lowest broken/overcast layer in ft")
    temp_c: float = Field(15.0)
    crosswind_kt: float = Field(0.0, ge=0, description="Crosswind component")
    active_notams: int = Field(0, ge=0, description="Number of active runway/nav NOTAMs")
    traffic_count: int = Field(0, ge=0, description="Aircraft movements in last 30 min")


class ShapFactor(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str     # "increases_risk" | "reduces_risk"


class GoNoGoResponse(BaseModel):
    icao: str
    decision: str          # "GO" | "NO-GO" | "CAUTION"
    confidence: float      # 0.0 – 1.0
    risk_score: float      # 0.0 – 1.0  (higher = riskier)
    top_factors: List[ShapFactor]
    explanation: str
