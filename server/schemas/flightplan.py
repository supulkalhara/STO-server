"""Flight plan schemas — ICAO FPL item 15/18 fields."""
from __future__ import annotations

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class FlightPlanCreate(BaseModel):
    callsign: str = Field(..., max_length=10)
    aircraft_type: str = Field(..., description="ICAO type designator, e.g. B738")
    wake_turbulence_category: str = Field(..., pattern="^[LMHJ]$")
    departure_icao: str = Field(..., min_length=4, max_length=4)
    destination_icao: str = Field(..., min_length=4, max_length=4)
    alternate_icao: Optional[str] = Field(None, min_length=4, max_length=4)
    eobt: datetime = Field(..., description="Estimated Off-Block Time (UTC)")
    tobt: Optional[datetime] = Field(None, description="Target Off-Block Time (UTC)")
    ctot: Optional[datetime] = Field(None, description="Calculated Take-Off Time (UTC)")
    cruising_level: str = Field(..., description="e.g. F350, A100, VFR")
    route: Optional[str] = None
    remarks: Optional[str] = None


class FlightPlanUpdate(BaseModel):
    tobt: Optional[datetime] = None
    ctot: Optional[datetime] = None
    cruising_level: Optional[str] = None
    route: Optional[str] = None
    remarks: Optional[str] = None
    status: Optional[str] = None   # FILED | ACTIVATED | CLOSED | CANCELLED


class FlightPlanOut(FlightPlanCreate):
    id: int
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}
