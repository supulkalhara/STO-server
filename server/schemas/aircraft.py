from __future__ import annotations
"""Pydantic schemas for aircraft CRUD."""

from typing import Optional
from pydantic import BaseModel


class AircraftCreate(BaseModel):
    icao_type_designator: str
    registration: str
    callsign: str
    operator: Optional[str] = None
    wake_turbulence_category: str  # L | M | H | J
    engine_type: str               # Jet | Turboprop | Piston | Electric
    mtow_kg: Optional[float] = None
    v1_kts: Optional[float] = None
    vr_kts: Optional[float] = None
    v2_kts: Optional[float] = None
    equipment_suffixes: Optional[str] = None
    rnav_approved: str = "N"
    rvsm_approved: str = "N"


class AircraftUpdate(BaseModel):
    icao_type_designator: Optional[str] = None
    callsign: Optional[str] = None
    operator: Optional[str] = None
    wake_turbulence_category: Optional[str] = None
    engine_type: Optional[str] = None
    mtow_kg: Optional[float] = None
    v1_kts: Optional[float] = None
    vr_kts: Optional[float] = None
    v2_kts: Optional[float] = None
    equipment_suffixes: Optional[str] = None
    rnav_approved: Optional[str] = None
    rvsm_approved: Optional[str] = None
    is_active: Optional[int] = None


class AircraftOut(AircraftCreate):
    id: int
    is_active: int

    model_config = {"from_attributes": True}
