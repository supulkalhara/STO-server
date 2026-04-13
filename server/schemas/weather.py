from __future__ import annotations
"""Pydantic schemas for weather / METAR data."""

from typing import Optional
from pydantic import BaseModel


class MetarResponse(BaseModel):
    icao: str
    raw_text: str
    observation_time: Optional[str] = None
    wind_dir_degrees: Optional[int] = None
    wind_speed_kt: Optional[int] = None
    wind_gust_kt: Optional[int] = None
    visibility_statute_mi: Optional[float] = None
    sky_condition: Optional[str] = None  # e.g. "BKN015 OVC060"
    temp_c: Optional[float] = None
    dewpoint_c: Optional[float] = None
    altim_in_hg: Optional[float] = None
    flight_category: Optional[str] = None  # VFR | MVFR | IFR | LIFR
    source: str = "NOAA Aviation Weather Center"
