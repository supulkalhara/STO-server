from __future__ import annotations
"""Weather router — METAR proxy via NOAA Aviation Weather Center (no API key needed)."""

import re
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, status

from schemas.weather import MetarResponse

router = APIRouter(prefix="/weather", tags=["Weather"])

NOAA_METAR_URL = "https://aviationweather.gov/api/data/metar"

# Regex patterns for parsing METAR raw text
_WIND_RE = re.compile(r"(\d{3}|VRB)(\d{2,3})(G(\d{2,3}))?KT")
# Visibility: US format "10SM" or "1/4SM"; International format "9999" or "0800" (metres)
_VIS_SM_RE = re.compile(r"\b(\d+(?:/\d+)?)\s*SM\b")
_VIS_M_RE = re.compile(r"\b(\d{4})\b")   # 4-digit block = metres (e.g. 9999, 0800)
_TEMP_RE = re.compile(r"\b(M?\d{2})/(M?\d{2})\b")
_ALT_RE = re.compile(r"\bA(\d{4})\b")
_QNH_RE = re.compile(r"\bQ(\d{4})\b")    # International QNH in hPa
_SKY_RE = re.compile(r"\b(FEW|SCT|BKN|OVC|CLR|SKC|CAVOK)(\d{3})?\b")
_FLIGHT_CAT_RE = re.compile(r"\b(VFR|MVFR|IFR|LIFR)\b")


def _parse_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def _c_from_metar(raw: str) -> Optional[float]:
    """Extract temperature/dewpoint pair, handle M (minus) prefix."""
    match = _TEMP_RE.search(raw)
    if not match:
        return None, None
    def conv(s):
        if s.startswith("M"):
            return -float(s[1:])
        return float(s)
    return conv(match.group(1)), conv(match.group(2))


def _parse_raw_metar(icao: str, raw: str) -> MetarResponse:
    """Extract structured fields from a raw METAR string."""
    wind = _WIND_RE.search(raw)
    wind_dir = _parse_int(wind.group(1)) if wind and wind.group(1) != "VRB" else None
    wind_spd = _parse_int(wind.group(2)) if wind else None
    wind_gust = _parse_int(wind.group(4)) if wind and wind.group(4) else None

    # Parse visibility — prefer SM format; fall back to metres → convert to SM
    vis_sm = _VIS_SM_RE.search(raw)
    if vis_sm:
        parts = vis_sm.group(1).split("/")
        vis = float(parts[0]) / float(parts[1]) if len(parts) == 2 else float(parts[0])
    else:
        vis_m = _VIS_M_RE.search(raw)
        vis = round(int(vis_m.group(1)) / 1609.34, 1) if vis_m else None  # metres → statute miles

    temp_c, dew_c = _c_from_metar(raw)

    alt_match = _ALT_RE.search(raw)
    qnh_match = _QNH_RE.search(raw)
    if alt_match:
        alt_inhg = round(int(alt_match.group(1)) / 100, 2)
    elif qnh_match:
        alt_inhg = round(int(qnh_match.group(1)) * 0.02953, 2)  # hPa → inHg
    else:
        alt_inhg = None

    sky_conditions = " ".join(
        m.group(0) for m in _SKY_RE.finditer(raw)
    ) or None

    fc_match = _FLIGHT_CAT_RE.search(raw)
    flight_cat = fc_match.group(1) if fc_match else None

    # Observation time: first token after ICAO (format DDHHMMz)
    tokens = raw.split()
    obs_time = tokens[1] if len(tokens) > 1 else None

    return MetarResponse(
        icao=icao.upper(),
        raw_text=raw,
        observation_time=obs_time,
        wind_dir_degrees=wind_dir,
        wind_speed_kt=wind_spd,
        wind_gust_kt=wind_gust,
        visibility_statute_mi=vis,
        sky_condition=sky_conditions,
        temp_c=temp_c,
        dewpoint_c=dew_c,
        altim_in_hg=alt_inhg,
        flight_category=flight_cat,
    )


@router.get("/metar/{icao}", response_model=MetarResponse)
async def get_metar(icao: str):
    """
    Fetch the latest METAR for an ICAO aerodrome code.
    Example: /weather/metar/VCBI  (Bandaranaike / Colombo)
    """
    icao = icao.upper().strip()
    if not re.match(r"^[A-Z]{4}$", icao):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="ICAO code must be exactly 4 uppercase letters (e.g. VCBI, EGLL)",
        )

    params = {"ids": icao, "format": "raw", "taf": "false", "hours": "1"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(NOAA_METAR_URL, params=params)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to fetch METAR from NOAA: {exc}",
            )

    raw_text = resp.text.strip()
    if not raw_text:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No METAR found for {icao}. Verify the ICAO code is correct.",
        )

    return _parse_raw_metar(icao, raw_text)
