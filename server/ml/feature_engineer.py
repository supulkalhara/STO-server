"""
Feature engineering for Safe TakeOff Go/No-Go decision model.

Transforms raw METAR + flight plan + traffic data into 20+ features
for XGBoost training and inference.

Base 8 features (direct from observations):
  1. wind_speed_kt — sustained wind speed
  2. wind_gust_kt — gust speed
  3. visibility_sm — visibility in statute miles
  4. ceiling_ft — cloud ceiling above ground level
  5. temp_c — ambient temperature
  6. crosswind_kt — perpendicular component to runway heading
  7. active_notams — count of active NOTAMs
  8. traffic_count — concurrent departures in queue

Derived 12+ features (engineered knowledge):
  9. runway_length_m — specific to departure aerodrome
  10. surface_condition — dry/wet/contaminated (from NOTAMs)
  11. metar_category — VFR/MVFR/IFR/LIFR
  12. tailwind_kt — longitudinal tail component
  13. headwind_kt — longitudinal head component
  14. relative_humidity — % (computed from temp + dewpoint)
  15. aircraft_mtow_ratio — payload / max takeoff weight
  16. wake_count_heavy — heavy/jumbo departures in last 30 min
  17. metar_age_min — freshness of METAR data
  18. notam_severity_index — weighted severity score
  19. ceiling_visibility_ratio — ratio (complexity metric)
  20. temperature_trend — deviation from seasonal average
"""
from typing import Dict, Optional, List
from math import sin, cos, radians, sqrt
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Runway Database — ICAO code → runway heading, length
# (In production: load from external DB or aviation data service)
# ─────────────────────────────────────────────────────────────────────────────

RUNWAY_DB = {
    "EGLL": {"heading_deg": 270, "length_m": 3901, "surface": "asphalt"},  # LHR 27L
    "EGKK": {"heading_deg": 260, "length_m": 3158, "surface": "asphalt"},  # Gatwick 26L
    "LFPG": {"heading_deg": 270, "length_m": 4000, "surface": "asphalt"},  # CDG 27L
    "LEMD": {"heading_deg": 320, "length_m": 3500, "surface": "asphalt"},  # MAD 32L
    "UUWW": {"heading_deg": 90, "length_m": 3500, "surface": "asphalt"},   # SVO 09L
    "KJFK": {"heading_deg": 40, "length_m": 4423, "surface": "asphalt"},   # JFK 04R
    "KLAX": {"heading_deg": 250, "length_m": 3885, "surface": "asphalt"},  # LAX 24L/R
}

# Seasonal average temperatures per ICAO (simplified — in prod: use historical DB)
SEASONAL_TEMPS = {
    "EGLL": {"avg_c": 10.0},
    "EGKK": {"avg_c": 10.0},
    "LFPG": {"avg_c": 11.0},
    "LEMD": {"avg_c": 15.0},
    "UUWW": {"avg_c": 5.0},
    "KJFK": {"avg_c": 12.0},
    "KLAX": {"avg_c": 18.0},
}


def compute_crosswind(wind_dir_deg: Optional[float], wind_speed_kt: Optional[float], runway_heading_deg: float) -> float:
    """
    Compute crosswind component perpendicular to runway.

    Formula: crosswind = wind_speed * sin(wind_dir - runway_heading)
    """
    if wind_dir_deg is None or wind_speed_kt is None:
        return 0.0

    try:
        delta = (wind_dir_deg - runway_heading_deg) * 3.14159 / 180.0
        crosswind = abs(wind_speed_kt * sin(delta))
        return round(crosswind, 2)
    except Exception as e:
        logger.warning(f"compute_crosswind error: {e}")
        return 0.0


def compute_headwind(wind_dir_deg: Optional[float], wind_speed_kt: Optional[float], runway_heading_deg: float) -> float:
    """
    Compute headwind component along runway (positive = head, negative = tail).

    Formula: headwind = wind_speed * cos(wind_dir - runway_heading)
    """
    if wind_dir_deg is None or wind_speed_kt is None:
        return 0.0

    try:
        delta = (wind_dir_deg - runway_heading_deg) * 3.14159 / 180.0
        headwind = wind_speed_kt * cos(delta)
        return round(headwind, 2)
    except Exception as e:
        logger.warning(f"compute_headwind error: {e}")
        return 0.0


def compute_relative_humidity(temp_c: Optional[float], dewpoint_c: Optional[float]) -> float:
    """
    Compute relative humidity from temperature and dewpoint.

    Simplified Magnus formula: RH = 100 * (e^((b*Td)/(c+Td)) / e^((b*T)/(c+T)))
    where b=17.27, c=237.7
    """
    if temp_c is None or dewpoint_c is None:
        return 50.0  # neutral default

    try:
        b = 17.27
        c = 237.7
        numerator = (b * dewpoint_c) / (c + dewpoint_c)
        denominator = (b * temp_c) / (c + temp_c)
        rh = 100.0 * (2.71828 ** numerator) / (2.71828 ** denominator)
        return round(max(0.0, min(100.0, rh)), 1)
    except Exception as e:
        logger.warning(f"compute_relative_humidity error: {e}")
        return 50.0


def classify_metar_category(ceiling_ft: Optional[float], visibility_sm: Optional[float]) -> str:
    """
    Classify METAR into VFR/MVFR/IFR/LIFR per FAA rules.

    VFR:  ceiling >= 3000 ft AND visibility >= 5 sm
    MVFR: 1000 <= ceiling < 3000 ft OR 3 <= visibility < 5 sm
    IFR:  500 <= ceiling < 1000 ft OR 1 <= visibility < 3 sm
    LIFR: ceiling < 500 ft OR visibility < 1 sm
    """
    if ceiling_ft is None or visibility_sm is None:
        return "MVFR"  # conservative default

    if ceiling_ft >= 3000 and visibility_sm >= 5:
        return "VFR"
    elif ceiling_ft >= 1000 and visibility_sm >= 3:
        return "MVFR"
    elif ceiling_ft >= 500 and visibility_sm >= 1:
        return "IFR"
    else:
        return "LIFR"


def classify_runway_condition(surface_condition_str: Optional[str], active_notams: int) -> str:
    """
    Classify runway surface from NOTAM text or default.

    DRY, WET, CONTAMINATED (snow, ice, water, debris)
    """
    if not surface_condition_str or active_notams == 0:
        return "DRY"

    lower = surface_condition_str.lower()
    if any(x in lower for x in ["snow", "ice", "slush", "contaminated"]):
        return "CONTAMINATED"
    elif any(x in lower for x in ["wet", "rain", "water"]):
        return "WET"
    else:
        return "DRY"


def engineer_features(
    icao: str,
    wind_dir_deg: Optional[float],
    wind_speed_kt: Optional[float],
    wind_gust_kt: Optional[float],
    visibility_sm: Optional[float],
    ceiling_ft: Optional[float],
    temp_c: Optional[float],
    dewpoint_c: Optional[float],
    active_notams: int,
    traffic_count: int,
    aircraft_mtow_kg: Optional[float],
    payload_kg: Optional[float],
    wake_count_heavy: int = 0,
    metar_age_min: float = 0.0,
    surface_condition_str: Optional[str] = None,
    historical_temp_c: Optional[float] = None,
) -> Dict[str, float]:
    """
    Engineer all 20+ features for XGBoost inference/training.

    Args:
        icao: departure aerodrome (e.g., 'EGLL')
        wind_* : raw METAR wind observations
        visibility_sm, ceiling_ft, temp_c, dewpoint_c : METAR
        active_notams, traffic_count: from API
        aircraft_mtow_kg, payload_kg: from aircraft registry
        wake_count_heavy: departures of H/J class in last 30 min
        metar_age_min: age of METAR data
        surface_condition_str: NOTAM surface description
        historical_temp_c: seasonal average for the aerodrome

    Returns:
        Dict with 20+ feature keys → float values, ready for XGBoost.predict()
    """

    # Get runway info for this ICAO
    runway_info = RUNWAY_DB.get(icao, {"heading_deg": 270, "length_m": 3500})
    runway_heading = runway_info["heading_deg"]
    runway_length_m = runway_info["length_m"]

    # Defaults for missing data
    wind_speed_kt = wind_speed_kt or 0.0
    wind_gust_kt = wind_gust_kt or wind_speed_kt
    visibility_sm = visibility_sm or 10.0
    ceiling_ft = ceiling_ft or 5000.0
    temp_c = temp_c or 15.0
    active_notams = active_notams or 0
    traffic_count = traffic_count or 1

    # ── BASE 8 FEATURES ───────────────────────────────────────────────────────
    features = {
        "wind_speed_kt": round(wind_speed_kt, 2),
        "wind_gust_kt": round(wind_gust_kt, 2),
        "visibility_sm": round(visibility_sm, 2),
        "ceiling_ft": round(ceiling_ft, 1),
        "temp_c": round(temp_c, 1),
        "crosswind_kt": compute_crosswind(wind_dir_deg, wind_speed_kt, runway_heading),
        "active_notams": float(active_notams),
        "traffic_count": float(traffic_count),
    }

    # ── DERIVED 12+ FEATURES ──────────────────────────────────────────────────

    # 9. Runway length
    features["runway_length_m"] = round(runway_length_m, 1)

    # 10. Surface condition (from NOTAMs)
    features["surface_condition_code"] = (
        0.0 if classify_runway_condition(surface_condition_str, active_notams) == "DRY"
        else 1.0 if classify_runway_condition(surface_condition_str, active_notams) == "WET"
        else 2.0  # CONTAMINATED
    )

    # 11. METAR category
    metar_cat = classify_metar_category(ceiling_ft, visibility_sm)
    features["metar_category_code"] = (
        0.0 if metar_cat == "VFR"
        else 1.0 if metar_cat == "MVFR"
        else 2.0 if metar_cat == "IFR"
        else 3.0  # LIFR
    )

    # 12-13. Tailwind and headwind
    features["tailwind_kt"] = -compute_headwind(wind_dir_deg, wind_speed_kt, runway_heading)  # negative = tail
    features["headwind_kt"] = compute_headwind(wind_dir_deg, wind_speed_kt, runway_heading)

    # 14. Relative humidity
    features["relative_humidity"] = compute_relative_humidity(temp_c, dewpoint_c)

    # 15. Aircraft MTOW ratio (payload / max takeoff weight)
    if aircraft_mtow_kg and payload_kg:
        features["aircraft_mtow_ratio"] = round(payload_kg / aircraft_mtow_kg, 3)
    else:
        features["aircraft_mtow_ratio"] = 0.7  # typical default

    # 16. Wake count (heavy/jumbo)
    features["wake_count_heavy"] = float(wake_count_heavy)

    # 17. METAR age (freshness)
    features["metar_age_min"] = round(metar_age_min, 1)

    # 18. NOTAM severity index (weighted by count + criticality)
    notam_severity = active_notams * 1.0  # simplified — in prod: weight by priority
    features["notam_severity_index"] = round(notam_severity, 2)

    # 19. Ceiling to visibility ratio (IFR complexity)
    if visibility_sm > 0:
        features["ceiling_visibility_ratio"] = round(ceiling_ft / (visibility_sm * 500), 2)  # normalized
    else:
        features["ceiling_visibility_ratio"] = 10.0

    # 20. Temperature trend (vs seasonal average)
    seasonal_avg = SEASONAL_TEMPS.get(icao, {}).get("avg_c", temp_c)
    if historical_temp_c:
        seasonal_avg = historical_temp_c
    features["temperature_trend"] = round(temp_c - seasonal_avg, 1)

    return features


def get_feature_names() -> List[str]:
    """Return ordered list of feature names for model training."""
    return [
        "wind_speed_kt",
        "wind_gust_kt",
        "visibility_sm",
        "ceiling_ft",
        "temp_c",
        "crosswind_kt",
        "active_notams",
        "traffic_count",
        "runway_length_m",
        "surface_condition_code",
        "metar_category_code",
        "tailwind_kt",
        "headwind_kt",
        "relative_humidity",
        "aircraft_mtow_ratio",
        "wake_count_heavy",
        "metar_age_min",
        "notam_severity_index",
        "ceiling_visibility_ratio",
        "temperature_trend",
    ]
