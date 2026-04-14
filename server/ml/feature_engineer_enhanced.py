"""
Enhanced Feature Engineering - Phase 3A Optimization
Adds decision-critical features to improve accuracy to 80%+

New features added:
- Runway-specific crosswind limits (aircraft type dependent)
- Decision thresholds (GO/CAUTION/NO-GO boundaries)
- Wind shear indicators
- Ceiling/visibility interaction terms
- Time of day (peak departure hours)
- Surface friction coefficient
- Tailwind limits (aircraft-dependent)
- Combined risk score
"""
from typing import Dict, Optional, List
from math import sin, cos, radians, sqrt
import logging

logger = logging.getLogger(__name__)

# Runway Database with enhanced runway data
RUNWAY_DB = {
    "EGLL": {"heading_deg": 270, "length_m": 3901, "surface": "asphalt", "friction": 0.35},
    "EGKK": {"heading_deg": 260, "length_m": 3158, "surface": "asphalt", "friction": 0.35},
    "LFPG": {"heading_deg": 270, "length_m": 4000, "surface": "asphalt", "friction": 0.35},
    "LEMD": {"heading_deg": 320, "length_m": 3500, "surface": "asphalt", "friction": 0.35},
    "UUWW": {"heading_deg": 90, "length_m": 3500, "surface": "asphalt", "friction": 0.30},
    "KJFK": {"heading_deg": 40, "length_m": 4423, "surface": "asphalt", "friction": 0.35},
    "KLAX": {"heading_deg": 250, "length_m": 3885, "surface": "asphalt", "friction": 0.35},
}

# Aircraft-specific crosswind and tailwind limits (kts)
AIRCRAFT_LIMITS = {
    "B738": {"max_crosswind": 25, "max_tailwind": 10, "min_runway_m": 2200, "mtow_kg": 79016},
    "B787": {"max_crosswind": 30, "max_tailwind": 15, "min_runway_m": 2500, "mtow_kg": 242500},
    "A320": {"max_crosswind": 25, "max_tailwind": 10, "min_runway_m": 2150, "mtow_kg": 79000},
    "A380": {"max_crosswind": 25, "max_tailwind": 10, "min_runway_m": 3000, "mtow_kg": 575000},
    "CRJ7": {"max_crosswind": 20, "max_tailwind": 8, "min_runway_m": 1500, "mtow_kg": 34019},
    "E190": {"max_crosswind": 20, "max_tailwind": 8, "min_runway_m": 1800, "mtow_kg": 61000},
}

# Seasonal average temperatures per ICAO
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
    """Compute crosswind component perpendicular to runway."""
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
    """Compute headwind component along runway (positive = head, negative = tail)."""
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
    """Compute relative humidity using Magnus formula."""
    if temp_c is None or dewpoint_c is None:
        return 50.0
    try:
        a = 17.27
        b = 237.7
        alpha = ((a * temp_c) / (b + temp_c)) + (((a * dewpoint_c) / (b + dewpoint_c)))
        rh = 100 * pow(10, (alpha - 2 * ((a * temp_c) / (b + temp_c))))
        return min(100.0, max(0.0, round(rh, 1)))
    except Exception as e:
        logger.warning(f"compute_relative_humidity error: {e}")
        return 50.0


def classify_metar_category(ceiling_ft: Optional[float], visibility_sm: Optional[float]) -> str:
    """Classify METAR weather category."""
    if ceiling_ft is None or visibility_sm is None:
        return "VFR"
    if ceiling_ft >= 3000 and visibility_sm >= 5:
        return "VFR"
    elif ceiling_ft >= 1000 and visibility_sm >= 3:
        return "MVFR"
    elif ceiling_ft >= 500 and visibility_sm >= 1:
        return "IFR"
    else:
        return "LIFR"


def classify_runway_condition(surface_condition_str: Optional[str], active_notams: int) -> str:
    """Classify runway surface condition."""
    if not surface_condition_str or active_notams == 0:
        return "DRY"
    lower = surface_condition_str.lower()
    if any(x in lower for x in ["snow", "ice", "slush", "contaminated"]):
        return "CONTAMINATED"
    elif any(x in lower for x in ["wet", "rain", "water"]):
        return "WET"
    else:
        return "DRY"


def compute_decision_risk_score(
    crosswind_kt: float,
    headwind_kt: float,
    ceiling_ft: float,
    visibility_sm: float,
    aircraft_type: str,
    runway_length_m: float,
    surface_condition: str,
) -> float:
    """
    Compute composite risk score combining all decision factors.
    Returns 0.0 (safe) to 1.0 (critical).
    
    This is the key enhancement: a single score that captures
    the complexity of the decision.
    """
    risk = 0.0
    
    # Get aircraft limits
    limits = AIRCRAFT_LIMITS.get(aircraft_type, AIRCRAFT_LIMITS["B738"])
    
    # 1. Crosswind risk (0-0.3)
    if crosswind_kt > limits["max_crosswind"]:
        risk += 0.3  # CRITICAL
    elif crosswind_kt > limits["max_crosswind"] * 0.8:
        risk += 0.2  # HIGH
    elif crosswind_kt > limits["max_crosswind"] * 0.6:
        risk += 0.1  # MODERATE
    
    # 2. Tailwind risk (0-0.2)
    tailwind = -headwind_kt if headwind_kt < 0 else 0
    if tailwind > limits["max_tailwind"]:
        risk += 0.2  # CRITICAL
    elif tailwind > limits["max_tailwind"] * 0.8:
        risk += 0.1  # HIGH
    
    # 3. Runway length risk (0-0.2)
    if runway_length_m < limits["min_runway_m"]:
        risk += 0.2  # CRITICAL
    elif runway_length_m < limits["min_runway_m"] * 1.1:
        risk += 0.1  # HIGH
    
    # 4. Visibility risk (0-0.15)
    if visibility_sm < 1:
        risk += 0.15  # CRITICAL
    elif visibility_sm < 2:
        risk += 0.1  # HIGH
    elif visibility_sm < 3:
        risk += 0.05  # MODERATE
    
    # 5. Ceiling risk (0-0.15)
    if ceiling_ft < 400:
        risk += 0.15  # CRITICAL
    elif ceiling_ft < 700:
        risk += 0.1  # HIGH
    elif ceiling_ft < 1500:
        risk += 0.05  # MODERATE
    
    # 6. Surface condition risk (0-0.1)
    if surface_condition == "CONTAMINATED":
        risk += 0.1  # CRITICAL
    elif surface_condition == "WET":
        risk += 0.05  # MODERATE
    
    return min(1.0, round(risk, 3))


def engineer_features_enhanced(
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
    aircraft_type: str,
    aircraft_mtow_kg: Optional[float],
    payload_kg: Optional[float],
    wake_count_heavy: int = 0,
    metar_age_min: float = 0.0,
    surface_condition_str: Optional[str] = None,
    historical_temp_c: Optional[float] = None,
    hour_of_day: int = 12,
) -> Dict[str, float]:
    """
    Enhanced feature engineering with decision-critical features.
    Target: 80%+ accuracy through better feature design.
    """

    runway_info = RUNWAY_DB.get(icao, {"heading_deg": 270, "length_m": 3500, "friction": 0.35})
    runway_heading = runway_info["heading_deg"]
    runway_length_m = runway_info["length_m"]
    surface_friction = runway_info["friction"]

    # Defaults for missing data
    wind_speed_kt = wind_speed_kt or 0.0
    wind_gust_kt = wind_gust_kt or wind_speed_kt
    visibility_sm = visibility_sm or 10.0
    ceiling_ft = ceiling_ft or 5000.0
    temp_c = temp_c or 15.0
    active_notams = active_notams or 0
    traffic_count = traffic_count or 1

    # Get aircraft limits
    limits = AIRCRAFT_LIMITS.get(aircraft_type, AIRCRAFT_LIMITS["B738"])

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

    # ── DERIVED FEATURES ──────────────────────────────────────────────────────
    features["runway_length_m"] = round(runway_length_m, 1)

    surface_condition = classify_runway_condition(surface_condition_str, active_notams)
    features["surface_condition_code"] = (
        0.0 if surface_condition == "DRY"
        else 1.0 if surface_condition == "WET"
        else 2.0
    )

    metar_cat = classify_metar_category(ceiling_ft, visibility_sm)
    features["metar_category_code"] = (
        0.0 if metar_cat == "VFR"
        else 1.0 if metar_cat == "MVFR"
        else 2.0 if metar_cat == "IFR"
        else 3.0
    )

    features["tailwind_kt"] = -compute_headwind(wind_dir_deg, wind_speed_kt, runway_heading)
    features["headwind_kt"] = compute_headwind(wind_dir_deg, wind_speed_kt, runway_heading)

    features["relative_humidity"] = compute_relative_humidity(temp_c, dewpoint_c)

    if aircraft_mtow_kg and payload_kg:
        features["aircraft_mtow_ratio"] = round(payload_kg / aircraft_mtow_kg, 3)
    else:
        features["aircraft_mtow_ratio"] = 0.7

    features["wake_count_heavy"] = float(wake_count_heavy)
    features["metar_age_min"] = round(metar_age_min, 1)

    notam_severity = active_notams * 1.0
    features["notam_severity_index"] = round(notam_severity, 2)

    if visibility_sm > 0:
        features["ceiling_visibility_ratio"] = round(ceiling_ft / (visibility_sm * 500), 2)
    else:
        features["ceiling_visibility_ratio"] = 10.0

    seasonal_avg = SEASONAL_TEMPS.get(icao, {}).get("avg_c", temp_c)
    if historical_temp_c:
        seasonal_avg = historical_temp_c
    features["temperature_trend"] = round(temp_c - seasonal_avg, 1)

    # ── NEW DECISION-CRITICAL FEATURES (Phase 3A Enhancement) ─────────────────

    # Feature 21: Crosswind vs aircraft limit (0-1, higher = worse)
    max_xw = limits["max_crosswind"]
    features["crosswind_ratio"] = round(
        min(1.0, features["crosswind_kt"] / max_xw) if max_xw > 0 else 0.0, 3
    )

    # Feature 22: Headwind sufficiency (0-1, higher = better)
    # Takeoff needs sufficient headwind. Below 5kt headwind is risky
    hw = features["headwind_kt"]
    features["headwind_ratio"] = round(
        min(1.0, max(0.0, (hw - 0) / 20.0)) if hw > 0 else 0.0, 3
    )

    # Feature 23: Runway sufficiency (0-1, higher = better)
    # Is runway length adequate for aircraft?
    min_rwy = limits["min_runway_m"]
    features["runway_sufficiency"] = round(
        min(1.0, runway_length_m / min_rwy) if min_rwy > 0 else 1.0, 3
    )

    # Feature 24: Visibility adequacy (0-1, higher = better)
    # Takeoff needs minimum visibility (~1SM)
    features["visibility_adequacy"] = round(
        min(1.0, visibility_sm / 5.0), 3
    )

    # Feature 25: Ceiling adequacy (0-1, higher = better)
    # Takeoff needs minimum ceiling (~500ft)
    features["ceiling_adequacy"] = round(
        min(1.0, ceiling_ft / 3000.0), 3
    )

    # Feature 26: Surface friction penalty (0-1, higher = worse)
    # Wet/contaminated surfaces reduce available runway
    if surface_condition == "CONTAMINATED":
        features["surface_friction_penalty"] = 0.3
    elif surface_condition == "WET":
        features["surface_friction_penalty"] = 0.15
    else:
        features["surface_friction_penalty"] = 0.0

    # Feature 27: Wind gust intensity (0-1)
    # Gust relative to sustained wind
    if wind_speed_kt > 0:
        features["gust_intensity_ratio"] = round(
            min(1.0, (wind_gust_kt - wind_speed_kt) / 10.0), 3
        )
    else:
        features["gust_intensity_ratio"] = 0.0

    # Feature 28: Traffic intensity (0-1)
    # Queue length relative to airport capacity (assume max 15 in queue)
    features["traffic_intensity"] = round(
        min(1.0, traffic_count / 15.0), 3
    )

    # Feature 29: NOTAM complexity (0-1)
    # Number of active NOTAMs relative to runway state
    features["notam_complexity"] = round(
        min(1.0, active_notams / 5.0), 3
    )

    # Feature 30: TIME OF DAY (peak departure hours risk)
    # 7-9am and 4-6pm are peak hours (higher traffic complexity)
    if hour_of_day in [7, 8, 9, 16, 17, 18]:
        features["peak_hour_risk"] = 0.3
    elif hour_of_day in [6, 10, 15, 19]:
        features["peak_hour_risk"] = 0.15
    else:
        features["peak_hour_risk"] = 0.0

    # Feature 31: COMPOSITE RISK SCORE (0-1, the key feature)
    # This is the master feature that combines all decision factors
    features["composite_risk_score"] = compute_decision_risk_score(
        features["crosswind_kt"],
        features["headwind_kt"],
        ceiling_ft,
        visibility_sm,
        aircraft_type,
        runway_length_m,
        surface_condition,
    )

    # Feature 32: GO/NO-GO decision boundary (expert rules)
    # Hard decision thresholds based on aircraft limits
    composite_risk = features["composite_risk_score"]
    if composite_risk > 0.6:
        features["decision_boundary"] = 2.0  # NO-GO
    elif composite_risk > 0.3:
        features["decision_boundary"] = 1.0  # CAUTION
    else:
        features["decision_boundary"] = 0.0  # GO

    return features


def get_feature_names_enhanced() -> List[str]:
    """Return ordered list of all 32 feature names."""
    return [
        # Base 8
        "wind_speed_kt",
        "wind_gust_kt",
        "visibility_sm",
        "ceiling_ft",
        "temp_c",
        "crosswind_kt",
        "active_notams",
        "traffic_count",
        # Derived 12
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
        # NEW: Decision-critical 12
        "crosswind_ratio",           # Feature 21
        "headwind_ratio",            # Feature 22
        "runway_sufficiency",        # Feature 23
        "visibility_adequacy",       # Feature 24
        "ceiling_adequacy",          # Feature 25
        "surface_friction_penalty",  # Feature 26
        "gust_intensity_ratio",      # Feature 27
        "traffic_intensity",         # Feature 28
        "notam_complexity",          # Feature 29
        "peak_hour_risk",            # Feature 30
        "composite_risk_score",      # Feature 31 (KEY)
        "decision_boundary",         # Feature 32 (Hard thresholds)
    ]
