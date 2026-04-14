"""
Decision History Model — stores ATC Go/No-Go decisions for learning.

Each record captures:
- Request snapshot (features at decision time)
- Model predictions (XGBoost + Claude agent)
- Ground truth (ATC actual decision + outcome from radar/logs)

Used for:
- RAG context injection (historical similar decisions)
- XGBoost retraining (supervised learning)
- Model evaluation (accuracy metrics per aerodrome)
"""
from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, Index
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import relationship

from database import Base


class DecisionHistory(Base):
    __tablename__ = "decision_history"

    # ── Primary key & foreign keys ───────────────────────────────────────────
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    flight_plan_id = Column(Integer, ForeignKey("flight_plans.id"), nullable=True)

    # ── Request snapshot (reproducibility) ────────────────────────────────────
    icao = Column(String(4), nullable=False)  # departure aerodrome
    callsign = Column(String(10), nullable=False)
    aircraft_type = Column(String(10), nullable=False)  # ICAO type (B738, A320, etc)

    # ── Base 8 features (from METAR + airport state) ──────────────────────────
    wind_speed_kt = Column(Float, nullable=True)
    wind_gust_kt = Column(Float, nullable=True)
    visibility_sm = Column(Float, nullable=True)  # statute miles
    ceiling_ft = Column(Float, nullable=True)  # above ground level
    temp_c = Column(Float, nullable=True)
    crosswind_kt = Column(Float, nullable=True)  # computed: perpendicular to runway
    active_notams = Column(Integer, nullable=True)  # count
    traffic_count = Column(Integer, nullable=True)  # concurrent departures in queue

    # ── Derived features (engineered for ML) ──────────────────────────────────
    runway_length_m = Column(Float, nullable=True)
    surface_condition = Column(String(20), nullable=True)  # DRY, WET, CONTAMINATED
    metar_category = Column(String(10), nullable=True)  # VFR, MVFR, IFR, LIFR
    tailwind_kt = Column(Float, nullable=True)
    headwind_kt = Column(Float, nullable=True)
    relative_humidity = Column(Float, nullable=True)  # %
    aircraft_mtow_ratio = Column(Float, nullable=True)  # payload_kg / mtow_kg
    wake_count_heavy = Column(Integer, nullable=True)  # H/J departures in last 30min
    metar_age_min = Column(Float, nullable=True)  # freshness of METAR
    notam_severity_index = Column(Float, nullable=True)  # weighted by priority
    ceiling_visibility_ratio = Column(Float, nullable=True)
    temperature_trend = Column(Float, nullable=True)  # degrees vs historical avg

    # ── XGBoost predictions (Phase 2 model) ───────────────────────────────────
    xgboost_decision = Column(String(20), nullable=True)  # GO, CAUTION, NO-GO
    xgboost_confidence = Column(Float, nullable=True)  # 0.0-1.0
    xgboost_risk_score = Column(Float, nullable=True)  # 0.0-1.0

    # ── Agent (Claude) predictions (Phase 3 model) ────────────────────────────
    agent_decision = Column(String(20), nullable=True)  # GO, CAUTION, NO-GO
    agent_confidence = Column(Float, nullable=True)  # 0.0-1.0
    agent_reasoning = Column(Text, nullable=True)  # Full explanation from Claude

    # ── Ground truth (feedback from ATC) ──────────────────────────────────────
    # These are populated later by ATC supervisor via /api/decisions/{id}/feedback
    atc_decision = Column(String(20), nullable=True)  # GO, CAUTION, NO-GO (what ATC actually did)
    outcome = Column(String(30), nullable=True)  # TAKEOFF_SUCCESS, DELAY, CANCELLED, DIVERTED
    outcome_reason = Column(Text, nullable=True)  # Why the outcome occurred (for context)

    # ── Audit trail ──────────────────────────────────────────────────────────
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    feedback_at = Column(DateTime, nullable=True)  # When ATC provided outcome
    request_id = Column(String(36), nullable=True)  # X-Request-ID header (tracing)

    # ── Indexes for common queries ───────────────────────────────────────────
    __table_args__ = (
        Index("ix_user_id", "user_id"),
        Index("ix_icao", "icao"),
        Index("ix_created_at", "created_at"),
        Index("ix_feedback_at", "feedback_at"),
        Index("ix_aircraft_type", "aircraft_type"),
    )

    def __repr__(self):
        return (
            f"<DecisionHistory(id={self.id}, callsign={self.callsign}, "
            f"xgboost={self.xgboost_decision}, agent={self.agent_decision}, "
            f"outcome={self.outcome})>"
        )

    def to_dict(self):
        """Serialize to JSON-friendly dict (for API responses)."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "flight_plan_id": self.flight_plan_id,
            "icao": self.icao,
            "callsign": self.callsign,
            "aircraft_type": self.aircraft_type,
            # Features
            "features": {
                "wind_speed_kt": self.wind_speed_kt,
                "wind_gust_kt": self.wind_gust_kt,
                "visibility_sm": self.visibility_sm,
                "ceiling_ft": self.ceiling_ft,
                "temp_c": self.temp_c,
                "crosswind_kt": self.crosswind_kt,
                "active_notams": self.active_notams,
                "traffic_count": self.traffic_count,
                "runway_length_m": self.runway_length_m,
                "surface_condition": self.surface_condition,
                "metar_category": self.metar_category,
                "tailwind_kt": self.tailwind_kt,
                "headwind_kt": self.headwind_kt,
                "relative_humidity": self.relative_humidity,
                "aircraft_mtow_ratio": self.aircraft_mtow_ratio,
                "wake_count_heavy": self.wake_count_heavy,
                "metar_age_min": self.metar_age_min,
                "notam_severity_index": self.notam_severity_index,
                "ceiling_visibility_ratio": self.ceiling_visibility_ratio,
                "temperature_trend": self.temperature_trend,
            },
            # Predictions
            "xgboost": {
                "decision": self.xgboost_decision,
                "confidence": self.xgboost_confidence,
                "risk_score": self.xgboost_risk_score,
            },
            "agent": {
                "decision": self.agent_decision,
                "confidence": self.agent_confidence,
                "reasoning": self.agent_reasoning,
            },
            # Ground truth
            "outcome": {
                "atc_decision": self.atc_decision,
                "outcome": self.outcome,
                "outcome_reason": self.outcome_reason,
            },
            # Audit
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "feedback_at": self.feedback_at.isoformat() if self.feedback_at else None,
            "request_id": self.request_id,
        }
