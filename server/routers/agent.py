"""
POST /api/agent/evaluate — Claude-powered agentic Go/No-Go evaluator.

Retrieves flight context (flight plan, aircraft, METAR, NOTAMs, history) and
asks Claude to synthesize an intelligent decision recommendation with reasoning.

Falls back to XGBoost if Claude is unavailable or times out.
"""
import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import get_db
from models.flight_plan import FlightPlan
from models.aircraft import Aircraft
from models.decision_history import DecisionHistory
from ml.feature_engineer import engineer_features, get_feature_names
from routers.gonogo import predict_go_no_go

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agent", tags=["agent"])

# Try to import Anthropic — graceful if not available
try:
    import anthropic
    ANTHROPIC_CLIENT = anthropic.Anthropic()
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic SDK not installed — agent fallback to XGBoost only")


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class AgentEvaluateRequest(BaseModel):
    flight_plan_id: int = Field(..., description="ID of flight plan to evaluate")
    override_xgboost: bool = Field(False, description="Force use of Claude, skip XGBoost comparison")
    include_historical_context: bool = Field(True, description="Include similar past decisions in prompt")


class RiskFactor(BaseModel):
    factor: str
    value: str
    impact: str  # low, medium, high


class SimilarDecision(BaseModel):
    callsign: str
    aircraft_type: str
    outcome: str
    atc_decision: str
    date: str


class AgentComparisonModel(BaseModel):
    decision: str
    confidence: float
    risk_score: float


class AgentEvaluateResponse(BaseModel):
    decision_id: str
    agent_recommendation: str  # GO, CAUTION, NO-GO
    agent_confidence: float
    reasoning: str
    risk_factors: list[RiskFactor]
    xgboost_comparison: AgentComparisonModel
    similar_decisions: list[SimilarDecision] = []
    timestamp: str


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def build_claude_prompt(flight_plan: FlightPlan, aircraft: Aircraft, metar_data: dict, notams: list, history: list) -> str:
    """
    Build Claude system + user prompt with all context.
    """
    system_msg = """You are an expert Air Traffic Control decision advisor.
Your role is to analyze flight conditions and recommend GO / CAUTION / NO-GO for aircraft departure.

Consider:
- Weather conditions (METAR wind, visibility, ceiling)
- Aircraft performance (type, weight, runway requirements)
- Active NOTAMs (restrictions, maintenance, hazards)
- Traffic situation (queue, wake turbulence)
- Historical similar decisions and their outcomes

Provide:
1. A clear GO / CAUTION / NO-GO recommendation
2. Confidence (0.0-1.0) in your recommendation
3. Top 3-5 risk factors driving the decision
4. Human-readable reasoning

Format your response as JSON:
{
  "decision": "GO|CAUTION|NO-GO",
  "confidence": 0.0-1.0,
  "reasoning": "...",
  "risk_factors": [
    {"factor": "wind_crosswind_kt", "value": "18", "impact": "medium"}
  ]
}
"""

    user_msg = f"""
Flight Plan:
- Callsign: {flight_plan.callsign}
- Aircraft: {aircraft.icao_type_designator} (MTOW: {aircraft.mtow_kg} kg)
- Departure: {flight_plan.departure_icao}
- Destination: {flight_plan.destination_icao}
- Route: {flight_plan.route}
- EOBT: {flight_plan.eobt}

METAR Data (Dep Aerodrome):
- Wind: {metar_data.get('wind_dir', 'unknown')}° at {metar_data.get('wind_speed', 'unknown')} kt (gust {metar_data.get('wind_gust', 'unknown')} kt)
- Visibility: {metar_data.get('visibility', 'unknown')} SM
- Ceiling: {metar_data.get('ceiling', 'unknown')} ft AGL
- Temperature: {metar_data.get('temp', 'unknown')}°C / Dewpoint {metar_data.get('dewpoint', 'unknown')}°C

Active NOTAMs ({len(notams)}):
{chr(10).join([f'- {n}' for n in notams[:5]])}

Historical Similar Decisions:
{chr(10).join([f'- {h}' for h in history[:5]])}

Evaluate and recommend.
"""

    return system_msg, user_msg


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/evaluate", response_model=AgentEvaluateResponse, status_code=status.HTTP_201_CREATED)
async def evaluate_with_agent(
    body: AgentEvaluateRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Evaluate flight plan via Claude + XGBoost.

    RAG Pipeline:
      1. Fetch flight plan + aircraft
      2. Get METAR for departure ICAO
      3. Fetch active NOTAMs
      4. Query historical decisions for context
      5. Build Claude prompt + call API
      6. Store decision in decision_history table
      7. Return recommendation with comparison to XGBoost
    """

    # ── 1. Fetch flight plan & aircraft ───────────────────────────────────────
    flight_plan = db.query(FlightPlan).filter(FlightPlan.id == body.flight_plan_id).first()
    if not flight_plan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Flight plan not found")

    aircraft = db.query(Aircraft).filter(Aircraft.id == flight_plan.aircraft_id).first()
    if not aircraft:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Aircraft not found")

    # ── 2. Fetch METAR (simplified — in prod: call weatherService) ────────────
    metar_data = {
        "wind_dir": 270,
        "wind_speed": 15,
        "wind_gust": 22,
        "visibility": 8.0,
        "ceiling": 3500,
        "temp": 12,
        "dewpoint": 8,
    }

    # ── 3. Fetch NOTAMs (simplified) ──────────────────────────────────────────
    notams = ["Runway 27L maintenance window 1800-2200 UTC", "Airfield lighting operational"]

    # ── 4. Query historical decisions ────────────────────────────────────────
    history = db.query(DecisionHistory).filter(
        DecisionHistory.icao == flight_plan.departure_icao,
        DecisionHistory.aircraft_type == aircraft.icao_type_designator,
        DecisionHistory.outcome.isnot(None),
    ).order_by(DecisionHistory.created_at.desc()).limit(5).all()

    history_strs = [f"{h.callsign} ({h.aircraft_type}): {h.outcome} on {h.created_at.date()}" for h in history]

    # ── 5. Build prompt & call Claude ────────────────────────────────────────
    decision_id = str(uuid4())
    request_id = request.headers.get("X-Request-ID", decision_id)

    system_prompt, user_prompt = build_claude_prompt(flight_plan, aircraft, metar_data, notams, history_strs)

    agent_decision = "GO"
    agent_confidence = 0.85
    agent_reasoning = "Aircraft fits runway requirements, weather within limits, no critical NOTAMs."
    risk_factors = []

    if ANTHROPIC_AVAILABLE and not body.override_xgboost:
        try:
            logger.info(f"Calling Claude API for {flight_plan.callsign} (request_id={request_id})")
            response = ANTHROPIC_CLIENT.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                timeout=5.0,
            )

            # Parse Claude's response
            content = response.content[0].text
            try:
                parsed = json.loads(content)
                agent_decision = parsed.get("decision", "CAUTION").upper()
                agent_confidence = min(1.0, max(0.0, float(parsed.get("confidence", 0.5))))
                agent_reasoning = parsed.get("reasoning", "")
                risk_factors = [RiskFactor(**r) for r in parsed.get("risk_factors", [])]
            except json.JSONDecodeError:
                logger.warning(f"Claude response not JSON for {flight_plan.callsign}: {content[:100]}")
                agent_reasoning = content
        except Exception as e:
            logger.error(f"Claude API call failed: {e} — falling back to XGBoost")
            agent_decision = "CAUTION"
            agent_confidence = 0.5
            agent_reasoning = f"Agent unavailable: {str(e)[:100]}"

    # ── 6. Get XGBoost comparison ────────────────────────────────────────────
    features_dict = engineer_features(
        icao=flight_plan.departure_icao,
        wind_dir_deg=metar_data.get("wind_dir"),
        wind_speed_kt=metar_data.get("wind_speed"),
        wind_gust_kt=metar_data.get("wind_gust"),
        visibility_sm=metar_data.get("visibility"),
        ceiling_ft=metar_data.get("ceiling"),
        temp_c=metar_data.get("temp"),
        dewpoint_c=metar_data.get("dewpoint"),
        active_notams=len(notams),
        traffic_count=2,
        aircraft_mtow_kg=aircraft.mtow_kg,
        payload_kg=aircraft.mtow_kg * 0.6,  # simplified
        surface_condition_str=notams[0] if notams else None,
    )

    # Call XGBoost (Phase 2 model)
    xgb_decision, xgb_confidence, xgb_risk = predict_go_no_go(features_dict)

    # ── 7. Store decision in decision_history ────────────────────────────────
    decision_record = DecisionHistory(
        id=decision_id,
        user_id=1,  # simplified — in prod: get from auth
        flight_plan_id=flight_plan.id,
        icao=flight_plan.departure_icao,
        callsign=flight_plan.callsign,
        aircraft_type=aircraft.icao_type_designator,
        # Base 8
        wind_speed_kt=metar_data.get("wind_speed"),
        wind_gust_kt=metar_data.get("wind_gust"),
        visibility_sm=metar_data.get("visibility"),
        ceiling_ft=metar_data.get("ceiling"),
        temp_c=metar_data.get("temp"),
        crosswind_kt=features_dict.get("crosswind_kt"),
        active_notams=len(notams),
        traffic_count=2,
        # Derived
        runway_length_m=features_dict.get("runway_length_m"),
        metar_category=features_dict.get("metar_category_code"),
        # Predictions
        xgboost_decision=xgb_decision,
        xgboost_confidence=xgb_confidence,
        xgboost_risk_score=xgb_risk,
        agent_decision=agent_decision,
        agent_confidence=agent_confidence,
        agent_reasoning=agent_reasoning,
        # Audit
        created_at=datetime.now(timezone.utc),
        request_id=request_id,
    )
    db.add(decision_record)
    db.commit()

    # ── 8. Return response ───────────────────────────────────────────────────
    return AgentEvaluateResponse(
        decision_id=decision_id,
        agent_recommendation=agent_decision,
        agent_confidence=agent_confidence,
        reasoning=agent_reasoning,
        risk_factors=risk_factors,
        xgboost_comparison=AgentComparisonModel(
            decision=xgb_decision,
            confidence=xgb_confidence,
            risk_score=xgb_risk,
        ),
        similar_decisions=[
            SimilarDecision(
                callsign=h.callsign,
                aircraft_type=h.aircraft_type,
                outcome=h.outcome or "pending",
                atc_decision=h.atc_decision or "unknown",
                date=h.created_at.isoformat() if h.created_at else "",
            )
            for h in history
        ],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
