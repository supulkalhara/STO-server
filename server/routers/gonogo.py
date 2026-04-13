"""
Go/No-Go decision router — XGBoost classifier with SHAP explainability.

The model is trained on a synthetic but domain-realistic dataset the first time
the server starts, then cached in memory.  In production, swap the training
block for `xgb.load_model("gonogo.ubj")` loaded from object storage.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import shap
import structlog
import xgboost as xgb
from fastapi import APIRouter, HTTPException, Request
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from schemas.gonogo import GoNoGoRequest, GoNoGoResponse, ShapFactor

router = APIRouter(prefix="/gonogo", tags=["Go/No-Go"])
log = structlog.get_logger()

# ── Feature order — must match training and inference ─────────────────────────
FEATURE_NAMES = [
    "wind_speed_kt",
    "wind_gust_kt",
    "visibility_sm",
    "ceiling_ft",
    "temp_c",
    "crosswind_kt",
    "active_notams",
    "traffic_count",
]

# ── Singleton model (lazily trained on first call) ────────────────────────────
_pipeline: Optional[Pipeline] = None
_explainer: Optional[shap.TreeExplainer] = None


def _build_synthetic_dataset(n: int = 2000):
    """
    Generate a synthetic training set using aviation meteorological rules:
    - High wind/gust, low visibility, low ceiling → No-Go
    - Moderate conditions → Caution (class 1)
    - Good conditions → Go (class 0)
    """
    rng = np.random.default_rng(42)

    wind = rng.uniform(0, 60, n)
    gust = wind + rng.uniform(0, 20, n)
    vis = rng.uniform(0, 10, n)
    ceil = rng.uniform(0, 5000, n)
    temp = rng.uniform(-30, 45, n)
    xwind = rng.uniform(0, 35, n)
    notams = rng.integers(0, 10, n)
    traffic = rng.integers(0, 30, n)

    X = np.column_stack([wind, gust, vis, ceil, temp, xwind, notams, traffic])

    # Label: 0=GO, 1=CAUTION, 2=NO-GO
    y = np.zeros(n, dtype=int)
    caution = (wind > 20) | (vis < 3) | (ceil < 1000) | (xwind > 15)
    nogo = (wind > 40) | (vis < 1) | (ceil < 300) | (xwind > 25) | (notams > 5)
    y[caution] = 1
    y[nogo] = 2

    return X, y


def _get_model() -> tuple[Optional[Pipeline], Optional[shap.TreeExplainer]]:
    global _pipeline, _explainer
    if _pipeline is not None:
        return _pipeline, _explainer

    log.info("gonogo_model_training_start")
    X, y = _build_synthetic_dataset()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        num_class=3,
        objective="multi:softprob",
    )
    clf.fit(X_scaled, y)

    _explainer = shap.TreeExplainer(clf)
    _pipeline = Pipeline([("scaler", scaler), ("clf", clf)])
    log.info("gonogo_model_ready")
    return _pipeline, _explainer


def _request_to_features(req: GoNoGoRequest) -> np.ndarray:
    return np.array([[
        req.wind_speed_kt,
        req.wind_gust_kt or req.wind_speed_kt,
        req.visibility_sm,
        req.ceiling_ft if req.ceiling_ft is not None else 5000.0,
        req.temp_c,
        req.crosswind_kt,
        req.active_notams,
        req.traffic_count,
    ]])


@router.post("/predict", response_model=GoNoGoResponse)
async def predict_gonogo(request: Request, payload: GoNoGoRequest):
    """
    Run the Go/No-Go XGBoost classifier and return a decision with SHAP
    top-5 risk factors.
    """
    try:
        pipeline, explainer = _get_model()
    except Exception as exc:
        log.error("gonogo_model_error", error=str(exc))
        raise HTTPException(status_code=500, detail="ML model unavailable")

    X_raw = _request_to_features(payload)
    X_scaled = pipeline["scaler"].transform(X_raw)

    # Predict probabilities
    proba = pipeline["clf"].predict_proba(X_scaled)[0]  # [P(GO), P(CAUTION), P(NO-GO)]
    class_idx = int(np.argmax(proba))
    confidence = float(proba[class_idx])
    risk_score = float(proba[1] * 0.4 + proba[2] * 1.0)   # weighted risk

    decision_map = {0: "GO", 1: "CAUTION", 2: "NO-GO"}
    decision = decision_map[class_idx]

    # SHAP values for the predicted class
    shap_all = explainer.shap_values(X_scaled)
    
    # Handle both list (one per class) and array types
    if isinstance(shap_all, list):
        sv_class = shap_all[class_idx]
    else:
        # Some versions return (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
        if len(shap_all.shape) == 3:
            # Check if classes are at axis 0 or axis 2
            if shap_all.shape[0] == 3: # 3 classes
                sv_class = shap_all[class_idx]
            else:
                sv_class = shap_all[:, :, class_idx]
        else:
            sv_class = shap_all

    # Ensure we take the first sample's SHAP values and it's 1D
    sv = sv_class[0] if len(sv_class.shape) > 1 else sv_class

    factors: list[ShapFactor] = []
    for i, (fname, fval, sval) in enumerate(zip(FEATURE_NAMES, X_raw[0], sv)):
        factors.append(
            ShapFactor(
                feature=fname,
                value=float(fval),
                shap_value=float(sval),
                direction="increases_risk" if sval > 0 else "reduces_risk",
            )
        )

    # Top 5 by |shap_value|
    top5 = sorted(factors, key=lambda f: abs(f.shap_value), reverse=True)[:5]

    # Human-readable explanation
    risk_words = [f.feature.replace("_", " ") for f in top5 if f.direction == "increases_risk"]
    if risk_words:
        explanation = f"Decision driven mainly by: {', '.join(risk_words[:3])}."
    else:
        explanation = "All parameters within acceptable bounds."

    log.info(
        "gonogo_prediction",
        icao=payload.icao,
        decision=decision,
        confidence=round(confidence, 3),
    )
    return GoNoGoResponse(
        icao=payload.icao.upper(),
        decision=decision,
        confidence=confidence,
        risk_score=min(risk_score, 1.0),
        top_factors=top5,
        explanation=explanation,
    )
