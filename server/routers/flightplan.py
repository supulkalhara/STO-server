"""Flight plan management router — ICAO FPL with EOBT/TOBT/CTOT tracking."""
from __future__ import annotations

from typing import List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from database import get_db
from models.flightplan import FlightPlan
from schemas.flightplan import FlightPlanCreate, FlightPlanOut, FlightPlanUpdate

router = APIRouter(prefix="/flightplans", tags=["Flight Plans"])
log = structlog.get_logger()

VALID_STATUSES = {"FILED", "ACTIVATED", "CLOSED", "CANCELLED"}


@router.get("/", response_model=List[FlightPlanOut])
def list_flight_plans(
    departure_icao: Optional[str] = Query(None, description="Filter by departure aerodrome"),
    destination_icao: Optional[str] = Query(None, description="Filter by destination aerodrome"),
    status: Optional[str] = Query(None, description="FILED|ACTIVATED|CLOSED|CANCELLED"),
    db: Session = Depends(get_db),
):
    q = db.query(FlightPlan)
    if departure_icao:
        q = q.filter(FlightPlan.departure_icao == departure_icao.upper())
    if destination_icao:
        q = q.filter(FlightPlan.destination_icao == destination_icao.upper())
    if status:
        q = q.filter(FlightPlan.status == status.upper())
    return q.order_by(FlightPlan.eobt).all()


@router.get("/{fp_id}", response_model=FlightPlanOut)
def get_flight_plan(fp_id: int, db: Session = Depends(get_db)):
    fp = db.get(FlightPlan, fp_id)
    if not fp:
        raise HTTPException(status_code=404, detail="Flight plan not found")
    return fp


@router.post("/", response_model=FlightPlanOut, status_code=201)
def create_flight_plan(payload: FlightPlanCreate, db: Session = Depends(get_db)):
    fp = FlightPlan(**payload.model_dump())
    fp.callsign = fp.callsign.upper()
    fp.aircraft_type = fp.aircraft_type.upper()
    fp.departure_icao = fp.departure_icao.upper()
    fp.destination_icao = fp.destination_icao.upper()
    if fp.alternate_icao:
        fp.alternate_icao = fp.alternate_icao.upper()
    fp.status = "FILED"
    db.add(fp)
    db.commit()
    db.refresh(fp)
    log.info("flight_plan_created", callsign=fp.callsign, fp_id=fp.id)
    return fp


@router.patch("/{fp_id}", response_model=FlightPlanOut)
def update_flight_plan(fp_id: int, payload: FlightPlanUpdate, db: Session = Depends(get_db)):
    fp = db.get(FlightPlan, fp_id)
    if not fp:
        raise HTTPException(status_code=404, detail="Flight plan not found")
    if payload.status and payload.status.upper() not in VALID_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status. Must be one of: {', '.join(VALID_STATUSES)}",
        )
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(fp, field, value)
    db.commit()
    db.refresh(fp)
    log.info("flight_plan_updated", fp_id=fp_id, status=fp.status)
    return fp


@router.delete("/{fp_id}", status_code=204)
def cancel_flight_plan(fp_id: int, db: Session = Depends(get_db)):
    fp = db.get(FlightPlan, fp_id)
    if not fp:
        raise HTTPException(status_code=404, detail="Flight plan not found")
    fp.status = "CANCELLED"
    db.commit()
    log.info("flight_plan_cancelled", fp_id=fp_id, callsign=fp.callsign)
