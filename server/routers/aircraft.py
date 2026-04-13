from __future__ import annotations
"""Aircraft CRUD router."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from database import get_db
from models.aircraft import Aircraft
from schemas.aircraft import AircraftCreate, AircraftOut, AircraftUpdate

router = APIRouter(prefix="/aircraft", tags=["Aircraft"])


@router.get("/", response_model=List[AircraftOut])
def list_aircraft(
    search: Optional[str] = Query(None, description="Filter by registration or callsign"),
    wtc: Optional[str] = Query(None, description="Filter by Wake Turbulence Category: L|M|H|J"),
    active_only: bool = Query(True, description="Return only active (non-AOG) aircraft"),
    db: Session = Depends(get_db),
):
    q = db.query(Aircraft)
    if active_only:
        q = q.filter(Aircraft.is_active == 1)
    if search:
        term = f"%{search.upper()}%"
        q = q.filter(
            Aircraft.registration.ilike(term) | Aircraft.callsign.ilike(term)
        )
    if wtc:
        q = q.filter(Aircraft.wake_turbulence_category == wtc.upper())
    return q.order_by(Aircraft.registration).all()


@router.get("/{aircraft_id}", response_model=AircraftOut)
def get_aircraft(aircraft_id: int, db: Session = Depends(get_db)):
    aircraft = db.query(Aircraft).filter(Aircraft.id == aircraft_id).first()
    if not aircraft:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Aircraft not found")
    return aircraft


@router.post("/", response_model=AircraftOut, status_code=status.HTTP_201_CREATED)
def create_aircraft(payload: AircraftCreate, db: Session = Depends(get_db)):
    # Check for duplicate registration
    existing = db.query(Aircraft).filter(
        Aircraft.registration == payload.registration.upper()
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Aircraft with registration {payload.registration} already exists",
        )
    aircraft = Aircraft(**payload.model_dump())
    aircraft.registration = aircraft.registration.upper()
    aircraft.callsign = aircraft.callsign.upper()
    aircraft.icao_type_designator = aircraft.icao_type_designator.upper()
    db.add(aircraft)
    db.commit()
    db.refresh(aircraft)
    return aircraft


@router.put("/{aircraft_id}", response_model=AircraftOut)
def update_aircraft(aircraft_id: int, payload: AircraftUpdate, db: Session = Depends(get_db)):
    aircraft = db.query(Aircraft).filter(Aircraft.id == aircraft_id).first()
    if not aircraft:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Aircraft not found")
    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(aircraft, field, value)
    db.commit()
    db.refresh(aircraft)
    return aircraft


@router.delete("/{aircraft_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_aircraft(aircraft_id: int, db: Session = Depends(get_db)):
    aircraft = db.query(Aircraft).filter(Aircraft.id == aircraft_id).first()
    if not aircraft:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Aircraft not found")
    # Soft-delete: mark as inactive rather than physically removing
    aircraft.is_active = 0
    db.commit()
