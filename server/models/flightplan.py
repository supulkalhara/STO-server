"""Flight plan SQLAlchemy model."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text

from database import Base


class FlightPlan(Base):
    __tablename__ = "flight_plans"

    id = Column(Integer, primary_key=True, index=True)
    callsign = Column(String(10), nullable=False, index=True)
    aircraft_type = Column(String(10), nullable=False)
    wake_turbulence_category = Column(String(1), nullable=False)
    departure_icao = Column(String(4), nullable=False)
    destination_icao = Column(String(4), nullable=False)
    alternate_icao = Column(String(4), nullable=True)
    eobt = Column(DateTime, nullable=False)
    tobt = Column(DateTime, nullable=True)
    ctot = Column(DateTime, nullable=True)
    cruising_level = Column(String(10), nullable=False)
    route = Column(Text, nullable=True)
    remarks = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="FILED")
    created_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )
