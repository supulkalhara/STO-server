from __future__ import annotations
"""SQLAlchemy Aircraft model — aviation-domain fields."""

from sqlalchemy import Column, Float, Integer, String
from database import Base


class Aircraft(Base):
    __tablename__ = "aircraft"

    id = Column(Integer, primary_key=True, index=True)

    # ICAO identification
    icao_type_designator = Column(String, nullable=False, index=True)  # e.g. "B738"
    registration = Column(String, unique=True, nullable=False, index=True)  # e.g. "4R-ALM"
    callsign = Column(String, nullable=False)  # e.g. "SLK201"
    operator = Column(String, nullable=True)  # e.g. "SriLankan Airlines"

    # Performance characteristics
    wake_turbulence_category = Column(String, nullable=False)  # L | M | H | J (super-heavy)
    engine_type = Column(String, nullable=False)   # Jet | Turboprop | Piston | Electric
    mtow_kg = Column(Float, nullable=True)         # Max Takeoff Weight in kg
    v1_kts = Column(Float, nullable=True)          # Takeoff decision speed (kts)
    vr_kts = Column(Float, nullable=True)          # Rotation speed (kts)
    v2_kts = Column(Float, nullable=True)          # Safety speed (kts)

    # Equipment / approvals
    equipment_suffixes = Column(String, nullable=True)  # e.g. "S" = SDE2FGHIJ (ICAO FPL item 10)
    rnav_approved = Column(String, default="N")         # Y | N
    rvsm_approved = Column(String, default="N")         # Y | N

    # Status
    is_active = Column(Integer, default=1)  # 1 = operational, 0 = AOG/maintenance
