from __future__ import annotations
"""
Database setup — SQLAlchemy with SQLite for development.
Swap DATABASE_URL to postgresql+asyncpg://... for production.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from passlib.context import CryptContext

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./safetakeoff_dev.db")

# SQLite needs connect_args; remove for PostgreSQL
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: yields a DB session, always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables and seed a demo ATC user + fleet on first run."""
    from models.user import User  # noqa: import here to avoid circular
    from models.aircraft import Aircraft  # noqa
    from models.flightplan import FlightPlan  # noqa

    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # Seed demo user
        if not db.query(User).filter(User.email == "atc@safetakeoff.dev").first():
            demo_user = User(
                email="atc@safetakeoff.dev",
                hashed_password=pwd_context.hash("SafeTakeOff2026!"),
                full_name="Demo ATC Officer",
                role="atc_officer",
                is_active=True,
            )
            db.add(demo_user)
            db.commit()

        # Seed aircraft fleet
        if db.query(Aircraft).count() == 0:
            seed_aircraft = [
                Aircraft(
                    icao_type_designator="A333", registration="4R-ALN",
                    callsign="ALK201", operator="SriLankan Airlines",
                    wake_turbulence_category="H", engine_type="Jet",
                    mtow_kg=242000, v1_kts=148, vr_kts=155, v2_kts=162,
                    equipment_suffixes="SDE2FGHIJ2RWY", rnav_approved="Y", rvsm_approved="Y",
                ),
                Aircraft(
                    icao_type_designator="B77W", registration="A6-ENA",
                    callsign="UAE412", operator="Emirates",
                    wake_turbulence_category="H", engine_type="Jet",
                    mtow_kg=351500, v1_kts=155, vr_kts=162, v2_kts=170,
                    equipment_suffixes="SDE2E3FGHIJ3J5M1RWXY", rnav_approved="Y", rvsm_approved="Y",
                ),
                Aircraft(
                    icao_type_designator="A388", registration="A6-EUV",
                    callsign="UAE2", operator="Emirates",
                    wake_turbulence_category="J", engine_type="Jet",
                    mtow_kg=575000, v1_kts=150, vr_kts=158, v2_kts=166,
                    equipment_suffixes="SDE2E3FGHIJ3J5M1RWXY", rnav_approved="Y", rvsm_approved="Y",
                ),
                Aircraft(
                    icao_type_designator="B738", registration="4R-ABM",
                    callsign="ALK504", operator="SriLankan Airlines",
                    wake_turbulence_category="M", engine_type="Jet",
                    mtow_kg=79016, v1_kts=140, vr_kts=148, v2_kts=153,
                    equipment_suffixes="SDE2FGHIJ2RWY", rnav_approved="Y", rvsm_approved="Y",
                ),
                Aircraft(
                    icao_type_designator="AT76", registration="4R-ATE",
                    callsign="ALK3201", operator="SriLankan Airlines",
                    wake_turbulence_category="M", engine_type="Turboprop",
                    mtow_kg=23000, v1_kts=105, vr_kts=110, v2_kts=115,
                    equipment_suffixes="SDE2FGIRWY", rnav_approved="Y", rvsm_approved="N",
                ),
                Aircraft(
                    icao_type_designator="B744", registration="G-BNLY",
                    callsign="BAW15", operator="British Airways",
                    wake_turbulence_category="H", engine_type="Jet",
                    mtow_kg=412775, v1_kts=160, vr_kts=170, v2_kts=178,
                    equipment_suffixes="SDE2E3FGHIJ3RWXY", rnav_approved="Y", rvsm_approved="Y",
                ),
                Aircraft(
                    icao_type_designator="C172", registration="4R-CES",
                    callsign="CES01", operator="Ceylon Flying School",
                    wake_turbulence_category="L", engine_type="Piston",
                    mtow_kg=1111, v1_kts=55, vr_kts=55, v2_kts=65,
                    equipment_suffixes="S", rnav_approved="N", rvsm_approved="N",
                ),
                Aircraft(
                    icao_type_designator="A321", registration="VT-IHA",
                    callsign="IGO234", operator="IndiGo",
                    wake_turbulence_category="M", engine_type="Jet",
                    mtow_kg=93500, v1_kts=145, vr_kts=152, v2_kts=158,
                    equipment_suffixes="SDE2FGHIJ2RWY", rnav_approved="Y", rvsm_approved="Y",
                ),
            ]
            db.add_all(seed_aircraft)
            db.commit()
    finally:
        db.close()
