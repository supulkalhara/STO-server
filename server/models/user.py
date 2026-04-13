from __future__ import annotations
"""SQLAlchemy User model."""

from sqlalchemy import Boolean, Column, Integer, String
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    role = Column(String, default="atc_officer")  # atc_supervisor | atc_officer | viewer
    is_active = Column(Boolean, default=True)
