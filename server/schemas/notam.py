"""NOTAM schemas."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class NotamItem(BaseModel):
    notam_id: str
    icao: str
    classification: str          # "N" | "M" | "C" | "R"
    effective_start: Optional[str] = None
    effective_end: Optional[str] = None
    message: str
    priority: int                # 1=Critical, 2=High, 3=Routine (computed)
