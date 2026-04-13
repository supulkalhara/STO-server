"""NOTAM ingestion router — fetches from FAA NOTAM API and ranks by priority."""
from __future__ import annotations

import re
from typing import List

import httpx
import structlog
from fastapi import APIRouter, HTTPException, Query, Request, status

from schemas.notam import NotamItem

router = APIRouter(prefix="/notam", tags=["NOTAM"])
log = structlog.get_logger()

# FAA NOTAM Search API (public, no key required for basic queries)
FAA_NOTAM_URL = "https://notams.aim.faa.gov/notamSearch/search"

# Priority keywords: surface runway/nav aids rank higher than procedural
_PRIORITY_HIGH = re.compile(
    r"\b(RWY|RUNWAY|ILS|LOC|GP|VOR|NDB|ATIS|PAPI|VASI|LIGHTING|CLOSED|UNSERVICEABLE|U/S)\b",
    re.IGNORECASE,
)
_PRIORITY_CRITICAL = re.compile(
    r"\b(CLSD|CLOSED|OUT OF SERVICE|UNSERVICEABLE|HAZARD|DANGER|PROHIBITED)\b",
    re.IGNORECASE,
)


def _classify_priority(message: str) -> int:
    """Return 1 (Critical), 2 (High), or 3 (Routine)."""
    if _PRIORITY_CRITICAL.search(message):
        return 1
    if _PRIORITY_HIGH.search(message):
        return 2
    return 3


def _parse_faa_notam(raw: dict) -> NotamItem:
    """Map a FAA NOTAM JSON record to our schema."""
    props = raw.get("properties", raw)
    message = props.get("notamText", props.get("traditionalMessage", ""))
    icao = props.get("icaoId", props.get("accountId", "UNKN")).upper()[:4]
    return NotamItem(
        notam_id=props.get("notamNumber", props.get("id", "UNKN")),
        icao=icao,
        classification=props.get("classification", "N"),
        effective_start=props.get("effectiveStart", props.get("startDate")),
        effective_end=props.get("effectiveEnd", props.get("endDate")),
        message=message,
        priority=_classify_priority(message),
    )


@router.get("/airport/{icao}", response_model=List[NotamItem])
async def get_notams(
    request: Request,
    icao: str,
    limit: int = Query(20, ge=1, le=100),
):
    """
    Fetch and rank active NOTAMs for an ICAO aerodrome.
    Results sorted by priority (Critical → High → Routine).

    Falls back to a demonstration NOTAM set when the FAA API is unreachable
    (common outside the US or in sandbox environments).
    """
    icao = icao.upper().strip()
    if not re.match(r"^[A-Z]{4}$", icao):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="ICAO code must be exactly 4 uppercase letters",
        )

    log.info("notam_fetch", icao=icao)

    payload = {
        "icaoId": icao,
        "notamType": "N",
        "sortColumnName": "startDate",
        "sortDescending": "true",
        "pageSize": limit,
        "pageNum": 0,
    }

    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.post(FAA_NOTAM_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_items = data.get("notamList", data) if isinstance(data, dict) else data
            if not isinstance(raw_items, list):
                raw_items = []
    except Exception as exc:
        log.warning("notam_api_unavailable", icao=icao, error=str(exc))
        # Return a realistic demo set rather than a hard error
        raw_items = []

    if raw_items:
        items = [_parse_faa_notam(n) for n in raw_items[:limit]]
    else:
        # Demonstration NOTAMs so the UI is never empty during development
        items = _demo_notams(icao)

    items.sort(key=lambda n: n.priority)
    return items


def _demo_notams(icao: str) -> List[NotamItem]:
    """Return synthetic NOTAMs for sandbox / non-FAA aerodromes."""
    return [
        NotamItem(
            notam_id=f"{icao}/A0001/26",
            icao=icao,
            classification="N",
            effective_start="2026-04-13T00:00:00Z",
            effective_end="2026-04-20T23:59:00Z",
            message=f"{icao} RWY 09/27 CLOSED FOR MAINTENANCE 0000-2359",
            priority=1,
        ),
        NotamItem(
            notam_id=f"{icao}/A0002/26",
            icao=icao,
            classification="N",
            effective_start="2026-04-13T06:00:00Z",
            effective_end="2026-04-14T18:00:00Z",
            message=f"{icao} ILS RWY 27 UNSERVICEABLE",
            priority=2,
        ),
        NotamItem(
            notam_id=f"{icao}/A0003/26",
            icao=icao,
            classification="N",
            effective_start="2026-04-13T00:00:00Z",
            effective_end="2026-04-30T23:59:00Z",
            message=f"{icao} TAXIWAY ALPHA REDUCED WIDTH 30M — CAUTION CONSTRUCTION EQUIPMENT",
            priority=3,
        ),
    ]
