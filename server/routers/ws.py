"""
WebSocket router — real-time weather alert push to connected ATC clients.

Clients connect to  ws://localhost:8000/ws/alerts
The server broadcasts a JSON alert whenever a METAR poll detects:
  - Wind speed > 35 kt
  - Visibility < 1 SM
  - Ceiling < 500 ft
  - Flight category transitions (VFR → IFR, etc.)
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Dict, Optional, Set

import httpx
import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["WebSocket"])
log = structlog.get_logger()

# ── Connection manager ────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)
        log.info("ws_client_connected", total=len(self.active))

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)
        log.info("ws_client_disconnected", total=len(self.active))

    async def broadcast(self, message: Dict[str, Any]):
        payload = json.dumps(message)
        dead = set()
        for ws in list(self.active):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active.discard(ws)


manager = ConnectionManager()

# ── METAR polling (background task) ──────────────────────────────────────────
NOAA_URL = "https://aviationweather.gov/api/data/metar"
POLL_INTERVAL = 60          # seconds between polls
MONITORED_ICAOS = ["EGLL", "KJFK", "VCBI", "OMDB"]  # extend via env in prod

_wind_re = re.compile(r"(\d{3}|VRB)(\d{2,3})(G(\d{2,3}))?KT")
_vis_sm_re = re.compile(r"\b(\d+(?:/\d+)?)\s*SM\b")
_sky_re = re.compile(r"\b(BKN|OVC)(\d{3})\b")
_fc_re = re.compile(r"\b(VFR|MVFR|IFR|LIFR)\b")

_prev_cat: Dict[str, str] = {}


def _parse_quick(raw: str) -> Dict[str, Any]:
    wind_m = _wind_re.search(raw)
    wind_kt = int(wind_m.group(2)) if wind_m else 0
    gust_kt = int(wind_m.group(4)) if wind_m and wind_m.group(4) else wind_kt

    vis_m = _vis_sm_re.search(raw)
    vis = float(vis_m.group(1).split("/")[0]) if vis_m else 10.0

    ceil = 9999
    for m in _sky_re.finditer(raw):
        ceil = min(ceil, int(m.group(2)) * 100)

    fc_m = _fc_re.search(raw)
    fc = fc_m.group(1) if fc_m else "VFR"

    return {"wind_kt": wind_kt, "gust_kt": gust_kt, "vis_sm": vis, "ceiling_ft": ceil, "fc": fc}


async def _poll_weather():
    """Background coroutine: polls NOAA every POLL_INTERVAL seconds."""
    while True:
        for icao in MONITORED_ICAOS:
            if not manager.active:
                break
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    r = await client.get(
                        NOAA_URL, params={"ids": icao, "format": "raw", "hours": "1"}
                    )
                    r.raise_for_status()
                    raw = r.text.strip()

                if not raw:
                    continue

                p = _parse_quick(raw)
                alerts = []

                if p["wind_kt"] > 35:
                    alerts.append(f"Wind {p['wind_kt']} kt exceeds 35 kt limit")
                if p["gust_kt"] > 45:
                    alerts.append(f"Gusts {p['gust_kt']} kt — severe turbulence risk")
                if p["vis_sm"] < 1.0:
                    alerts.append(f"Visibility {p['vis_sm']} SM — below 1 SM minima")
                if p["ceiling_ft"] < 500:
                    alerts.append(f"Ceiling {p['ceiling_ft']} ft — below Category I minima")

                prev = _prev_cat.get(icao, p["fc"])
                if prev != p["fc"]:
                    alerts.append(
                        f"Flight category changed {prev} → {p['fc']}"
                    )
                _prev_cat[icao] = p["fc"]

                if alerts:
                    await manager.broadcast(
                        {
                            "type": "weather_alert",
                            "icao": icao,
                            "severity": "critical" if any("severe" in a or "below" in a for a in alerts) else "warning",
                            "alerts": alerts,
                            "metar_summary": {
                                "wind_kt": p["wind_kt"],
                                "vis_sm": p["vis_sm"],
                                "ceiling_ft": p["ceiling_ft"],
                                "flight_category": p["fc"],
                            },
                            "timestamp": time.time(),
                        }
                    )

            except Exception as exc:
                log.warning("ws_poll_error", icao=icao, error=str(exc))

        await asyncio.sleep(POLL_INTERVAL)


# Start background polling task on first client connection
_poll_task: Optional[asyncio.Task] = None


@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    global _poll_task
    await manager.connect(websocket)

    if _poll_task is None or _poll_task.done():
        _poll_task = asyncio.create_task(_poll_weather())
        log.info("ws_poll_task_started")

    # Send immediate welcome / heartbeat
    await websocket.send_text(
        json.dumps({"type": "connected", "message": "Safe TakeOff alert stream active"})
    )

    try:
        while True:
            # Keep connection alive; handle pings from client
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
