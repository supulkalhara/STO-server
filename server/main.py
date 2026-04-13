from __future__ import annotations
"""
Safe TakeOff — FastAPI backend  (Phase 2)
Run with: uvicorn main:app --reload --port 8000
API docs: http://localhost:8000/docs
"""

import os
import uuid

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from database import init_db
from mongodb import close_mongo
from routers import auth, aircraft, weather, notam, flightplan, gonogo, ws
from routers import signup as signup_router

# ── Structured logging setup ─────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO+
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()

# ── Rate limiter (slowapi) ────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

app = FastAPI(
    title="Safe TakeOff API",
    description="Aviation decision-support platform for ATCs — backend service",
    version="0.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ── CORS ──────────────────────────────────────────────────────────────────────
raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000")
allowed_origins = [o.strip() for o in raw_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request-ID + structured-log middleware ────────────────────────────────────
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    log.info("request", status=response.status_code)
    return response

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(signup_router.router)
app.include_router(aircraft.router)
app.include_router(weather.router)
app.include_router(notam.router)
app.include_router(flightplan.router)
app.include_router(gonogo.router)
app.include_router(ws.router)

# ── Startup / Shutdown ────────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    log.info("Initialising database …")
    init_db()
    log.info("Safe TakeOff API ready", version="0.3.0")


@app.on_event("shutdown")
async def on_shutdown():
    await close_mongo()
    log.info("Safe TakeOff API shut down")

# ── Health / readiness endpoints ──────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "version": "0.3.0"}


@app.get("/ready", tags=["System"])
def readiness_check():
    """Kubernetes-style readiness probe — extend to check DB connection."""
    return {"status": "ready"}

# ── Global exception handler ───────────────────────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. The incident has been logged."},
    )
