# Safe TakeOff — Backend (STO-server)

> FastAPI backend powering the Safe TakeOff ATC decision-support platform.  
> Deployed at **https://sto-server.onrender.com**

---

## What it does

Safe TakeOff is a dual-model Go / No-Go decision support system for Air Traffic Controllers. The backend provides:

- **Authentication** — JWT-based login / signup with SQLite user store
- **Weather data** — METAR parsing and ICAO aerodrome lookups
- **NOTAM digest** — active NOTAM retrieval and severity scoring
- **Flight plans** — CRUD for departure flight plans
- **Go/No-Go engine** — XGBoost model trained on 20 engineered features
- **Claude Agent** — Anthropic Claude 3.5 Sonnet overlay for reasoning + explainability
- **Decision history** — persistent audit trail with ATC feedback loop
- **Model metrics** — per-aerodrome accuracy stats (agent vs XGBoost)
- **Wake turbulence** — separation timer calculations

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| API framework | FastAPI 0.111 + Uvicorn |
| ORM / SQL | SQLAlchemy 2.0 + SQLite |
| NoSQL | MongoDB Atlas (Motor async driver) |
| ML model | XGBoost 2.0 + scikit-learn 1.5 |
| AI agent | Anthropic Claude claude-3-5-sonnet-20241022 |
| Auth | JWT (python-jose) + bcrypt (passlib) |
| Rate limiting | slowapi |
| Logging | structlog (JSON) |

---

## Project structure

```
server/
├── main.py                  # FastAPI app, startup, CORS, rate limiter
├── database.py              # SQLAlchemy engine, session, init_db() + seeding
├── mongodb.py               # Motor async client for MongoDB Atlas
├── auth_utils.py            # JWT creation / verification, password hashing
├── models/
│   ├── user.py              # SQLAlchemy User (table: users)
│   ├── aircraft.py          # SQLAlchemy Aircraft (table: aircraft)
│   ├── flightplan.py        # SQLAlchemy FlightPlan (table: flight_plans)
│   └── decision_history.py  # SQLAlchemy DecisionHistory (table: decision_history)
├── ml/
│   ├── feature_engineer.py          # 20-feature pipeline (base)
│   └── feature_engineer_enhanced.py # 32-feature pipeline with composite risk score
├── routers/
│   ├── auth.py              # POST /auth/login, /auth/refresh, /auth/logout, /auth/me
│   ├── signup.py            # POST /auth/signup → writes to SQLite + MongoDB
│   ├── aircraft.py          # GET/POST/PUT /aircraft
│   ├── weather.py           # GET /weather/{icao}
│   ├── notam.py             # GET /notam/{icao}
│   ├── flightplan.py        # CRUD /flightplan
│   ├── gonogo.py            # POST /gonogo/evaluate (XGBoost)
│   ├── agent.py             # POST /api/agent/evaluate (Claude + XGBoost)
│   ├── decision_history.py  # GET /api/decisions, POST /api/decisions/{id}/feedback
│   └── ws.py                # WebSocket /ws (live updates)
└── requirements.txt
```

---

## Getting started locally

### Prerequisites
- Python 3.11+
- MongoDB Atlas account (free tier works)
- Anthropic API key (for Claude agent feature)

### Setup

```bash
# 1. Clone and enter server directory
git clone https://github.com/supulkalhara/STO-server.git
cd STO-server/server

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
cat > .env << 'EOF'
SECRET_KEY=your-secret-key-at-least-32-chars
MONGODB_URL=mongodb+srv://admin:password@cluster.mongodb.net/?appName=safetakeoff
DATABASE_URL=sqlite:///./safetakeoff_dev.db
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:5174
ANTHROPIC_API_KEY=sk-ant-...   # optional — Claude agent falls back to XGBoost
EOF

# 5. Run the server
uvicorn main:app --reload --port 8000
```

The database is created and seeded automatically on first startup.

### Demo credentials (auto-seeded)

| Email | Password | Role |
|-------|----------|------|
| `atc@safetakeoff.dev` | `SafeTakeOff2026!` | atc_officer |
| `supul@safetakeoff.dev` | `SafeTakeOff2026!` | atc_supervisor |
| `supervisor@safetakeoff.dev` | `SafeTakeOff2026!` | atc_supervisor |
| `officer1@safetakeoff.dev` | `SafeTakeOff2026!` | atc_officer |
| `viewer@safetakeoff.dev` | `SafeTakeOff2026!` | viewer |

---

## API reference

Interactive docs available at:
- **Swagger UI**: https://sto-server.onrender.com/docs
- **ReDoc**: https://sto-server.onrender.com/redoc

### Key endpoints

```
POST   /auth/login                          Login → returns JWT
POST   /auth/signup                         Register new account
POST   /auth/refresh                        Refresh access token

GET    /weather/{icao}                      Live METAR for aerodrome
GET    /notam/{icao}                        Active NOTAMs

GET    /aircraft                            Fleet list
POST   /aircraft                            Add aircraft

POST   /gonogo/evaluate                     XGBoost Go/No-Go decision
POST   /api/agent/evaluate                  Claude Agent + XGBoost dual evaluation

GET    /api/decisions                       Decision history (filterable)
POST   /api/decisions/{id}/feedback         ATC submits actual outcome
GET    /api/decisions/stats/{icao}          Accuracy metrics per aerodrome

GET    /health                              Readiness check
WS     /ws                                  Live data WebSocket
```

---

## ML pipeline

### Feature engineering (20 base features)

Input (8 raw) → Output (20 engineered):

| Feature | Source | Description |
|---------|--------|-------------|
| `wind_speed_kt` | METAR | Raw wind speed |
| `wind_gust_kt` | METAR | Gust speed |
| `visibility_sm` | METAR | Visibility in statute miles |
| `ceiling_ft` | METAR | Lowest cloud ceiling AGL |
| `temp_c` | METAR | Temperature |
| `crosswind_kt` | Computed | Wind component perpendicular to runway |
| `headwind_kt` | Computed | Wind component along runway |
| `tailwind_kt` | Computed | Negative headwind |
| `relative_humidity` | Computed | Magnus formula |
| `metar_category_code` | Computed | VFR=0, MVFR=1, IFR=2, LIFR=3 |
| `surface_condition_code` | Computed | DRY=0, WET=1, CONTAMINATED=2 |
| `active_notams` | NOTAM API | Count of active NOTAMs |
| `notam_severity_index` | Computed | Weighted NOTAM severity |
| `traffic_count` | ATC state | Concurrent departures queued |
| `runway_length_m` | Airport DB | Runway length |
| `aircraft_mtow_ratio` | Fleet DB | Aircraft MTOW normalised |
| `wake_count_heavy` | ATC state | Heavy/super aircraft in queue |
| `metar_age_min` | METAR | Age of METAR observation |
| `ceiling_visibility_ratio` | Computed | Combined IMC indicator |
| `temperature_trend` | Computed | Temp vs dewpoint spread |

### Decision classes

- **GO** — conditions within all limits, proceed
- **CAUTION** — marginal conditions, enhanced monitoring
- **NO-GO** — one or more hard limits exceeded

---

## Deployment (Render)

The service deploys automatically from the `main` branch via `render.yaml`.

### Required environment variables (set in Render dashboard)

| Variable | Description |
|----------|-------------|
| `SECRET_KEY` | JWT signing key (auto-generated by Render) |
| `MONGODB_URL` | MongoDB Atlas connection string |
| `ALLOWED_ORIGINS` | Comma-separated list of allowed frontend origins |
| `DATABASE_URL` | SQLite path (default: `sqlite:///./safetakeoff_prod.db`) |
| `ANTHROPIC_API_KEY` | Optional — enables Claude Agent feature |

### ALLOWED_ORIGINS example
```
https://safetakeoff.supul9229kalhara.workers.dev,http://localhost:5173,http://localhost:5174
```

---

## Running tests

```bash
cd STO-server
python run_accuracy_final.py        # ML accuracy test (13 scenarios)
python run_accuracy_tests_simple.py # Feature engineering validation
pytest tests/ -v                    # Unit tests
```

---

## Architecture notes

**Dual database design:**
- **SQLite** — auth credentials, flight plans, aircraft fleet, decision history (fast local reads)
- **MongoDB Atlas** — user profiles with org/role metadata (rich documents)

Signup writes to **both** stores. Login reads from SQLite only (fast path).

**Dual model design:**
- **XGBoost** handles the numerical decision boundary (speed, accuracy)
- **Claude Agent** provides natural-language reasoning and safety narrative (explainability)
- Both predictions are shown side-by-side; ATC feedback is collected to improve both over time
