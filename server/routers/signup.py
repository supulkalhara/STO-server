"""
POST /auth/signup  — registers a new user.

Writes to two stores so the whole system stays consistent:
  1. MongoDB Atlas  — rich profile doc (org, timestamps, role)
  2. SQLite via SQLAlchemy — auth credentials read by /auth/login
"""
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, field_validator
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from database import get_db
from models.user import User
from mongodb import get_mongo_db

router = APIRouter(prefix="/auth", tags=["auth"])
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SignUpRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    organisation: Optional[str] = None

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

    @field_validator("full_name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Full name must not be empty")
        return v.strip()


class SignUpResponse(BaseModel):
    message: str
    email: str


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post(
    "/signup",
    response_model=SignUpResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
async def signup(body: SignUpRequest, db: Session = Depends(get_db)):
    hashed = pwd_ctx.hash(body.password)

    # ── 1. Check SQLite first (source of truth for login) ────────────────────
    existing_sql = db.query(User).filter(User.email == body.email).first()
    if existing_sql:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    # ── 2. Write to SQLite — this is what /auth/login reads ─────────────────
    new_user = User(
        email=body.email,
        hashed_password=hashed,
        full_name=body.full_name,
        role="atc_officer",
        is_active=True,
    )
    db.add(new_user)
    db.commit()

    # ── 3. Mirror to MongoDB Atlas — rich profile doc ────────────────────────
    try:
        mongo_db = get_mongo_db()
        users_col = mongo_db["users"]
        await users_col.create_index("email", unique=True)
        doc = {
            "full_name": body.full_name,
            "email": body.email,
            "hashed_password": hashed,
            "organisation": body.organisation or "",
            "role": "atc_officer",
            "is_active": True,
            "created_at": datetime.now(timezone.utc),
        }
        await users_col.insert_one(doc)
    except Exception:
        # MongoDB is optional — login still works via SQLite
        pass

    return SignUpResponse(
        message="Account created successfully.",
        email=body.email,
    )
