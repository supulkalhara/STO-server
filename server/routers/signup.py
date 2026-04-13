"""
POST /auth/signup  — registers a new user in MongoDB Atlas.

Fields stored:
  full_name, email (unique, indexed), hashed_password,
  organisation, role (default "atc"), is_active (True),
  created_at (UTC)
"""
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr, field_validator
from passlib.context import CryptContext

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
async def signup(body: SignUpRequest):
    db = get_mongo_db()
    users = db["users"]

    # Ensure unique email index (idempotent)
    await users.create_index("email", unique=True)

    # Reject duplicate emails
    existing = await users.find_one({"email": body.email}, {"_id": 1})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    hashed = pwd_ctx.hash(body.password)
    doc = {
        "full_name": body.full_name,
        "email": body.email,
        "hashed_password": hashed,
        "organisation": body.organisation or "",
        "role": "atc",
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
    }
    await users.insert_one(doc)

    return SignUpResponse(
        message="Account created successfully.",
        email=body.email,
    )
