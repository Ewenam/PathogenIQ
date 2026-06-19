"""
saas/auth.py
JWT verification + org resolution for PATHOGENIQ_MODE=saas.

Verifies a Bearer JWT (HS256, shared secret) on each API request and resolves
it to an organization. This works identically whether the token was issued by
a real Supabase Auth project (HS256-signed with the project's JWT secret) or
a self-issued token signed with the same PATHOGENIQ_JWT_SECRET for local/dev
testing — only the env var value changes between the two.
"""
from __future__ import annotations

import os

import jwt
from fastapi import HTTPException, Request
from sqlalchemy.engine import Engine

from .db import get_engine, get_org_for_user

_engine: Engine | None = None


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


def _jwt_secret() -> str:
    secret = os.environ.get("PATHOGENIQ_JWT_SECRET")
    if not secret:
        raise RuntimeError("PATHOGENIQ_JWT_SECRET is not set (required for PATHOGENIQ_MODE=saas).")
    return secret


def verify_jwt(token: str) -> dict:
    """Decode and validate a Bearer token. Raises jwt.PyJWTError on failure."""
    return jwt.decode(
        token,
        _jwt_secret(),
        algorithms=["HS256"],
        options={"verify_aud": False},  # Supabase sets aud="authenticated"; not load-bearing here
    )


def get_current_user_id(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing or malformed Authorization header")
    try:
        claims = verify_jwt(auth[len("Bearer "):])
    except jwt.PyJWTError as exc:
        raise HTTPException(401, f"Invalid token: {exc}")
    user_id = claims.get("sub")
    if not user_id:
        raise HTTPException(401, "Token missing 'sub' claim")
    return user_id


def get_current_org(request: Request) -> str:
    """FastAPI dependency: returns the org_id for the authenticated request."""
    user_id = get_current_user_id(request)
    org_id = get_org_for_user(_get_engine(), user_id)
    if not org_id:
        raise HTTPException(403, "User is not a member of any organization")
    return org_id
