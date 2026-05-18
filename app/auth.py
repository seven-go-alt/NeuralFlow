"""JWT authentication module — minimal, portfolio-ready.

Usage:
    from app.auth import require_auth, create_token

    @router.post("/chat")
    async def chat(user: Annotated[dict, Depends(require_auth)], request: ChatRequest):
        ...
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated

import jwt
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.config import get_settings

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
ALGORITHM = "HS256"


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    tenant_id: str


@router.post("/token", response_model=TokenResponse)
async def login(request: TokenRequest):
    settings = get_settings()
    if (
        request.username != settings.auth_admin_username
        or request.password != settings.auth_admin_password
    ):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    now = datetime.utcnow()
    payload = {
        "sub": request.username,
        "tenant_id": settings.tenant_default_id,
        "exp": now + timedelta(hours=24),
        "iat": now,
    }
    token = jwt.encode(payload, settings.auth_jwt_secret, algorithm=ALGORITHM)
    return TokenResponse(access_token=token, tenant_id=settings.tenant_default_id)


async def require_auth(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict:
    """FastAPI dependency: validates Bearer JWT and returns user payload.

    When auth_enabled=False (default), skips validation and returns anonymous context.
    """
    settings = get_settings()
    if not settings.auth_enabled:
        return {"sub": "anonymous", "tenant_id": settings.tenant_default_id}

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid Authorization scheme")

    try:
        return jwt.decode(token, settings.auth_jwt_secret, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired") from None
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token") from None
