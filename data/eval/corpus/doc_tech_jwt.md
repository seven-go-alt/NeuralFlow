# Meridian Analytics — JWT Signature Validation in FastAPI Middleware

**Document ID:** doc_tech_jwt
**Owner:** Platform Engineering
**Last updated:** 2026-07-13

## Overview

Meridian Analytics authenticates internal service-to-service communication using **JWT** bearer tokens issued by an in-house token service. This document describes the **JWT** **signature validation** method implemented in **Python** **FastAPI** **middleware** using the **PyJWT** library, without any third-party authentication providers.

## Middleware Structure

Meridian's authentication **middleware** intercepts all API requests and validates the **JWT** token before allowing the request to reach the route handler:

```python
import jwt
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class JWTAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, public_key: str, allowed_issuer: str, allowed_audience: str):
        super().__init__(app)
        self.public_key = public_key
        self.allowed_issuer = allowed_issuer
        self.allowed_audience = allowed_audience

    async def dispatch(self, request: Request, call_next):
        # Skip auth for health check and public endpoints
        if request.url.path in ("/health", "/openapi.json", "/docs"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "Missing or malformed Authorization header"}
            )

        token = auth_header.split(" ", 1)[1]
        payload = self._validate_jwt(token)
        if payload is None:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or expired JWT"}
            )

        request.state.user = payload
        return await call_next(request)
```

The **middleware** validates the **JWT** on every request, decoding the **signature validation** result into `request.state.user` for downstream handlers.

## Signature Validation with PyJWT

The core **signature validation** logic uses **PyJWT**'s `decode` method with an RSA public key:

```python
def _validate_jwt(self, token: str) -> dict | None:
    try:
        payload = jwt.decode(
            token,
            self.public_key,
            algorithms=["RS256"],
            issuer=self.allowed_issuer,
            audience=self.allowed_audience,
            options={
                "verify_exp": True,
                "verify_iat": True,
                "require": ["exp", "iat", "iss", "aud", "sub"]
            }
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning("JWT validation failed: %s", str(e))
        return None
```

The **PyJWT** library automatically verifies the **JWT** **signature** using the provided RSA public key and the RS256 algorithm. Meridian's in-house token service signs tokens with the corresponding RSA private key. The `issuer` and `audience` claims are verified as additional security measures.

## Key Management

**JWT** **signature validation** requires access to the signing public key. Meridian stores the public key as a **Python** string from an environment variable:

```python
# In application startup
public_key = os.environ["MERIDIAN_JWT_PUBLIC_KEY"]
app.add_middleware(JWTAuthMiddleware, public_key=public_key,
                   allowed_issuer="meridian-token-service",
                   allowed_audience="meridian-api")
```

The public key is rotated quarterly by deploying updated environment variables. The **PyJWT** library supports key rotation through multiple allowed public keys, though Meridian currently uses a single key per environment.

## Testing Validation

Meridian unit-tests the **FastAPI** **middleware** **JWT** **signature validation** by generating test tokens using **PyJWT**'s `encode` method:

```python
def test_valid_jwt_passes_through():
    private_key = generate_test_rsa_key()
    public_key = private_key.public_key()

    token = jwt.encode(
        {"sub": "service:ingestion", "iss": "meridian-token-service",
         "aud": "meridian-api", "exp": int(time.time()) + 300,
         "iat": int(time.time())},
        private_key,
        algorithm="RS256"
    )

    response = client.get("/api/v1/transactions",
                          headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
```

The **FastAPI** test client passes the token through the **middleware**, exercising the full **signature validation** pipeline.

## Revision History

This document was last updated on 13 July 2026 following the addition of audience claim verification to the FastAPI middleware.
