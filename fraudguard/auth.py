"""
fraudguard/auth.py — API Key Authentication

Provides optional API key authentication via X-API-Key header.
If API_KEY is not set in environment, authentication is disabled.
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from fraudguard.config import get_settings
from fraudguard.logging_config import get_logger

logger = get_logger("fraudguard.auth")

# API key header scheme (auto_error=False so we can handle missing key ourselves)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(api_key_header),
) -> str | None:
    """Verify the API key from the X-API-Key header.

    If API_KEY is not configured (empty string), authentication is
    disabled and all requests are allowed through.

    Args:
        api_key: The API key from the request header.

    Returns:
        The validated API key, or None if auth is disabled.

    Raises:
        HTTPException: 401 if the key is missing or invalid.
    """
    settings = get_settings()

    # Auth disabled — allow all requests
    if not settings.auth_enabled:
        return None

    # Auth enabled but no key provided
    if not api_key:
        logger.warning("Authentication failed: missing API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide it via the X-API-Key header.",
        )

    # Auth enabled and key doesn't match
    if api_key != settings.api_key:
        logger.warning("Authentication failed: invalid API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    return api_key
