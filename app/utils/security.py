"""Security utilities for authentication."""
from fastapi import Header, HTTPException, status
from app.config import settings
import logging

logger = logging.getLogger(__name__)


def require_password(password: str = Header(None, alias="password")) -> str:
    """
    Simple password check for operations that cost money.

    Args:
        password: Password from header

    Returns:
        The validated password

    Raises:
        HTTPException: If password is missing or incorrect
    """
    if not settings.API_KEY:
        logger.error("API_KEY not configured in environment!")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication not properly configured"
        )

    if not password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing password. Include 'password' header with API key."
        )

    if password != settings.API_KEY:
        logger.warning(f"Failed authentication attempt from header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password"
        )

    return password
