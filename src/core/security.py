"""Security utilities for authentication and data protection."""
import base64
import hashlib
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import redis.asyncio as redis
import structlog
from cryptography.fernet import Fernet
from jose import ExpiredSignatureError, JWTError, jwt
from passlib.context import CryptContext

from src.core.config import settings

logger = structlog.get_logger()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenStatus(str, Enum):
    """Status of a JWT token."""
    VALID = "valid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    INVALID = "invalid"
    VALIDATION_ERROR = "validation_error"

# Token blacklist Redis key prefix
TOKEN_BLACKLIST_PREFIX = "token_blacklist:"
TOKEN_BLACKLIST_TTL = 60 * 60 * 24 * 7  # 7 days

# Redis connection for token blacklist
_redis_client: redis.Redis | None = None


async def get_redis() -> redis.Redis:
    """Get Redis connection for token operations."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


# ===========================================
# Password Hashing
# ===========================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


# ===========================================
# JWT Token Management
# ===========================================

def create_access_token(
    subject: str | int,
    expires_delta: timedelta | None = None,
    additional_claims: dict[str, Any] | None = None,
) -> str:
    """Create a JWT access token."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode = {
        "exp": expire,
        "sub": str(subject),
        "type": "access",
        "jti": _generate_jti(),  # Unique token identifier for blacklisting
    }
    if additional_claims:
        to_encode.update(additional_claims)

    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(
    subject: str | int,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a JWT refresh token."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )

    to_encode = {
        "exp": expire,
        "sub": str(subject),
        "type": "refresh",
        "jti": _generate_jti(),
    }

    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> dict[str, Any] | None:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        return payload
    except JWTError:
        return None


def decode_token_detailed(token: str) -> tuple[dict[str, Any] | None, TokenStatus]:
    """
    Decode a JWT token with detailed status information.

    Returns:
        Tuple of (payload or None, TokenStatus)
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        return payload, TokenStatus.VALID
    except ExpiredSignatureError:
        return None, TokenStatus.EXPIRED
    except JWTError:
        return None, TokenStatus.INVALID


def _generate_jti() -> str:
    """Generate a unique token identifier."""
    import uuid
    return str(uuid.uuid4())


# ===========================================
# Token Blacklist (Redis-based)
# ===========================================

async def blacklist_token(token: str) -> bool:
    """
    Add a token to the blacklist.
    Used when user logs out to invalidate their tokens.
    """
    try:
        payload = decode_token(token)
        if not payload:
            return False

        jti = payload.get("jti")
        if not jti:
            # Fallback: hash the token itself
            jti = hashlib.sha256(token.encode()).hexdigest()[:32]

        # Calculate TTL based on token expiration
        exp = payload.get("exp")
        if exp:
            exp_datetime = datetime.fromtimestamp(exp, tz=timezone.utc)
            ttl = int((exp_datetime - datetime.now(timezone.utc)).total_seconds())
            ttl = max(ttl, 60)  # At least 60 seconds
        else:
            ttl = TOKEN_BLACKLIST_TTL

        redis_client = await get_redis()
        key = f"{TOKEN_BLACKLIST_PREFIX}{jti}"
        await redis_client.setex(key, ttl, "1")

        logger.info("Token blacklisted", jti=jti[:8])
        return True

    except Exception as e:
        logger.error("Failed to blacklist token", error=str(e))
        return False


async def is_token_blacklisted(token: str, jti: str | None = None) -> bool:
    """
    Check if a token is blacklisted.

    Args:
        token: The JWT token string
        jti: Optional pre-extracted JTI to avoid re-decoding

    Returns:
        True if blacklisted, False otherwise

    Raises:
        Exception: If Redis is unavailable (fail-safe behavior)
    """
    try:
        if not jti:
            payload = decode_token(token)
            if not payload:
                return True  # Invalid token is effectively blacklisted

            jti = payload.get("jti")
            if not jti:
                jti = hashlib.sha256(token.encode()).hexdigest()[:32]

        redis_client = await get_redis()
        key = f"{TOKEN_BLACKLIST_PREFIX}{jti}"
        result = await redis_client.get(key)

        return result is not None

    except Exception as e:
        logger.error("Redis unavailable for token blacklist check", error=str(e))
        raise  # Fail-safe: deny access if Redis is unavailable


async def check_token_status(token: str) -> tuple[TokenStatus, dict[str, Any] | None]:
    """
    Comprehensive token validation with detailed status.

    Checks:
    1. Token format and signature validity
    2. Token expiration
    3. Token revocation (blacklist)

    Returns:
        Tuple of (TokenStatus, payload or None)
    """
    # Step 1: Decode and check basic validity
    payload, status = decode_token_detailed(token)

    if status != TokenStatus.VALID:
        return status, None

    # Step 2: Check if token is blacklisted (revoked)
    try:
        jti = payload.get("jti")
        if await is_token_blacklisted(token, jti):
            return TokenStatus.REVOKED, payload
    except Exception:
        # Redis unavailable - fail safe
        return TokenStatus.VALIDATION_ERROR, None

    return TokenStatus.VALID, payload


# ===========================================
# CPF Encryption (AES-256 via Fernet)
# ===========================================

def _get_encryption_key() -> bytes:
    """
    Derive a Fernet key from the SECRET_KEY.
    Fernet requires a 32-byte base64-encoded key.
    """
    # Use SHA256 to get a consistent 32-byte key from SECRET_KEY
    key_hash = hashlib.sha256(settings.SECRET_KEY.encode()).digest()
    return base64.urlsafe_b64encode(key_hash)


def encrypt_cpf(cpf: str) -> str:
    """
    Encrypt a CPF number using Fernet (AES-128-CBC).

    Args:
        cpf: Plain text CPF (can include formatting)

    Returns:
        Encrypted CPF as base64 string
    """
    # Remove formatting from CPF
    cpf_clean = "".join(filter(str.isdigit, cpf))

    if len(cpf_clean) != 11:
        raise ValueError("CPF must have 11 digits")

    fernet = Fernet(_get_encryption_key())
    encrypted = fernet.encrypt(cpf_clean.encode())

    return encrypted.decode()


def decrypt_cpf(encrypted_cpf: str) -> str:
    """
    Decrypt an encrypted CPF.

    Args:
        encrypted_cpf: Base64 encoded encrypted CPF

    Returns:
        Plain text CPF (11 digits, no formatting)
    """
    try:
        fernet = Fernet(_get_encryption_key())
        decrypted = fernet.decrypt(encrypted_cpf.encode())
        return decrypted.decode()
    except Exception as e:
        logger.error("Failed to decrypt CPF", error=str(e))
        raise ValueError("Invalid encrypted CPF")


def mask_cpf(cpf: str) -> str:
    """
    Mask a CPF for display purposes.

    Args:
        cpf: Plain text CPF (11 digits)

    Returns:
        Masked CPF like "***.***.***-12"
    """
    cpf_clean = "".join(filter(str.isdigit, cpf))
    if len(cpf_clean) != 11:
        return "***.***.***-**"

    return f"***.***.*{cpf_clean[8]}{cpf_clean[9]}-{cpf_clean[9:11]}"


def validate_cpf(cpf: str) -> bool:
    """
    Validate a Brazilian CPF number.

    Args:
        cpf: CPF string (can include formatting)

    Returns:
        True if valid, False otherwise
    """
    cpf_clean = "".join(filter(str.isdigit, cpf))

    if len(cpf_clean) != 11:
        return False

    # Check for known invalid patterns
    if cpf_clean == cpf_clean[0] * 11:
        return False

    # Calculate first check digit
    sum1 = sum(int(cpf_clean[i]) * (10 - i) for i in range(9))
    digit1 = (sum1 * 10 % 11) % 10

    if digit1 != int(cpf_clean[9]):
        return False

    # Calculate second check digit
    sum2 = sum(int(cpf_clean[i]) * (11 - i) for i in range(10))
    digit2 = (sum2 * 10 % 11) % 10

    return digit2 == int(cpf_clean[10])


# ===========================================
# Password Reset Tokens
# ===========================================

PASSWORD_RESET_TOKEN_EXPIRE_HOURS = 24
PASSWORD_RESET_PREFIX = "password_reset:"


def create_password_reset_token(user_id: int, email: str) -> str:
    """
    Create a JWT token for password reset.

    Args:
        user_id: User ID
        email: User email

    Returns:
        JWT token string
    """
    expire = datetime.now(timezone.utc) + timedelta(hours=PASSWORD_RESET_TOKEN_EXPIRE_HOURS)

    to_encode = {
        "exp": expire,
        "sub": str(user_id),
        "email": email,
        "type": "password_reset",
        "jti": _generate_jti(),
    }

    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_password_reset_token(token: str) -> dict | None:
    """
    Decode and validate a password reset token.

    Args:
        token: JWT token string

    Returns:
        Token payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )

        if payload.get("type") != "password_reset":
            return None

        return payload
    except JWTError:
        return None


async def mark_reset_token_used(token: str) -> bool:
    """Mark a password reset token as used (single-use tokens)."""
    try:
        payload = decode_password_reset_token(token)
        if not payload:
            return False

        jti = payload.get("jti")
        if not jti:
            return False

        redis_client = await get_redis()
        key = f"{PASSWORD_RESET_PREFIX}{jti}"

        # Check if already used
        existing = await redis_client.get(key)
        if existing:
            return False  # Token already used

        # Mark as used
        ttl = PASSWORD_RESET_TOKEN_EXPIRE_HOURS * 3600
        await redis_client.setex(key, ttl, "used")

        return True

    except Exception as e:
        logger.error("Failed to mark reset token as used", error=str(e))
        return False


async def is_reset_token_used(token: str) -> bool:
    """Check if a password reset token has already been used."""
    try:
        payload = decode_password_reset_token(token)
        if not payload:
            return True  # Invalid token

        jti = payload.get("jti")
        if not jti:
            return True

        redis_client = await get_redis()
        key = f"{PASSWORD_RESET_PREFIX}{jti}"
        result = await redis_client.get(key)

        return result is not None

    except Exception:
        return False


# ===========================================
# Email Verification Tokens
# ===========================================

EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS = 48
EMAIL_VERIFICATION_PREFIX = "email_verification:"


def create_email_verification_token(user_id: int, email: str) -> str:
    """
    Create a JWT token for email verification.

    Args:
        user_id: User ID
        email: User email

    Returns:
        JWT token string
    """
    expire = datetime.now(timezone.utc) + timedelta(hours=EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS)

    to_encode = {
        "exp": expire,
        "sub": str(user_id),
        "email": email,
        "type": "email_verification",
        "jti": _generate_jti(),
    }

    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_email_verification_token(token: str) -> dict | None:
    """
    Decode and validate an email verification token.

    Args:
        token: JWT token string

    Returns:
        Token payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )

        if payload.get("type") != "email_verification":
            return None

        return payload
    except JWTError:
        return None


async def mark_verification_token_used(token: str) -> bool:
    """Mark an email verification token as used (single-use tokens)."""
    try:
        payload = decode_email_verification_token(token)
        if not payload:
            return False

        jti = payload.get("jti")
        if not jti:
            return False

        redis_client = await get_redis()
        key = f"{EMAIL_VERIFICATION_PREFIX}{jti}"

        # Check if already used
        existing = await redis_client.get(key)
        if existing:
            return False  # Token already used

        # Mark as used
        ttl = EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS * 3600
        await redis_client.setex(key, ttl, "used")

        return True

    except Exception as e:
        logger.error("Failed to mark verification token as used", error=str(e))
        return False


async def is_verification_token_used(token: str) -> bool:
    """Check if an email verification token has already been used."""
    try:
        payload = decode_email_verification_token(token)
        if not payload:
            return True  # Invalid token

        jti = payload.get("jti")
        if not jti:
            return True

        redis_client = await get_redis()
        key = f"{EMAIL_VERIFICATION_PREFIX}{jti}"
        result = await redis_client.get(key)

        return result is not None

    except Exception:
        return False
