"""Authentication service."""
from datetime import datetime, timezone

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import (
    blacklist_token,
    create_access_token,
    create_email_verification_token,
    create_password_reset_token,
    create_refresh_token,
    decode_email_verification_token,
    decode_password_reset_token,
    decode_token,
    encrypt_cpf,
    get_password_hash,
    is_reset_token_used,
    is_verification_token_used,
    mark_reset_token_used,
    mark_verification_token_used,
    validate_cpf,
    verify_password,
)
from src.models.user import User
from src.schemas.auth import LoginRequest, RegisterRequest

logger = structlog.get_logger()


class AuthService:
    """Service for authentication operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_by_email(self, email: str) -> User | None:
        """Get user by email."""
        result = await self.db.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def get_user_by_id(self, user_id: int) -> User | None:
        """Get user by ID."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def create_user(self, request: RegisterRequest) -> User:
        """Create a new user."""
        # Check if user already exists
        existing_user = await self.get_user_by_email(request.email)
        if existing_user:
            raise ValueError("Email already registered")

        # Encrypt and validate CPF if provided
        encrypted_cpf = None
        if request.cpf:
            if not validate_cpf(request.cpf):
                raise ValueError("Invalid CPF")
            encrypted_cpf = encrypt_cpf(request.cpf)

        # Create user
        user = User(
            email=request.email,
            hashed_password=get_password_hash(request.password),
            full_name=request.full_name,
            phone=request.phone,
            cpf_encrypted=encrypted_cpf,
        )

        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def authenticate_user(self, request: LoginRequest) -> User | None:
        """Authenticate user with email and password."""
        user = await self.get_user_by_email(request.email)
        if not user:
            return None
        if not verify_password(request.password, user.hashed_password):
            return None
        if not user.is_active:
            return None

        # Update last login
        user.last_login_at = datetime.now(timezone.utc)
        await self.db.commit()

        return user

    def create_tokens(self, user: User) -> dict:
        """Create access and refresh tokens for user."""
        access_token = create_access_token(
            subject=user.id,
            additional_claims={
                "email": user.email,
                "plan": user.subscription_plan.value,
            }
        )
        refresh_token = create_refresh_token(subject=user.id)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": 30 * 60,  # 30 minutes in seconds
        }

    async def refresh_access_token(self, refresh_token: str) -> dict | None:
        """Refresh access token using refresh token."""
        payload = decode_token(refresh_token)
        if not payload:
            return None

        if payload.get("type") != "refresh":
            return None

        user_id = payload.get("sub")
        if not user_id:
            return None

        user = await self.get_user_by_id(int(user_id))
        if not user or not user.is_active:
            return None

        return self.create_tokens(user)

    async def validate_access_token(self, token: str) -> User | None:
        """Validate access token and return user."""
        payload = decode_token(token)
        if not payload:
            return None

        if payload.get("type") != "access":
            return None

        user_id = payload.get("sub")
        if not user_id:
            return None

        user = await self.get_user_by_id(int(user_id))
        if not user or not user.is_active:
            return None

        return user

    async def logout_user(self, access_token: str, refresh_token: str | None = None) -> bool:
        """
        Logout user by blacklisting their tokens.

        Args:
            access_token: The current access token to invalidate
            refresh_token: Optional refresh token to also invalidate

        Returns:
            True if at least one token was blacklisted successfully
        """
        success = await blacklist_token(access_token)

        if refresh_token:
            await blacklist_token(refresh_token)

        return success

    async def create_password_reset(self, email: str) -> str | None:
        """
        Create a password reset token for a user.

        Args:
            email: User's email address

        Returns:
            Reset token if user exists, None otherwise
        """
        user = await self.get_user_by_email(email)
        if not user:
            return None

        if not user.is_active:
            return None

        token = create_password_reset_token(user.id, user.email)
        logger.info("Password reset token created", user_id=user.id)

        return token

    async def reset_password(self, token: str, new_password: str) -> bool:
        """
        Reset user's password using a reset token.

        Args:
            token: Password reset token
            new_password: New password to set

        Returns:
            True if password was reset successfully
        """
        # Decode and validate token
        payload = decode_password_reset_token(token)
        if not payload:
            logger.warning("Invalid password reset token")
            return False

        # Check if token was already used
        if await is_reset_token_used(token):
            logger.warning("Password reset token already used")
            return False

        # Get user
        user_id = int(payload.get("sub", 0))
        email = payload.get("email")

        user = await self.get_user_by_id(user_id)
        if not user or user.email != email:
            logger.warning("User not found for password reset", user_id=user_id)
            return False

        if not user.is_active:
            logger.warning("Inactive user attempted password reset", user_id=user_id)
            return False

        # Mark token as used
        if not await mark_reset_token_used(token):
            logger.error("Failed to mark reset token as used")
            return False

        # Update password
        user.hashed_password = get_password_hash(new_password)
        user.updated_at = datetime.now(timezone.utc)

        await self.db.commit()

        logger.info("Password reset successful", user_id=user.id)
        return True

    async def change_password(
        self, user: User, current_password: str, new_password: str
    ) -> bool:
        """
        Change user's password (requires current password).

        Args:
            user: User object
            current_password: Current password for verification
            new_password: New password to set

        Returns:
            True if password was changed successfully
        """
        if not verify_password(current_password, user.hashed_password):
            return False

        user.hashed_password = get_password_hash(new_password)
        user.updated_at = datetime.now(timezone.utc)

        await self.db.commit()

        logger.info("Password changed", user_id=user.id)
        return True

    async def create_email_verification(self, user_id: int, email: str) -> str:
        """
        Create an email verification token.

        Args:
            user_id: User ID
            email: User email

        Returns:
            Verification token
        """
        token = create_email_verification_token(user_id, email)
        logger.info("Email verification token created", user_id=user_id)
        return token

    async def verify_email(self, token: str) -> bool:
        """
        Verify user's email using a verification token.

        Args:
            token: Email verification token

        Returns:
            True if email was verified successfully
        """
        # Decode and validate token
        payload = decode_email_verification_token(token)
        if not payload:
            logger.warning("Invalid email verification token")
            return False

        # Check if token was already used
        if await is_verification_token_used(token):
            logger.warning("Email verification token already used")
            return False

        # Get user
        user_id = int(payload.get("sub", 0))
        email = payload.get("email")

        user = await self.get_user_by_id(user_id)
        if not user or user.email != email:
            logger.warning("User not found for email verification", user_id=user_id)
            return False

        if user.is_verified:
            logger.info("Email already verified", user_id=user_id)
            return True

        # Mark token as used
        if not await mark_verification_token_used(token):
            logger.error("Failed to mark verification token as used")
            return False

        # Verify user
        user.is_verified = True
        user.updated_at = datetime.now(timezone.utc)

        await self.db.commit()

        logger.info("Email verified successfully", user_id=user.id)
        return True

    async def resend_verification_email(self, email: str) -> str | None:
        """
        Create a new verification token for resending verification email.

        Args:
            email: User's email address

        Returns:
            New verification token if user exists and is not verified, None otherwise
        """
        user = await self.get_user_by_email(email)
        if not user:
            return None

        if not user.is_active:
            return None

        if user.is_verified:
            return None  # Already verified

        token = await self.create_email_verification(user.id, user.email)
        return token
