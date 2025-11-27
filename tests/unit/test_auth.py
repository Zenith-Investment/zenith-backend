"""Tests for authentication endpoints."""
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.user import User
from src.core.security import (
    create_email_verification_token,
    create_password_reset_token,
    decode_email_verification_token,
    decode_password_reset_token,
    get_password_hash,
    verify_password,
)


class TestAuthEndpoints:
    """Test authentication API endpoints."""

    @pytest.mark.asyncio
    async def test_register_user_success(self, client: AsyncClient):
        """Test successful user registration."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "securepassword123",
                "full_name": "New User",
                "accepted_terms": True,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["full_name"] == "New User"
        assert "id" in data
        assert "hashed_password" not in data

    @pytest.mark.asyncio
    async def test_register_user_without_terms(self, client: AsyncClient):
        """Test registration without accepting terms fails."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "noterms@example.com",
                "password": "password123",
                "full_name": "No Terms User",
                "accepted_terms": False,
            },
        )
        assert response.status_code == 400
        assert "terms" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_register_user_duplicate_email(
        self, client: AsyncClient, test_user: User
    ):
        """Test registration with existing email fails."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": test_user.email,
                "password": "anotherpassword",
                "full_name": "Another User",
                "accepted_terms": True,
            },
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_register_user_invalid_email(self, client: AsyncClient):
        """Test registration with invalid email fails."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "invalid-email",
                "password": "password123",
                "full_name": "Test User",
                "accepted_terms": True,
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_login_success(self, client: AsyncClient, test_user: User):
        """Test successful login."""
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user.email,
                "password": "testpassword123",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, client: AsyncClient, test_user: User):
        """Test login with wrong password fails."""
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user.email,
                "password": "wrongpassword",
            },
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_login_nonexistent_user(self, client: AsyncClient):
        """Test login with nonexistent user fails."""
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "somepassword",
            },
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_token_success(self, client: AsyncClient, test_user: User):
        """Test refreshing access token."""
        # First login to get tokens
        login_response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user.email,
                "password": "testpassword123",
            },
        )
        tokens = login_response.json()

        # Refresh token
        response = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": tokens["refresh_token"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data


class TestPasswordManagement:
    """Test password reset and change functionality."""

    @pytest.mark.asyncio
    async def test_forgot_password_success(self, client: AsyncClient, test_user: User):
        """Test forgot password endpoint."""
        response = await client.post(
            "/api/v1/auth/forgot-password",
            json={"email": test_user.email},
        )
        # Should always return success to prevent enumeration
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_forgot_password_nonexistent_email(self, client: AsyncClient):
        """Test forgot password with nonexistent email still returns success."""
        response = await client.post(
            "/api/v1/auth/forgot-password",
            json={"email": "nonexistent@example.com"},
        )
        # Should always return success to prevent enumeration
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_change_password_success(
        self, client: AsyncClient, test_user: User, auth_headers: dict
    ):
        """Test changing password for authenticated user."""
        response = await client.post(
            "/api/v1/auth/change-password",
            headers=auth_headers,
            json={
                "current_password": "testpassword123",
                "new_password": "newpassword456",
            },
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_change_password_wrong_current(
        self, client: AsyncClient, test_user: User, auth_headers: dict
    ):
        """Test changing password with wrong current password fails."""
        response = await client.post(
            "/api/v1/auth/change-password",
            headers=auth_headers,
            json={
                "current_password": "wrongpassword",
                "new_password": "newpassword456",
            },
        )
        assert response.status_code == 400


class TestEmailVerification:
    """Test email verification functionality."""

    @pytest.mark.asyncio
    async def test_verify_email_success(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        """Test email verification with valid token."""
        # Create an unverified user
        user = User(
            email="unverified@example.com",
            hashed_password=get_password_hash("password123"),
            full_name="Unverified User",
            is_active=True,
            is_verified=False,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        # Create verification token
        token = create_email_verification_token(user.id, user.email)

        # Verify email
        response = await client.post(
            "/api/v1/auth/verify-email",
            json={"token": token},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_verify_email_invalid_token(self, client: AsyncClient):
        """Test email verification with invalid token fails."""
        response = await client.post(
            "/api/v1/auth/verify-email",
            json={"token": "invalid-token"},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_resend_verification_success(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        """Test resending verification email."""
        # Create an unverified user
        user = User(
            email="resend@example.com",
            hashed_password=get_password_hash("password123"),
            full_name="Resend User",
            is_active=True,
            is_verified=False,
        )
        db_session.add(user)
        await db_session.commit()

        response = await client.post(
            "/api/v1/auth/resend-verification",
            json={"email": user.email},
        )
        # Should always return success to prevent enumeration
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_resend_verification_already_verified(
        self, client: AsyncClient, test_user: User
    ):
        """Test resending verification to already verified user still returns success."""
        response = await client.post(
            "/api/v1/auth/resend-verification",
            json={"email": test_user.email},
        )
        # Should always return success to prevent enumeration
        assert response.status_code == 200


class TestSecurityFunctions:
    """Test security utility functions."""

    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "mysecretpassword"
        hashed = get_password_hash(password)

        assert hashed != password
        assert verify_password(password, hashed)
        assert not verify_password("wrongpassword", hashed)

    def test_email_verification_token(self):
        """Test email verification token creation and decoding."""
        user_id = 123
        email = "test@example.com"

        token = create_email_verification_token(user_id, email)
        payload = decode_email_verification_token(token)

        assert payload is not None
        assert payload["sub"] == str(user_id)
        assert payload["email"] == email
        assert payload["type"] == "email_verification"

    def test_password_reset_token(self):
        """Test password reset token creation and decoding."""
        user_id = 456
        email = "reset@example.com"

        token = create_password_reset_token(user_id, email)
        payload = decode_password_reset_token(token)

        assert payload is not None
        assert payload["sub"] == str(user_id)
        assert payload["email"] == email
        assert payload["type"] == "password_reset"

    def test_invalid_token_decode(self):
        """Test that invalid tokens return None."""
        assert decode_email_verification_token("invalid-token") is None
        assert decode_password_reset_token("invalid-token") is None
