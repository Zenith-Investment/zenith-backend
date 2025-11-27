"""Authentication endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
import structlog

from src.core.deps import DbSession

security = HTTPBearer()
from src.core.deps import CurrentUser
from src.core.security import TokenStatus, check_token_status
from src.schemas.auth import (
    ChangePasswordRequest,
    ForgotPasswordRequest,
    ForgotPasswordResponse,
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    RegisterResponse,
    RefreshTokenRequest,
    ResendVerificationRequest,
    ResendVerificationResponse,
    ResetPasswordRequest,
    ResetPasswordResponse,
    TokenResponse,
    VerifyEmailRequest,
    VerifyEmailResponse,
)
from src.services.auth import AuthService
from src.services.user import UserService
from src.services.email import email_service, get_password_reset_email_html, get_email_verification_html
from src.core.config import settings

router = APIRouter()
logger = structlog.get_logger()


class TokenStatusResponse(BaseModel):
    """Response for token status check."""
    valid: bool
    status: str
    message: str


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    db: DbSession,
) -> RegisterResponse:
    """Register a new user."""
    if not request.accepted_terms:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You must accept the terms of service",
        )

    auth_service = AuthService(db)
    user_service = UserService(db)

    try:
        # Create user
        user = await auth_service.create_user(request)
        logger.info("User registered", user_id=user.id, email=user.email)

        # Create default portfolio
        await user_service.create_default_portfolio(user)

        # Send verification email
        try:
            token = await auth_service.create_email_verification(user.id, user.email)
            verification_url = f"{settings.FRONTEND_URL}/auth/verify-email?token={token}"

            html_content = get_email_verification_html(
                name=user.full_name,
                verification_url=verification_url,
            )

            email_service.send_email(
                to_email=user.email,
                subject="Confirme seu Email - InvestAI",
                html_content=html_content,
            )
            logger.info("Verification email sent", user_id=user.id)
        except Exception as e:
            # Don't fail registration if email fails
            logger.warning("Failed to send verification email", user_id=user.id, error=str(e))

        return RegisterResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            message="User registered successfully. Please verify your email.",
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    db: DbSession,
) -> LoginResponse:
    """Authenticate user and return tokens."""
    auth_service = AuthService(db)

    user = await auth_service.authenticate_user(request)
    if not user:
        logger.warning("Failed login attempt", email=request.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    tokens = auth_service.create_tokens(user)
    logger.info("User logged in", user_id=user.id)

    return LoginResponse(**tokens)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    db: DbSession,
) -> TokenResponse:
    """Refresh access token using refresh token."""
    auth_service = AuthService(db)

    tokens = await auth_service.refresh_access_token(request.refresh_token)
    if not tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return TokenResponse(
        access_token=tokens["access_token"],
        token_type=tokens["token_type"],
        expires_in=tokens["expires_in"],
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    db: DbSession,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    refresh_token: str | None = None,
) -> None:
    """Logout user and invalidate tokens.

    The access token is taken from the Authorization header.
    Optionally, a refresh token can be provided to also invalidate it.
    """
    auth_service = AuthService(db)
    token = credentials.credentials

    await auth_service.logout_user(token, refresh_token)
    logger.info("User logged out")


@router.post("/forgot-password", response_model=ForgotPasswordResponse)
async def forgot_password(
    request: ForgotPasswordRequest,
    db: DbSession,
) -> ForgotPasswordResponse:
    """Request password reset email.

    Always returns success to prevent email enumeration attacks.
    """
    auth_service = AuthService(db)
    user_service = UserService(db)

    # Get user to send email (if exists)
    user = await auth_service.get_user_by_email(request.email)

    if user and user.is_active:
        # Create reset token
        token = await auth_service.create_password_reset(request.email)

        if token:
            # Build reset URL
            reset_url = f"{settings.FRONTEND_URL}/auth/reset-password?token={token}"

            # Send email
            html_content = get_password_reset_email_html(
                name=user.full_name,
                reset_url=reset_url,
            )

            email_service.send_email(
                to_email=user.email,
                subject="Redefinição de Senha - InvestAI",
                html_content=html_content,
            )

            logger.info("Password reset email sent", user_id=user.id)

    # Always return success to prevent enumeration
    return ForgotPasswordResponse()


@router.post("/reset-password", response_model=ResetPasswordResponse)
async def reset_password(
    request: ResetPasswordRequest,
    db: DbSession,
) -> ResetPasswordResponse:
    """Reset password using a valid reset token."""
    auth_service = AuthService(db)

    success = await auth_service.reset_password(
        token=request.token,
        new_password=request.new_password,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token invalido, expirado ou ja utilizado.",
        )

    return ResetPasswordResponse()


@router.post("/change-password", status_code=status.HTTP_200_OK)
async def change_password(
    request: ChangePasswordRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Change password for authenticated user."""
    auth_service = AuthService(db)

    success = await auth_service.change_password(
        user=current_user,
        current_password=request.current_password,
        new_password=request.new_password,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Senha atual incorreta.",
        )

    logger.info("Password changed", user_id=current_user.id)
    return {"message": "Senha alterada com sucesso."}


@router.post("/verify-email", response_model=VerifyEmailResponse)
async def verify_email(
    request: VerifyEmailRequest,
    db: DbSession,
) -> VerifyEmailResponse:
    """Verify user's email using a verification token."""
    auth_service = AuthService(db)

    success = await auth_service.verify_email(request.token)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token invalido, expirado ou ja utilizado.",
        )

    return VerifyEmailResponse()


@router.post("/resend-verification", response_model=ResendVerificationResponse)
async def resend_verification(
    request: ResendVerificationRequest,
    db: DbSession,
) -> ResendVerificationResponse:
    """Resend email verification email.

    Always returns success to prevent email enumeration attacks.
    """
    auth_service = AuthService(db)

    # Get user to send email (if exists and not verified)
    token = await auth_service.resend_verification_email(request.email)

    if token:
        user = await auth_service.get_user_by_email(request.email)
        if user:
            verification_url = f"{settings.FRONTEND_URL}/auth/verify-email?token={token}"

            html_content = get_email_verification_html(
                name=user.full_name,
                verification_url=verification_url,
            )

            email_service.send_email(
                to_email=user.email,
                subject="Confirme seu Email - InvestAI",
                html_content=html_content,
            )
            logger.info("Verification email resent", user_id=user.id)

    # Always return success to prevent enumeration
    return ResendVerificationResponse()


@router.get("/token-status", response_model=TokenStatusResponse)
async def get_token_status(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> TokenStatusResponse:
    """
    Check the status of a JWT token without full authentication.

    Useful for frontend to verify if token is still valid before making requests.

    Returns:
        - valid: true if token is valid and not revoked
        - status: one of 'valid', 'expired', 'revoked', 'invalid', 'validation_error'
        - message: human-readable status message
    """
    token = credentials.credentials
    token_status, _ = await check_token_status(token)

    status_messages = {
        TokenStatus.VALID: "Token is valid",
        TokenStatus.EXPIRED: "Token has expired. Please login again.",
        TokenStatus.REVOKED: "Token has been revoked. Please login again.",
        TokenStatus.INVALID: "Token is invalid or malformed.",
        TokenStatus.VALIDATION_ERROR: "Unable to validate token. Please try again.",
    }

    return TokenStatusResponse(
        valid=token_status == TokenStatus.VALID,
        status=token_status.value,
        message=status_messages.get(token_status, "Unknown token status"),
    )
