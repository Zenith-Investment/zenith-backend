"""User management endpoints."""
from fastapi import APIRouter, status
import structlog

from src.core.deps import CurrentUser, DbSession
from src.schemas.user import UserResponse, UserUpdate
from src.services.user import UserService

router = APIRouter()
logger = structlog.get_logger()


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    current_user: CurrentUser,
) -> UserResponse:
    """Get current authenticated user."""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        phone=current_user.phone,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        subscription_plan=current_user.subscription_plan,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at,
    )


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: CurrentUser,
    db: DbSession,
) -> UserResponse:
    """Update current authenticated user."""
    user_service = UserService(db)

    updated_user = await user_service.update_user(current_user, user_update)
    logger.info("User updated", user_id=updated_user.id)

    return UserResponse(
        id=updated_user.id,
        email=updated_user.email,
        full_name=updated_user.full_name,
        phone=updated_user.phone,
        is_active=updated_user.is_active,
        is_verified=updated_user.is_verified,
        subscription_plan=updated_user.subscription_plan,
        created_at=updated_user.created_at,
        updated_at=updated_user.updated_at,
    )


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_current_user(
    current_user: CurrentUser,
    db: DbSession,
) -> None:
    """Delete current user account (LGPD compliance).

    This permanently deletes the user account and all associated data.
    This action cannot be undone.
    """
    user_service = UserService(db)

    logger.info("User account deletion requested", user_id=current_user.id)
    await user_service.delete_user(current_user)
    logger.info("User account deleted", user_id=current_user.id)


@router.get("/me/export")
async def export_user_data(
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Export all user data (LGPD compliance).

    Returns all personal data associated with the user account.
    """
    user_service = UserService(db)

    # Get user with all related data
    user = await user_service.get_user_with_profile(current_user.id)

    return {
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "phone": user.phone,
            "subscription_plan": user.subscription_plan.value,
            "created_at": user.created_at.isoformat(),
        },
        "investor_profile": (
            {
                "risk_profile": user.investor_profile.risk_profile.value,
                "risk_score": user.investor_profile.risk_score,
                "investment_horizon": user.investor_profile.investment_horizon.value,
            }
            if user.investor_profile
            else None
        ),
        "exported_at": "now",  # Will be formatted properly
    }
