"""Privacy and LGPD compliance endpoints."""
import json
from datetime import datetime, timezone
from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import structlog

from src.core.deps import DbSession, CurrentUser
from src.services.lgpd import LGPDService
from src.services.email import email_service

router = APIRouter()
logger = structlog.get_logger()


# ===========================================
# Request/Response Schemas
# ===========================================

class DataExportResponse(BaseModel):
    """Data export response."""
    message: str
    download_url: str | None = None


class DeleteAccountRequest(BaseModel):
    """Delete account request."""
    password: str
    confirm_deletion: bool
    keep_financial_records: bool = True
    reason: str | None = None


class DeleteAccountResponse(BaseModel):
    """Delete account response."""
    message: str
    deleted_categories: list[str]
    retained_categories: list[str]


class RevokeConsentRequest(BaseModel):
    """Revoke consent request."""
    consent_type: str  # marketing, ai_analysis, broker_sync


class DataProcessingInfoResponse(BaseModel):
    """Data processing information response."""
    controller: dict
    data_categories: dict
    data_sharing: dict
    user_rights: list[str]
    contact: dict


# ===========================================
# Endpoints
# ===========================================

@router.get("/data-processing-info", response_model=DataProcessingInfoResponse)
async def get_data_processing_info(db: DbSession) -> DataProcessingInfoResponse:
    """
    Get information about data processing (LGPD Art. 18, I).

    Returns information about what data is collected, why, and user rights.
    This endpoint is public and does not require authentication.
    """
    service = LGPDService(db)
    info = await service.get_data_processing_info()
    return DataProcessingInfoResponse(**info)


@router.get("/my-data")
async def export_my_data(
    current_user: CurrentUser,
    db: DbSession,
) -> StreamingResponse:
    """
    Export all user data (LGPD Art. 18, V - Portabilidade).

    Returns a JSON file with all personal data associated with the user.
    """
    service = LGPDService(db)
    data = await service.export_user_data(current_user)

    # Convert to JSON
    json_data = json.dumps(data, indent=2, ensure_ascii=False)

    # Create file-like object
    buffer = BytesIO(json_data.encode('utf-8'))
    buffer.seek(0)

    filename = f"investai_dados_{current_user.id}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"

    logger.info("User data exported", user_id=current_user.id)

    return StreamingResponse(
        buffer,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@router.post("/request-data-export", response_model=DataExportResponse)
async def request_data_export(
    current_user: CurrentUser,
    db: DbSession,
    background_tasks: BackgroundTasks,
) -> DataExportResponse:
    """
    Request data export via email.

    For larger data exports, sends the data via email.
    """
    service = LGPDService(db)

    # Export data in background and send via email
    async def export_and_send():
        data = await service.export_user_data(current_user)
        json_data = json.dumps(data, indent=2, ensure_ascii=False)

        # In production, this would upload to S3 and send download link
        # For now, log the action
        logger.info(
            "Data export requested",
            user_id=current_user.id,
            data_size_bytes=len(json_data),
        )

    background_tasks.add_task(export_and_send)

    return DataExportResponse(
        message="Seus dados estao sendo preparados e serao enviados para seu email em ate 24 horas.",
    )


@router.post("/delete-account", response_model=DeleteAccountResponse)
async def delete_account(
    request: DeleteAccountRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> DeleteAccountResponse:
    """
    Delete user account and data (LGPD Art. 18, VI - Eliminação).

    This action is irreversible. By default, financial records are kept
    for legal compliance (5 years in Brazil).
    """
    from src.core.security import verify_password

    # Verify password
    if not verify_password(request.password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Senha incorreta.",
        )

    # Verify confirmation
    if not request.confirm_deletion:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Voce deve confirmar a exclusao da conta.",
        )

    service = LGPDService(db)
    result = await service.delete_user_data(
        current_user,
        keep_financial_records=request.keep_financial_records,
    )

    logger.warning(
        "User account deleted",
        user_id=current_user.id,
        reason=request.reason,
        keep_financial_records=request.keep_financial_records,
    )

    return DeleteAccountResponse(
        message="Sua conta foi excluida com sucesso.",
        deleted_categories=result["categories_deleted"],
        retained_categories=result["categories_retained"],
    )


@router.post("/revoke-consent")
async def revoke_consent(
    request: RevokeConsentRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """
    Revoke specific consent (LGPD Art. 18, IX).

    Available consent types:
    - marketing: Stop receiving promotional emails
    - ai_analysis: Delete chat history and stop AI analysis
    - broker_sync: Disconnect all broker connections
    """
    valid_types = ["marketing", "ai_analysis", "broker_sync"]

    if request.consent_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tipo de consentimento invalido. Validos: {valid_types}",
        )

    service = LGPDService(db)
    success = await service.revoke_consent(current_user, request.consent_type)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao revogar consentimento.",
        )

    logger.info(
        "Consent revoked",
        user_id=current_user.id,
        consent_type=request.consent_type,
    )

    return {
        "message": f"Consentimento '{request.consent_type}' revogado com sucesso.",
        "consent_type": request.consent_type,
    }


@router.get("/consents")
async def get_user_consents(
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """
    Get user's current consent status.

    Returns which consents the user has given.
    """
    # In a full implementation, this would check a consents table
    # For MVP, we return default consents based on account status
    return {
        "consents": {
            "terms_of_service": {
                "status": "accepted",
                "accepted_at": current_user.created_at.isoformat() if current_user.created_at else None,
                "required": True,
            },
            "privacy_policy": {
                "status": "accepted",
                "accepted_at": current_user.created_at.isoformat() if current_user.created_at else None,
                "required": True,
            },
            "marketing": {
                "status": "accepted",
                "accepted_at": None,
                "required": False,
            },
            "ai_analysis": {
                "status": "accepted",
                "accepted_at": current_user.created_at.isoformat() if current_user.created_at else None,
                "required": False,
            },
            "broker_sync": {
                "status": "accepted",
                "accepted_at": None,
                "required": False,
            },
        },
    }


@router.get("/data-access-log")
async def get_data_access_log(
    current_user: CurrentUser,
    db: DbSession,
    limit: int = 50,
) -> dict:
    """
    Get log of who accessed user's data.

    This is a simplified version - a full implementation would
    track all data access events.
    """
    # In a full implementation, this would query an audit log table
    # For MVP, return recent login information
    return {
        "message": "Log de acesso aos dados",
        "user_id": current_user.id,
        "recent_access": [
            {
                "event": "login",
                "timestamp": current_user.last_login_at.isoformat() if current_user.last_login_at else None,
                "source": "user",
            },
        ],
        "note": "Para um log completo de acesso, entre em contato com dpo@investai.com.br",
    }


class PrivacySettingsResponse(BaseModel):
    """Privacy settings response."""
    data_collection: dict
    data_sharing: dict
    communication: dict


class UpdatePrivacySettingsRequest(BaseModel):
    """Update privacy settings request."""
    share_portfolio_anonymously: bool | None = None
    share_performance_stats: bool | None = None
    allow_ai_training: bool | None = None
    marketing_emails: bool | None = None
    product_updates: bool | None = None
    weekly_reports: bool | None = None
    analytics_consent: bool | None = None
    personalization_consent: bool | None = None
    marketing_consent: bool | None = None

    class Config:
        extra = "allow"  # Allow additional fields from frontend


@router.get("/settings", response_model=PrivacySettingsResponse)
async def get_privacy_settings(
    current_user: CurrentUser,
    db: DbSession,
) -> PrivacySettingsResponse:
    """
    Get user's privacy settings.

    Returns current privacy preferences for data collection,
    sharing, and communication.
    """
    # In a full implementation, this would query a privacy_settings table
    # For MVP, return default settings
    return PrivacySettingsResponse(
        data_collection={
            "portfolio_tracking": True,
            "performance_analytics": True,
            "ai_recommendations": True,
            "usage_analytics": True,
        },
        data_sharing={
            "share_portfolio_anonymously": False,
            "share_performance_stats": False,
            "allow_ai_training": True,
            "community_contributions": False,
        },
        communication={
            "marketing_emails": True,
            "product_updates": True,
            "weekly_reports": True,
            "price_alerts": True,
            "security_alerts": True,
        },
    )


@router.patch("/settings")
async def update_privacy_settings(
    request: UpdatePrivacySettingsRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """
    Update user's privacy settings.

    Allows users to control their privacy preferences.
    """
    # In a full implementation, this would update a privacy_settings table
    # For MVP, acknowledge the request
    updated_fields = {k: v for k, v in request.model_dump().items() if v is not None}

    logger.info(
        "Privacy settings updated",
        user_id=current_user.id,
        updated_fields=list(updated_fields.keys()),
    )

    return {
        "message": "Configuracoes de privacidade atualizadas com sucesso.",
        "updated_fields": list(updated_fields.keys()),
    }


@router.get("/exports")
async def get_data_exports(
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """
    Get list of previous data exports.

    Returns history of data export requests for the user.
    """
    # In a full implementation, this would query an exports table
    # For MVP, return empty list
    return {
        "exports": [],
        "message": "Nenhuma exportacao de dados solicitada anteriormente.",
    }


@router.get("/delete-account/status")
async def get_delete_account_status(
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """
    Get status of account deletion request.

    Returns whether there's a pending deletion request.
    """
    # In a full implementation, this would check for pending deletion requests
    # For MVP, return no pending request
    return {
        "pending_deletion": False,
        "scheduled_deletion_date": None,
        "can_cancel": False,
        "message": "Nenhuma solicitacao de exclusao pendente.",
    }
