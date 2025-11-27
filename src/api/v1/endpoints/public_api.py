"""Public API endpoints for third-party integrations."""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
import structlog
from sqlalchemy import select

from src.core.deps import CurrentUser, DbSession
from src.models.api_key import APIKey, APIKeyPermission
from src.models.portfolio import Portfolio
from src.services.public_api import PublicAPIService, get_public_api_service
from src.services.market import market_service

router = APIRouter()
logger = structlog.get_logger()


# ===========================================
# Schemas
# ===========================================

class CreateAPIKeyRequest(BaseModel):
    """Request to create an API key."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    permission: str = Field(default="read_only", pattern="^(read_only|read_write|full_access)$")
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1000)
    rate_limit_per_day: int = Field(default=10000, ge=1, le=100000)
    allowed_ips: Optional[list[str]] = None
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)
    is_test_mode: bool = False


class APIKeyResponse(BaseModel):
    """API key response."""
    id: int
    name: str
    description: Optional[str]
    key_prefix: str
    permission: str
    rate_limit_per_minute: int
    rate_limit_per_day: int
    is_active: bool
    is_test_mode: bool
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    total_requests: int
    created_at: datetime


class NewAPIKeyResponse(APIKeyResponse):
    """Response when creating a new API key (includes raw key)."""
    raw_key: str


# ===========================================
# API Key Management Endpoints
# ===========================================

@router.post("/keys", response_model=NewAPIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: CreateAPIKeyRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> NewAPIKeyResponse:
    """
    Create a new API key.

    IMPORTANT: The raw_key is only shown once! Store it securely.
    """
    service = get_public_api_service(db)

    try:
        permission = APIKeyPermission(request.permission)

        api_key, raw_key = await service.create_api_key(
            user=current_user,
            name=request.name,
            description=request.description,
            permission=permission,
            rate_limit_per_minute=request.rate_limit_per_minute,
            rate_limit_per_day=request.rate_limit_per_day,
            allowed_ips=request.allowed_ips,
            expires_in_days=request.expires_in_days,
            is_test_mode=request.is_test_mode,
        )

        return NewAPIKeyResponse(
            id=api_key.id,
            name=api_key.name,
            description=api_key.description,
            key_prefix=api_key.key_prefix,
            permission=api_key.permission,
            rate_limit_per_minute=api_key.rate_limit_per_minute,
            rate_limit_per_day=api_key.rate_limit_per_day,
            is_active=api_key.is_active,
            is_test_mode=api_key.is_test_mode,
            expires_at=api_key.expires_at,
            last_used_at=api_key.last_used_at,
            total_requests=api_key.total_requests,
            created_at=api_key.created_at,
            raw_key=raw_key,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.get("/keys")
async def list_api_keys(
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """List all API keys for the current user."""
    service = get_public_api_service(db)
    keys = await service.list_user_keys(current_user)

    return {
        "keys": [
            APIKeyResponse(
                id=k.id,
                name=k.name,
                description=k.description,
                key_prefix=k.key_prefix,
                permission=k.permission,
                rate_limit_per_minute=k.rate_limit_per_minute,
                rate_limit_per_day=k.rate_limit_per_day,
                is_active=k.is_active,
                is_test_mode=k.is_test_mode,
                expires_at=k.expires_at,
                last_used_at=k.last_used_at,
                total_requests=k.total_requests,
                created_at=k.created_at,
            )
            for k in keys
        ],
        "total": len(keys),
    }


@router.delete("/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: int,
    current_user: CurrentUser,
    db: DbSession,
) -> None:
    """Revoke an API key."""
    service = get_public_api_service(db)

    if not await service.revoke_api_key(key_id, current_user):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found.",
        )


@router.get("/keys/{key_id}/stats")
async def get_key_stats(
    key_id: int,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Get usage statistics for an API key."""
    service = get_public_api_service(db)

    try:
        return await service.get_key_stats(key_id, current_user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


# ===========================================
# Public API Endpoints (authenticated via API key)
# ===========================================

async def get_api_key(
    db: DbSession,
    x_api_key: str = Header(..., alias="X-API-Key"),
    request: Request = None,
) -> APIKey:
    """Dependency to validate API key from header."""
    service = get_public_api_service(db)

    client_ip = None
    if request:
        client_ip = request.client.host if request.client else None

    api_key = await service.validate_api_key(
        raw_key=x_api_key,
        client_ip=client_ip,
    )

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key.",
            headers={"WWW-Authenticate": "API-Key"},
        )

    return api_key


@router.get("/v1/portfolio")
async def public_get_portfolio(
    db: DbSession,
    api_key: APIKey = Depends(get_api_key),
) -> dict:
    """
    Get user's portfolio via public API.

    Requires X-API-Key header.
    """
    # Get user's portfolios
    query = select(Portfolio).where(Portfolio.user_id == api_key.user_id)
    result = await db.execute(query)
    portfolios = result.scalars().all()

    portfolio_data = []
    for portfolio in portfolios:
        assets = []
        total_value = 0

        for asset in portfolio.assets:
            current_price = await market_service.get_current_price(asset.ticker)
            value = float(asset.quantity * (current_price or asset.average_price))
            total_value += value

            assets.append({
                "ticker": asset.ticker,
                "quantity": float(asset.quantity),
                "average_price": float(asset.average_price),
                "current_price": float(current_price) if current_price else None,
                "value": value,
                "asset_class": asset.asset_class.value,
            })

        portfolio_data.append({
            "id": portfolio.id,
            "name": portfolio.name,
            "is_primary": portfolio.is_primary,
            "total_value": total_value,
            "assets_count": len(assets),
            "assets": assets,
        })

    return {
        "portfolios": portfolio_data,
        "total_portfolios": len(portfolio_data),
    }


@router.get("/v1/market/quote/{ticker}")
async def public_get_quote(
    ticker: str,
    db: DbSession,
    api_key: APIKey = Depends(get_api_key),
) -> dict:
    """
    Get current quote for a ticker via public API.

    Requires X-API-Key header.
    """
    quote = await market_service.get_quote(ticker.upper())

    if not quote:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticker {ticker} not found.",
        )

    return {
        "ticker": quote.ticker,
        "name": quote.name,
        "price": float(quote.price),
        "change": float(quote.change),
        "change_percent": float(quote.change_percent),
        "volume": quote.volume,
        "market_cap": quote.market_cap,
        "currency": quote.currency,
    }


@router.get("/v1/market/history/{ticker}")
async def public_get_history(
    ticker: str,
    db: DbSession,
    period: str = Query(default="1mo", pattern="^(1d|5d|1mo|3mo|6mo|1y|2y|5y|max)$"),
    api_key: APIKey = Depends(get_api_key),
) -> dict:
    """
    Get price history for a ticker via public API.

    Requires X-API-Key header.
    """
    history = await market_service.get_history(ticker.upper(), period=period)

    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"History for {ticker} not found.",
        )

    return {
        "ticker": ticker.upper(),
        "period": period,
        "data": [
            {
                "date": h.date.isoformat(),
                "open": float(h.open),
                "high": float(h.high),
                "low": float(h.low),
                "close": float(h.close),
                "volume": h.volume,
            }
            for h in history
        ],
        "count": len(history),
    }


@router.get("/v1/docs")
async def public_api_docs() -> dict:
    """Get public API documentation."""
    return {
        "name": "InvestAI Public API",
        "version": "1.0",
        "description": "API para integração com a plataforma InvestAI",
        "authentication": {
            "type": "API Key",
            "header": "X-API-Key",
            "description": "Inclua sua chave de API no header X-API-Key",
        },
        "endpoints": [
            {
                "path": "/api/v1/public/v1/portfolio",
                "method": "GET",
                "description": "Obter carteiras do usuário",
                "permission": "read_only",
            },
            {
                "path": "/api/v1/public/v1/market/quote/{ticker}",
                "method": "GET",
                "description": "Obter cotação atual de um ativo",
                "permission": "read_only",
            },
            {
                "path": "/api/v1/public/v1/market/history/{ticker}",
                "method": "GET",
                "description": "Obter histórico de preços de um ativo",
                "permission": "read_only",
            },
        ],
        "rate_limits": {
            "default": {
                "per_minute": 60,
                "per_day": 10000,
            },
        },
        "response_format": "JSON",
        "errors": {
            "401": "Chave de API inválida ou expirada",
            "403": "Permissão insuficiente",
            "404": "Recurso não encontrado",
            "429": "Rate limit excedido",
        },
    }
