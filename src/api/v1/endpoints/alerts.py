"""Price alert endpoints."""
from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select, func
import structlog

from src.core.deps import CurrentUser, DbSession
from src.models.alert import PriceAlert, AlertCondition as ModelAlertCondition
from src.schemas.alert import (
    PriceAlertCreate,
    PriceAlertUpdate,
    PriceAlertResponse,
    PriceAlertListResponse,
    AlertCondition,
)

router = APIRouter()
logger = structlog.get_logger()


@router.get("/", response_model=PriceAlertListResponse)
async def list_alerts(
    current_user: CurrentUser,
    db: DbSession,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    is_active: bool | None = None,
    ticker: str | None = None,
) -> PriceAlertListResponse:
    """List all price alerts for the current user."""
    query = select(PriceAlert).where(PriceAlert.user_id == current_user.id)

    if is_active is not None:
        query = query.where(PriceAlert.is_active == is_active)
    if ticker:
        query = query.where(PriceAlert.ticker == ticker.upper())

    # Count total
    count_query = select(func.count(PriceAlert.id)).where(
        PriceAlert.user_id == current_user.id
    )
    if is_active is not None:
        count_query = count_query.where(PriceAlert.is_active == is_active)
    if ticker:
        count_query = count_query.where(PriceAlert.ticker == ticker.upper())

    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Paginate and order
    query = (
        query.order_by(PriceAlert.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )

    result = await db.execute(query)
    alerts = result.scalars().all()

    return PriceAlertListResponse(
        alerts=[
            PriceAlertResponse(
                id=a.id,
                ticker=a.ticker,
                target_price=a.target_price,
                condition=AlertCondition(a.condition.value),
                is_active=a.is_active,
                is_triggered=a.is_triggered,
                triggered_at=a.triggered_at,
                triggered_price=a.triggered_price,
                notes=a.notes,
                created_at=a.created_at,
                updated_at=a.updated_at,
            )
            for a in alerts
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/", response_model=PriceAlertResponse, status_code=status.HTTP_201_CREATED)
async def create_alert(
    request: PriceAlertCreate,
    current_user: CurrentUser,
    db: DbSession,
) -> PriceAlertResponse:
    """Create a new price alert."""
    # Check if user already has an active alert for this ticker and condition
    existing_query = select(PriceAlert).where(
        PriceAlert.user_id == current_user.id,
        PriceAlert.ticker == request.ticker.upper(),
        PriceAlert.condition == ModelAlertCondition(request.condition.value),
        PriceAlert.is_active == True,
    )
    existing_result = await db.execute(existing_query)
    existing = existing_result.scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ja existe um alerta ativo para {request.ticker.upper()} com a condicao '{request.condition.value}'.",
        )

    alert = PriceAlert(
        user_id=current_user.id,
        ticker=request.ticker.upper(),
        target_price=request.target_price,
        condition=ModelAlertCondition(request.condition.value),
        notes=request.notes,
    )

    db.add(alert)
    await db.commit()
    await db.refresh(alert)

    logger.info(
        "Price alert created",
        user_id=current_user.id,
        alert_id=alert.id,
        ticker=alert.ticker,
    )

    return PriceAlertResponse(
        id=alert.id,
        ticker=alert.ticker,
        target_price=alert.target_price,
        condition=AlertCondition(alert.condition.value),
        is_active=alert.is_active,
        is_triggered=alert.is_triggered,
        triggered_at=alert.triggered_at,
        triggered_price=alert.triggered_price,
        notes=alert.notes,
        created_at=alert.created_at,
        updated_at=alert.updated_at,
    )


@router.patch("/{alert_id}", response_model=PriceAlertResponse)
async def update_alert(
    alert_id: int,
    request: PriceAlertUpdate,
    current_user: CurrentUser,
    db: DbSession,
) -> PriceAlertResponse:
    """Update a price alert."""
    result = await db.execute(
        select(PriceAlert).where(
            PriceAlert.id == alert_id,
            PriceAlert.user_id == current_user.id,
        )
    )
    alert = result.scalar_one_or_none()

    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alerta nao encontrado.",
        )

    update_data = request.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if field == "condition" and value:
            setattr(alert, field, ModelAlertCondition(value.value))
        else:
            setattr(alert, field, value)

    await db.commit()
    await db.refresh(alert)

    logger.info(
        "Price alert updated",
        user_id=current_user.id,
        alert_id=alert.id,
    )

    return PriceAlertResponse(
        id=alert.id,
        ticker=alert.ticker,
        target_price=alert.target_price,
        condition=AlertCondition(alert.condition.value),
        is_active=alert.is_active,
        is_triggered=alert.is_triggered,
        triggered_at=alert.triggered_at,
        triggered_price=alert.triggered_price,
        notes=alert.notes,
        created_at=alert.created_at,
        updated_at=alert.updated_at,
    )


@router.delete("/{alert_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_alert(
    alert_id: int,
    current_user: CurrentUser,
    db: DbSession,
) -> None:
    """Delete a price alert."""
    result = await db.execute(
        select(PriceAlert).where(
            PriceAlert.id == alert_id,
            PriceAlert.user_id == current_user.id,
        )
    )
    alert = result.scalar_one_or_none()

    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alerta nao encontrado.",
        )

    await db.delete(alert)
    await db.commit()

    logger.info(
        "Price alert deleted",
        user_id=current_user.id,
        alert_id=alert_id,
    )
