"""In-app notifications endpoints."""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func, and_, update
import structlog

from src.core.deps import CurrentUser, DbSession
from src.models.notification import (
    Notification,
    NotificationPreferences,
    NotificationType,
    NotificationPriority,
)

router = APIRouter()
logger = structlog.get_logger()


class NotificationResponse(BaseModel):
    """Notification response."""
    id: int
    type: str
    title: str
    message: str
    priority: str
    data: Optional[dict]
    action_url: Optional[str]
    is_read: bool
    created_at: datetime
    read_at: Optional[datetime]


class NotificationPreferencesUpdate(BaseModel):
    """Update notification preferences."""
    email_price_alerts: Optional[bool] = None
    email_portfolio_updates: Optional[bool] = None
    email_recommendations: Optional[bool] = None
    email_community: Optional[bool] = None
    email_news: Optional[bool] = None
    email_daily_report: Optional[bool] = None
    email_weekly_report: Optional[bool] = None
    push_price_alerts: Optional[bool] = None
    push_portfolio_updates: Optional[bool] = None
    push_recommendations: Optional[bool] = None
    push_community: Optional[bool] = None
    quiet_hours_enabled: Optional[bool] = None
    quiet_hours_start: Optional[int] = Field(None, ge=0, le=23)
    quiet_hours_end: Optional[int] = Field(None, ge=0, le=23)


@router.get("/")
async def list_notifications(
    current_user: CurrentUser,
    db: DbSession,
    unread_only: bool = False,
    type: Optional[str] = None,
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """List user notifications."""
    query = select(Notification).where(Notification.user_id == current_user.id)

    if unread_only:
        query = query.where(Notification.is_read == False)

    if type:
        try:
            notification_type = NotificationType(type)
            query = query.where(Notification.type == notification_type)
        except ValueError:
            pass

    count_query = select(func.count(Notification.id)).where(
        Notification.user_id == current_user.id
    )
    if unread_only:
        count_query = count_query.where(Notification.is_read == False)
    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    query = query.order_by(Notification.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    notifications = result.scalars().all()

    return {
        "notifications": [
            NotificationResponse(
                id=n.id,
                type=n.type.value,
                title=n.title,
                message=n.message,
                priority=n.priority.value,
                data=n.data,
                action_url=n.action_url,
                is_read=n.is_read,
                created_at=n.created_at,
                read_at=n.read_at,
            )
            for n in notifications
        ],
        "total": total,
        "unread_count": await _get_unread_count(db, current_user.id),
        "limit": limit,
        "offset": offset,
    }


@router.post("/{notification_id}/read")
async def mark_as_read(
    notification_id: int,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Mark a notification as read."""
    result = await db.execute(
        select(Notification).where(
            and_(
                Notification.id == notification_id,
                Notification.user_id == current_user.id,
            )
        )
    )
    notification = result.scalar_one_or_none()

    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notificacao nao encontrada.",
        )

    if not notification.is_read:
        notification.is_read = True
        notification.read_at = datetime.utcnow()
        await db.commit()

    return {"success": True, "message": "Notificacao marcada como lida."}


@router.post("/read-all")
async def mark_all_as_read(
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Mark all notifications as read."""
    await db.execute(
        update(Notification)
        .where(
            and_(
                Notification.user_id == current_user.id,
                Notification.is_read == False,
            )
        )
        .values(is_read=True, read_at=datetime.utcnow())
    )
    await db.commit()

    return {"success": True, "message": "Todas as notificacoes foram marcadas como lidas."}


@router.delete("/{notification_id}")
async def delete_notification(
    notification_id: int,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Delete a notification."""
    result = await db.execute(
        select(Notification).where(
            and_(
                Notification.id == notification_id,
                Notification.user_id == current_user.id,
            )
        )
    )
    notification = result.scalar_one_or_none()

    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notificacao nao encontrada.",
        )

    await db.delete(notification)
    await db.commit()

    return {"success": True, "message": "Notificacao excluida."}


@router.delete("/")
async def delete_all_read(
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Delete all read notifications."""
    result = await db.execute(
        select(Notification).where(
            and_(
                Notification.user_id == current_user.id,
                Notification.is_read == True,
            )
        )
    )
    notifications = result.scalars().all()
    deleted_count = len(notifications)

    for n in notifications:
        await db.delete(n)

    await db.commit()

    return {
        "success": True,
        "message": f"{deleted_count} notificacoes excluidas.",
        "deleted_count": deleted_count,
    }


@router.get("/preferences")
async def get_preferences(
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Get notification preferences."""
    result = await db.execute(
        select(NotificationPreferences).where(
            NotificationPreferences.user_id == current_user.id
        )
    )
    prefs = result.scalar_one_or_none()

    if not prefs:
        prefs = NotificationPreferences(user_id=current_user.id)
        db.add(prefs)
        await db.commit()
        await db.refresh(prefs)

    return {
        "email": {
            "price_alerts": prefs.email_price_alerts,
            "portfolio_updates": prefs.email_portfolio_updates,
            "recommendations": prefs.email_recommendations,
            "community": prefs.email_community,
            "news": prefs.email_news,
            "daily_report": prefs.email_daily_report,
            "weekly_report": prefs.email_weekly_report,
        },
        "push": {
            "price_alerts": prefs.push_price_alerts,
            "portfolio_updates": prefs.push_portfolio_updates,
            "recommendations": prefs.push_recommendations,
            "community": prefs.push_community,
        },
        "quiet_hours": {
            "enabled": prefs.quiet_hours_enabled,
            "start": prefs.quiet_hours_start,
            "end": prefs.quiet_hours_end,
        },
    }


@router.put("/preferences")
async def update_preferences(
    update_data: NotificationPreferencesUpdate,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Update notification preferences."""
    result = await db.execute(
        select(NotificationPreferences).where(
            NotificationPreferences.user_id == current_user.id
        )
    )
    prefs = result.scalar_one_or_none()

    if not prefs:
        prefs = NotificationPreferences(user_id=current_user.id)
        db.add(prefs)

    for field, value in update_data.model_dump(exclude_unset=True).items():
        setattr(prefs, field, value)

    await db.commit()
    await db.refresh(prefs)

    return {
        "success": True,
        "message": "Preferencias atualizadas com sucesso.",
    }


async def _get_unread_count(db: DbSession, user_id: int) -> int:
    """Get unread notification count for user."""
    result = await db.execute(
        select(func.count(Notification.id)).where(
            and_(
                Notification.user_id == user_id,
                Notification.is_read == False,
            )
        )
    )
    return result.scalar() or 0


async def create_notification(
    db: DbSession,
    user_id: int,
    type: NotificationType,
    title: str,
    message: str,
    priority: NotificationPriority = NotificationPriority.NORMAL,
    data: Optional[dict] = None,
    action_url: Optional[str] = None,
) -> Notification:
    """Create a new notification for a user."""
    notification = Notification(
        user_id=user_id,
        type=type,
        title=title,
        message=message,
        priority=priority,
        data=data,
        action_url=action_url,
    )

    db.add(notification)
    await db.commit()
    await db.refresh(notification)

    logger.info(
        "Notification created",
        user_id=user_id,
        type=type.value,
        title=title,
    )

    return notification
