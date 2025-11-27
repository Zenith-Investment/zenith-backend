"""In-app notification model."""
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String, Text, func, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.database import Base


class NotificationType(str, PyEnum):
    """Notification type enum."""
    PRICE_ALERT = "price_alert"
    PORTFOLIO_UPDATE = "portfolio_update"
    RECOMMENDATION = "recommendation"
    COMMUNITY = "community"
    SUBSCRIPTION = "subscription"
    SYSTEM = "system"
    ACHIEVEMENT = "achievement"
    NEWS = "news"


class NotificationPriority(str, PyEnum):
    """Notification priority enum."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Notification(Base):
    """In-app notification model."""
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Notification content
    type: Mapped[NotificationType] = mapped_column(
        Enum(NotificationType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[NotificationPriority] = mapped_column(
        Enum(NotificationPriority, values_callable=lambda x: [e.value for e in x]),
        default=NotificationPriority.NORMAL,
        nullable=False,
    )

    # Additional data (JSON for flexibility)
    data: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Action URL (optional deep link)
    action_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Status
    is_read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="notifications")

    def __repr__(self) -> str:
        return f"<Notification(id={self.id}, type={self.type.value}, user_id={self.user_id})>"


class NotificationPreferences(Base):
    """User notification preferences."""
    __tablename__ = "notification_preferences"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Email preferences
    email_price_alerts: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    email_portfolio_updates: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    email_recommendations: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    email_community: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    email_news: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    email_daily_report: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    email_weekly_report: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Push preferences
    push_price_alerts: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    push_portfolio_updates: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    push_recommendations: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    push_community: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Quiet hours
    quiet_hours_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    quiet_hours_start: Mapped[int | None] = mapped_column(Integer, nullable=True)  # Hour 0-23
    quiet_hours_end: Mapped[int | None] = mapped_column(Integer, nullable=True)  # Hour 0-23

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        onupdate=func.now(),
        nullable=True,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="notification_preferences")

    def __repr__(self) -> str:
        return f"<NotificationPreferences(user_id={self.user_id})>"
