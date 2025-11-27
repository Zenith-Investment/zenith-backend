from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.database import Base
from src.schemas.user import SubscriptionPlan


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(100), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    cpf_encrypted: Mapped[str | None] = mapped_column(String(255), nullable=True)
    avatar_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    subscription_plan: Mapped[SubscriptionPlan] = mapped_column(
        Enum(SubscriptionPlan, values_callable=lambda x: [e.value for e in x]),
        default=SubscriptionPlan.STARTER,
        nullable=False,
    )

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
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    investor_profile: Mapped["InvestorProfile"] = relationship(
        "InvestorProfile",
        back_populates="user",
        uselist=False,
    )
    portfolios: Mapped[list["Portfolio"]] = relationship(
        "Portfolio",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    chat_sessions: Mapped[list["ChatSession"]] = relationship(
        "ChatSession",
        back_populates="user",
    )
    price_alerts: Mapped[list["PriceAlert"]] = relationship(
        "PriceAlert",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    broker_connections: Mapped[list["BrokerConnection"]] = relationship(
        "BrokerConnection",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    subscription: Mapped["Subscription"] = relationship(
        "Subscription",
        back_populates="user",
        uselist=False,
    )
    payments: Mapped[list["Payment"]] = relationship(
        "Payment",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    # Analytics relationships
    backtests: Mapped[list["Backtest"]] = relationship(
        "Backtest",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    forecasts: Mapped[list["PriceForecastHistory"]] = relationship(
        "PriceForecastHistory",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    strategy_recommendations: Mapped[list["StrategyRecommendationHistory"]] = relationship(
        "StrategyRecommendationHistory",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    # Notification relationships
    notifications: Mapped[list["Notification"]] = relationship(
        "Notification",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    notification_preferences: Mapped["NotificationPreferences"] = relationship(
        "NotificationPreferences",
        back_populates="user",
        uselist=False,
    )
    settings: Mapped["UserSettings"] = relationship(
        "UserSettings",
        back_populates="user",
        uselist=False,
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"
