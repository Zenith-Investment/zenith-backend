"""Price alert models."""
from datetime import datetime
from decimal import Decimal
from enum import Enum

from sqlalchemy import Boolean, DateTime, Enum as SQLEnum, ForeignKey, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.database import Base


class AlertCondition(str, Enum):
    ABOVE = "above"
    BELOW = "below"


class PriceAlert(Base):
    __tablename__ = "price_alerts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    target_price: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    condition: Mapped[AlertCondition] = mapped_column(
        SQLEnum(AlertCondition, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_triggered: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    triggered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    triggered_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 2), nullable=True)

    notes: Mapped[str | None] = mapped_column(String(500), nullable=True)

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
    user: Mapped["User"] = relationship("User", back_populates="price_alerts")

    def __repr__(self) -> str:
        return f"<PriceAlert(id={self.id}, ticker={self.ticker}, target={self.target_price})>"
