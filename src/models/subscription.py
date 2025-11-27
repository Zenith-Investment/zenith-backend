"""Subscription and payment models."""
from datetime import datetime
from decimal import Decimal
from enum import Enum

from sqlalchemy import Boolean, DateTime, Enum as SQLEnum, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.database import Base
from src.schemas.user import SubscriptionPlan


class PaymentStatus(str, Enum):
    """Payment status enum."""
    PENDING = "pending"
    PROCESSING = "processing"
    PAID = "paid"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


class PaymentMethod(str, Enum):
    """Payment method enum."""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    CARD = "card"  # Generic card (Stripe)
    PIX = "pix"
    BOLETO = "boleto"


class SubscriptionStatus(str, Enum):
    """Subscription status enum."""
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    TRIAL = "trial"


# Plan features and pricing
PLAN_DETAILS = {
    SubscriptionPlan.STARTER: {
        "name": "Starter",
        "price_monthly": Decimal("0.00"),
        "price_yearly": Decimal("0.00"),
        "features": {
            "portfolios": 1,
            "assets_per_portfolio": 10,
            "price_alerts": 3,
            "ai_messages_per_month": 20,
            "market_data_delay_minutes": 15,
            "broker_connections": 0,
            "export_formats": ["csv"],
            "email_reports": False,
            "priority_support": False,
        },
        "description": "Perfeito para comecar a investir",
    },
    SubscriptionPlan.SMART: {
        "name": "Smart",
        "price_monthly": Decimal("29.90"),
        "price_yearly": Decimal("299.00"),
        "features": {
            "portfolios": 3,
            "assets_per_portfolio": 50,
            "price_alerts": 20,
            "ai_messages_per_month": 100,
            "market_data_delay_minutes": 5,
            "broker_connections": 2,
            "export_formats": ["csv", "excel"],
            "email_reports": True,
            "priority_support": False,
        },
        "description": "Para investidores que querem mais controle",
    },
    SubscriptionPlan.PRO: {
        "name": "Pro",
        "price_monthly": Decimal("79.90"),
        "price_yearly": Decimal("799.00"),
        "features": {
            "portfolios": 10,
            "assets_per_portfolio": 200,
            "price_alerts": 100,
            "ai_messages_per_month": 500,
            "market_data_delay_minutes": 1,
            "broker_connections": 5,
            "export_formats": ["csv", "excel", "pdf"],
            "email_reports": True,
            "priority_support": True,
        },
        "description": "Para investidores serios",
    },
    SubscriptionPlan.PREMIUM: {
        "name": "Premium",
        "price_monthly": Decimal("199.90"),
        "price_yearly": Decimal("1999.00"),
        "features": {
            "portfolios": -1,  # Unlimited
            "assets_per_portfolio": -1,  # Unlimited
            "price_alerts": -1,  # Unlimited
            "ai_messages_per_month": -1,  # Unlimited
            "market_data_delay_minutes": 0,  # Real-time
            "broker_connections": -1,  # Unlimited
            "export_formats": ["csv", "excel", "pdf"],
            "email_reports": True,
            "priority_support": True,
        },
        "description": "Acesso completo sem limites",
    },
}


class Subscription(Base):
    """User subscription model."""
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    plan: Mapped[SubscriptionPlan] = mapped_column(
        SQLEnum(SubscriptionPlan, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    status: Mapped[SubscriptionStatus] = mapped_column(
        SQLEnum(SubscriptionStatus, values_callable=lambda x: [e.value for e in x]),
        default=SubscriptionStatus.ACTIVE,
        nullable=False,
    )
    billing_cycle: Mapped[str] = mapped_column(
        String(20),
        default="monthly",
        nullable=False,
    )  # monthly or yearly

    # Stripe/Payment provider IDs
    stripe_subscription_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    stripe_customer_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Dates
    current_period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    current_period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    trial_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    cancelled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
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

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="subscription")
    payments: Mapped[list["Payment"]] = relationship(
        "Payment",
        back_populates="subscription",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Subscription(id={self.id}, user_id={self.user_id}, plan={self.plan.value})>"


class Payment(Base):
    """Payment history model."""
    __tablename__ = "payments"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    subscription_id: Mapped[int] = mapped_column(
        ForeignKey("subscriptions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Payment details
    amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2),
        nullable=False,
    )
    currency: Mapped[str] = mapped_column(
        String(3),
        default="BRL",
        nullable=False,
    )
    status: Mapped[PaymentStatus] = mapped_column(
        SQLEnum(PaymentStatus, values_callable=lambda x: [e.value for e in x]),
        default=PaymentStatus.PENDING,
        nullable=False,
    )
    payment_method: Mapped[PaymentMethod | None] = mapped_column(
        SQLEnum(PaymentMethod, values_callable=lambda x: [e.value for e in x]),
        nullable=True,
    )

    # Stripe/Payment provider IDs
    stripe_payment_intent_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    stripe_invoice_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Invoice info
    invoice_number: Mapped[str | None] = mapped_column(String(50), nullable=True)
    invoice_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Card info (last 4 digits only, for display)
    card_last_four: Mapped[str | None] = mapped_column(String(4), nullable=True)
    card_brand: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Description
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Dates
    paid_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    refunded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    subscription: Mapped["Subscription"] = relationship("Subscription", back_populates="payments")
    user: Mapped["User"] = relationship("User", back_populates="payments")

    def __repr__(self) -> str:
        return f"<Payment(id={self.id}, amount={self.amount}, status={self.status.value})>"


class Coupon(Base):
    """Discount coupon model."""
    __tablename__ = "coupons"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
    )
    discount_percent: Mapped[int | None] = mapped_column(Integer, nullable=True)
    discount_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 2),
        nullable=True,
    )
    max_uses: Mapped[int | None] = mapped_column(Integer, nullable=True)
    times_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Validity
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    valid_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Restrictions
    applicable_plans: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )  # Comma-separated plan names, null = all plans
    first_subscription_only: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Coupon(id={self.id}, code={self.code})>"
