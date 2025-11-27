"""Subscription and payment schemas."""
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field

from src.schemas.user import SubscriptionPlan
from src.models.subscription import (
    SubscriptionStatus,
    PaymentStatus,
    PaymentMethod,
)


# ===========================================
# Plan Schemas
# ===========================================

class PlanFeatures(BaseModel):
    """Plan features schema."""
    portfolios: int
    assets_per_portfolio: int
    price_alerts: int
    ai_messages_per_month: int
    market_data_delay_minutes: int
    broker_connections: int
    export_formats: list[str]
    email_reports: bool
    priority_support: bool


class PlanInfo(BaseModel):
    """Plan information schema."""
    plan: SubscriptionPlan
    name: str
    description: str
    price_monthly: Decimal
    price_yearly: Decimal
    features: PlanFeatures


class PlansResponse(BaseModel):
    """Response with all available plans."""
    plans: list[PlanInfo]


# ===========================================
# Subscription Schemas
# ===========================================

class SubscriptionResponse(BaseModel):
    """Subscription response schema."""
    id: int
    plan: SubscriptionPlan
    status: SubscriptionStatus
    billing_cycle: str
    current_period_start: datetime
    current_period_end: datetime
    trial_end: datetime | None = None
    cancelled_at: datetime | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class SubscriptionCreate(BaseModel):
    """Create subscription request."""
    plan: SubscriptionPlan
    billing_cycle: str = Field(default="monthly", pattern="^(monthly|yearly)$")
    coupon_code: str | None = None
    payment_method_id: str | None = None  # Stripe payment method ID


class SubscriptionUpdate(BaseModel):
    """Update subscription request."""
    plan: SubscriptionPlan | None = None
    billing_cycle: str | None = Field(default=None, pattern="^(monthly|yearly)$")


class CancelSubscriptionRequest(BaseModel):
    """Cancel subscription request."""
    reason: str | None = None
    feedback: str | None = None
    cancel_immediately: bool = False


# ===========================================
# Payment Schemas
# ===========================================

class PaymentResponse(BaseModel):
    """Payment response schema."""
    id: int
    amount: Decimal
    currency: str
    status: PaymentStatus
    payment_method: PaymentMethod | None
    card_last_four: str | None
    card_brand: str | None
    description: str | None
    invoice_url: str | None
    paid_at: datetime | None
    created_at: datetime

    class Config:
        from_attributes = True


class PaymentHistoryResponse(BaseModel):
    """Payment history response."""
    payments: list[PaymentResponse]
    total: int


# ===========================================
# Checkout Schemas
# ===========================================

class CreateCheckoutRequest(BaseModel):
    """Create checkout session request."""
    plan: SubscriptionPlan
    billing_cycle: str = Field(default="monthly", pattern="^(monthly|yearly)$")
    success_url: str
    cancel_url: str
    coupon_code: str | None = None


class CheckoutResponse(BaseModel):
    """Checkout session response."""
    checkout_url: str
    session_id: str


class CreatePortalSessionRequest(BaseModel):
    """Create customer portal session request."""
    return_url: str


class PortalSessionResponse(BaseModel):
    """Customer portal session response."""
    portal_url: str


# ===========================================
# Coupon Schemas
# ===========================================

class CouponValidateRequest(BaseModel):
    """Validate coupon request."""
    code: str
    plan: SubscriptionPlan


class CouponValidateResponse(BaseModel):
    """Coupon validation response."""
    valid: bool
    code: str | None = None
    discount_percent: int | None = None
    discount_amount: Decimal | None = None
    message: str | None = None


# ===========================================
# Usage Schemas
# ===========================================

class UsageResponse(BaseModel):
    """Current usage response."""
    plan: SubscriptionPlan
    usage: dict
    limits: dict
    percentage_used: dict


# ===========================================
# Webhook Schemas
# ===========================================

class StripeWebhookEvent(BaseModel):
    """Stripe webhook event schema."""
    id: str
    type: str
    data: dict
