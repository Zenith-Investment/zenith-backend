"""Subscription endpoints."""
import stripe
from fastapi import APIRouter, Depends, HTTPException, status, Request, Header
import structlog

from src.core.config import settings
from src.core.deps import DbSession, CurrentUser
from src.schemas.subscription import (
    PlansResponse,
    PlanInfo,
    SubscriptionResponse,
    SubscriptionCreate,
    SubscriptionUpdate,
    CancelSubscriptionRequest,
    PaymentHistoryResponse,
    PaymentResponse,
    CouponValidateRequest,
    CouponValidateResponse,
    UsageResponse,
)
from src.services.subscription import SubscriptionService

router = APIRouter()
logger = structlog.get_logger()

# Configure Stripe
stripe.api_key = getattr(settings, "STRIPE_SECRET_KEY", None)


# ===========================================
# Plan Endpoints
# ===========================================

@router.get("/plans", response_model=PlansResponse)
async def get_plans(db: DbSession) -> PlansResponse:
    """Get all available subscription plans."""
    service = SubscriptionService(db)
    plans = service.get_all_plans()
    return PlansResponse(plans=plans)


# ===========================================
# Subscription Endpoints
# ===========================================

@router.get("/current", response_model=SubscriptionResponse | None)
async def get_current_subscription(
    current_user: CurrentUser,
    db: DbSession,
) -> SubscriptionResponse | None:
    """Get user's current subscription."""
    service = SubscriptionService(db)
    subscription = await service.get_user_subscription(current_user.id)

    if not subscription:
        return None

    return SubscriptionResponse.model_validate(subscription)


@router.post("/", response_model=SubscriptionResponse, status_code=status.HTTP_201_CREATED)
async def create_subscription(
    request: SubscriptionCreate,
    current_user: CurrentUser,
    db: DbSession,
) -> SubscriptionResponse:
    """Create a new subscription."""
    service = SubscriptionService(db)

    try:
        subscription = await service.create_subscription(current_user, request)
        logger.info(
            "Subscription created",
            user_id=current_user.id,
            plan=request.plan.value,
        )
        return SubscriptionResponse.model_validate(subscription)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.put("/", response_model=SubscriptionResponse)
async def update_subscription(
    request: SubscriptionUpdate,
    current_user: CurrentUser,
    db: DbSession,
) -> SubscriptionResponse:
    """Update subscription (upgrade/downgrade plan)."""
    service = SubscriptionService(db)

    try:
        if request.plan:
            subscription = await service.upgrade_subscription(
                current_user,
                request.plan,
                request.billing_cycle,
            )
        else:
            subscription = await service.get_user_subscription(current_user.id)
            if not subscription:
                raise ValueError("No subscription found")

        return SubscriptionResponse.model_validate(subscription)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/cancel", response_model=SubscriptionResponse)
async def cancel_subscription(
    request: CancelSubscriptionRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> SubscriptionResponse:
    """Cancel subscription."""
    service = SubscriptionService(db)

    try:
        subscription = await service.cancel_subscription(
            current_user,
            cancel_immediately=request.cancel_immediately,
            reason=request.reason,
        )
        logger.info(
            "Subscription cancelled",
            user_id=current_user.id,
            immediately=request.cancel_immediately,
        )
        return SubscriptionResponse.model_validate(subscription)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/reactivate", response_model=SubscriptionResponse)
async def reactivate_subscription(
    current_user: CurrentUser,
    db: DbSession,
) -> SubscriptionResponse:
    """Reactivate a cancelled subscription."""
    service = SubscriptionService(db)

    try:
        subscription = await service.reactivate_subscription(current_user)
        logger.info("Subscription reactivated", user_id=current_user.id)
        return SubscriptionResponse.model_validate(subscription)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


# ===========================================
# Payment Endpoints
# ===========================================

@router.get("/payments", response_model=PaymentHistoryResponse)
async def get_payment_history(
    current_user: CurrentUser,
    db: DbSession,
    limit: int = 10,
    offset: int = 0,
) -> PaymentHistoryResponse:
    """Get user's payment history."""
    service = SubscriptionService(db)
    payments, total = await service.get_user_payments(
        current_user.id,
        limit=limit,
        offset=offset,
    )

    return PaymentHistoryResponse(
        payments=[PaymentResponse.model_validate(p) for p in payments],
        total=total,
    )


# ===========================================
# Usage Endpoints
# ===========================================

@router.get("/usage", response_model=UsageResponse)
async def get_usage(
    current_user: CurrentUser,
    db: DbSession,
) -> UsageResponse:
    """Get user's current usage against plan limits."""
    service = SubscriptionService(db)
    usage_data = await service.get_user_usage(current_user)

    return UsageResponse(**usage_data)


# ===========================================
# Coupon Endpoints
# ===========================================

@router.post("/coupons/validate", response_model=CouponValidateResponse)
async def validate_coupon(
    request: CouponValidateRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> CouponValidateResponse:
    """Validate a coupon code."""
    service = SubscriptionService(db)
    coupon = await service.validate_coupon(request.code, request.plan)

    if not coupon:
        return CouponValidateResponse(
            valid=False,
            message="Cupom invalido ou expirado.",
        )

    return CouponValidateResponse(
        valid=True,
        code=coupon.code,
        discount_percent=coupon.discount_percent,
        discount_amount=coupon.discount_amount,
        message="Cupom valido!",
    )


# ===========================================
# Feature Check Endpoint
# ===========================================

@router.get("/can-use/{feature}")
async def check_feature_limit(
    feature: str,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Check if user can use more of a feature."""
    service = SubscriptionService(db)

    valid_features = [
        "portfolios",
        "price_alerts",
        "broker_connections",
        "ai_messages_per_month",
    ]

    if feature not in valid_features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Feature invalida. Validas: {valid_features}",
        )

    can_use = await service.check_limit(current_user, feature)

    return {
        "feature": feature,
        "can_use": can_use,
        "plan": current_user.subscription_plan.value,
    }


# ===========================================
# Stripe Webhook Endpoint
# ===========================================

@router.post("/webhook/stripe")
async def stripe_webhook(
    request: Request,
    db: DbSession,
    stripe_signature: str = Header(None, alias="Stripe-Signature"),
) -> dict:
    """Handle Stripe webhook events for payment processing."""
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import select
    from src.models.subscription import Subscription, Payment, SubscriptionStatus, PaymentStatus, PaymentMethod
    from src.models.user import User
    from src.schemas.user import SubscriptionPlan

    webhook_secret = getattr(settings, "STRIPE_WEBHOOK_SECRET", None)

    if not webhook_secret:
        logger.warning("Stripe webhook secret not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook not configured",
        )

    # Get raw body
    payload = await request.body()

    try:
        # Verify webhook signature
        event = stripe.Webhook.construct_event(
            payload, stripe_signature, webhook_secret
        )
    except ValueError as e:
        logger.error("Invalid payload", error=str(e))
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        logger.error("Invalid signature", error=str(e))
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_type = event["type"]
    data = event["data"]["object"]

    logger.info("Stripe webhook received", event_type=event_type)

    # Handle different event types
    if event_type == "checkout.session.completed":
        # Payment successful, create/update subscription
        customer_email = data.get("customer_details", {}).get("email")
        subscription_id = data.get("subscription")

        if customer_email:
            result = await db.execute(
                select(User).where(User.email == customer_email)
            )
            user = result.scalar_one_or_none()

            if user:
                # Get plan from metadata
                metadata = data.get("metadata", {})
                plan_name = metadata.get("plan", "smart")

                try:
                    plan = SubscriptionPlan(plan_name)
                except ValueError:
                    plan = SubscriptionPlan.SMART

                # Create or update subscription
                service = SubscriptionService(db)
                existing = await service.get_user_subscription(user.id)

                if existing:
                    existing.status = SubscriptionStatus.ACTIVE
                    existing.plan = plan
                    existing.stripe_subscription_id = subscription_id
                    existing.current_period_start = datetime.now(timezone.utc)
                    existing.current_period_end = datetime.now(timezone.utc) + timedelta(days=30)
                else:
                    subscription = Subscription(
                        user_id=user.id,
                        plan=plan,
                        status=SubscriptionStatus.ACTIVE,
                        stripe_subscription_id=subscription_id,
                        billing_cycle="monthly",
                        current_period_start=datetime.now(timezone.utc),
                        current_period_end=datetime.now(timezone.utc) + timedelta(days=30),
                    )
                    db.add(subscription)

                user.subscription_plan = plan
                await db.commit()

                logger.info(
                    "Subscription activated via Stripe",
                    user_id=user.id,
                    plan=plan.value,
                )

    elif event_type == "invoice.paid":
        # Recurring payment successful
        subscription_id = data.get("subscription")
        amount = data.get("amount_paid", 0) / 100  # Convert from cents

        if subscription_id:
            result = await db.execute(
                select(Subscription).where(
                    Subscription.stripe_subscription_id == subscription_id
                )
            )
            subscription = result.scalar_one_or_none()

            if subscription:
                # Record payment
                payment = Payment(
                    subscription_id=subscription.id,
                    user_id=subscription.user_id,
                    amount=amount,
                    status=PaymentStatus.PAID,
                    payment_method=PaymentMethod.CARD,
                    stripe_payment_intent_id=data.get("payment_intent"),
                    stripe_invoice_id=data.get("id"),
                    paid_at=datetime.now(timezone.utc),
                )
                db.add(payment)

                # Extend subscription period
                subscription.current_period_end = datetime.now(timezone.utc) + timedelta(days=30)

                await db.commit()

                logger.info(
                    "Payment recorded",
                    subscription_id=subscription.id,
                    amount=amount,
                )

    elif event_type == "invoice.payment_failed":
        # Payment failed
        subscription_id = data.get("subscription")

        if subscription_id:
            result = await db.execute(
                select(Subscription).where(
                    Subscription.stripe_subscription_id == subscription_id
                )
            )
            subscription = result.scalar_one_or_none()

            if subscription:
                subscription.status = SubscriptionStatus.PAST_DUE
                await db.commit()

                logger.warning(
                    "Payment failed",
                    subscription_id=subscription.id,
                    user_id=subscription.user_id,
                )

    elif event_type == "customer.subscription.deleted":
        # Subscription cancelled in Stripe
        subscription_id = data.get("id")

        if subscription_id:
            result = await db.execute(
                select(Subscription).where(
                    Subscription.stripe_subscription_id == subscription_id
                )
            )
            subscription = result.scalar_one_or_none()

            if subscription:
                subscription.status = SubscriptionStatus.CANCELLED
                subscription.cancelled_at = datetime.now(timezone.utc)

                # Downgrade user to starter
                user_result = await db.execute(
                    select(User).where(User.id == subscription.user_id)
                )
                user = user_result.scalar_one_or_none()
                if user:
                    user.subscription_plan = SubscriptionPlan.STARTER

                await db.commit()

                logger.info(
                    "Subscription cancelled via Stripe",
                    subscription_id=subscription.id,
                )

    return {"received": True, "event_type": event_type}
