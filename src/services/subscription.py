"""Subscription service."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.models.subscription import (
    Subscription,
    Payment,
    Coupon,
    SubscriptionStatus,
    PaymentStatus,
    PaymentMethod,
    PLAN_DETAILS,
)
from src.models.user import User
from src.schemas.user import SubscriptionPlan
from src.schemas.subscription import (
    PlanInfo,
    PlanFeatures,
    SubscriptionCreate,
)

logger = structlog.get_logger()


class SubscriptionService:
    """Service for managing subscriptions and payments."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ===========================================
    # Plan Information
    # ===========================================

    def get_all_plans(self) -> list[PlanInfo]:
        """Get all available subscription plans."""
        plans = []
        for plan, details in PLAN_DETAILS.items():
            plans.append(PlanInfo(
                plan=plan,
                name=details["name"],
                description=details["description"],
                price_monthly=details["price_monthly"],
                price_yearly=details["price_yearly"],
                features=PlanFeatures(**details["features"]),
            ))
        return plans

    def get_plan_details(self, plan: SubscriptionPlan) -> dict:
        """Get details for a specific plan."""
        return PLAN_DETAILS.get(plan, PLAN_DETAILS[SubscriptionPlan.STARTER])

    def get_plan_price(self, plan: SubscriptionPlan, billing_cycle: str) -> Decimal:
        """Get price for a plan and billing cycle."""
        details = self.get_plan_details(plan)
        if billing_cycle == "yearly":
            return details["price_yearly"]
        return details["price_monthly"]

    # ===========================================
    # Subscription Management
    # ===========================================

    async def get_user_subscription(self, user_id: int) -> Subscription | None:
        """Get user's current subscription."""
        result = await self.db.execute(
            select(Subscription)
            .where(Subscription.user_id == user_id)
            .order_by(Subscription.created_at.desc())
        )
        return result.scalar_one_or_none()

    async def create_subscription(
        self,
        user: User,
        request: SubscriptionCreate,
    ) -> Subscription:
        """Create a new subscription for user."""
        # Check if user already has active subscription
        existing = await self.get_user_subscription(user.id)
        if existing and existing.status == SubscriptionStatus.ACTIVE:
            raise ValueError("User already has an active subscription")

        # Calculate period dates
        now = datetime.now(timezone.utc)
        if request.billing_cycle == "yearly":
            period_end = now + timedelta(days=365)
        else:
            period_end = now + timedelta(days=30)

        # Create subscription
        subscription = Subscription(
            user_id=user.id,
            plan=request.plan,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle=request.billing_cycle,
            current_period_start=now,
            current_period_end=period_end,
        )

        self.db.add(subscription)

        # Update user's subscription plan
        user.subscription_plan = request.plan

        await self.db.commit()
        await self.db.refresh(subscription)

        logger.info(
            "Subscription created",
            user_id=user.id,
            plan=request.plan.value,
            billing_cycle=request.billing_cycle,
        )

        return subscription

    async def create_free_subscription(self, user: User) -> Subscription:
        """Create a free starter subscription for new user."""
        now = datetime.now(timezone.utc)

        subscription = Subscription(
            user_id=user.id,
            plan=SubscriptionPlan.STARTER,
            status=SubscriptionStatus.ACTIVE,
            billing_cycle="monthly",
            current_period_start=now,
            current_period_end=now + timedelta(days=36500),  # ~100 years
        )

        self.db.add(subscription)
        await self.db.commit()
        await self.db.refresh(subscription)

        return subscription

    async def upgrade_subscription(
        self,
        user: User,
        new_plan: SubscriptionPlan,
        billing_cycle: str | None = None,
    ) -> Subscription:
        """Upgrade user's subscription to a new plan."""
        subscription = await self.get_user_subscription(user.id)

        if not subscription:
            raise ValueError("No subscription found")

        # Determine billing cycle
        if billing_cycle is None:
            billing_cycle = subscription.billing_cycle

        # Update subscription
        subscription.plan = new_plan
        subscription.billing_cycle = billing_cycle
        subscription.updated_at = datetime.now(timezone.utc)

        # Update user's plan
        user.subscription_plan = new_plan

        await self.db.commit()
        await self.db.refresh(subscription)

        logger.info(
            "Subscription upgraded",
            user_id=user.id,
            new_plan=new_plan.value,
        )

        return subscription

    async def cancel_subscription(
        self,
        user: User,
        cancel_immediately: bool = False,
        reason: str | None = None,
    ) -> Subscription:
        """Cancel user's subscription."""
        subscription = await self.get_user_subscription(user.id)

        if not subscription:
            raise ValueError("No subscription found")

        if subscription.status == SubscriptionStatus.CANCELLED:
            raise ValueError("Subscription is already cancelled")

        now = datetime.now(timezone.utc)
        subscription.cancelled_at = now

        if cancel_immediately:
            subscription.status = SubscriptionStatus.CANCELLED
            subscription.current_period_end = now
            # Downgrade to starter plan
            user.subscription_plan = SubscriptionPlan.STARTER
        else:
            # Subscription remains active until end of period
            subscription.status = SubscriptionStatus.CANCELLED

        await self.db.commit()
        await self.db.refresh(subscription)

        logger.info(
            "Subscription cancelled",
            user_id=user.id,
            immediately=cancel_immediately,
            reason=reason,
        )

        return subscription

    async def reactivate_subscription(self, user: User) -> Subscription:
        """Reactivate a cancelled subscription."""
        subscription = await self.get_user_subscription(user.id)

        if not subscription:
            raise ValueError("No subscription found")

        if subscription.status != SubscriptionStatus.CANCELLED:
            raise ValueError("Subscription is not cancelled")

        # Check if still within period
        if subscription.current_period_end < datetime.now(timezone.utc):
            raise ValueError("Subscription period has ended. Please create a new subscription.")

        subscription.status = SubscriptionStatus.ACTIVE
        subscription.cancelled_at = None

        # Restore user's plan
        user.subscription_plan = subscription.plan

        await self.db.commit()
        await self.db.refresh(subscription)

        logger.info("Subscription reactivated", user_id=user.id)

        return subscription

    # ===========================================
    # Payment Management
    # ===========================================

    async def get_user_payments(
        self,
        user_id: int,
        limit: int = 10,
        offset: int = 0,
    ) -> tuple[list[Payment], int]:
        """Get user's payment history."""
        # Get total count
        from sqlalchemy import func
        count_result = await self.db.execute(
            select(func.count(Payment.id))
            .where(Payment.user_id == user_id)
        )
        total = count_result.scalar() or 0

        # Get payments
        result = await self.db.execute(
            select(Payment)
            .where(Payment.user_id == user_id)
            .order_by(Payment.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        payments = list(result.scalars().all())

        return payments, total

    async def create_payment(
        self,
        subscription: Subscription,
        amount: Decimal,
        payment_method: PaymentMethod | None = None,
        description: str | None = None,
    ) -> Payment:
        """Create a payment record."""
        payment = Payment(
            subscription_id=subscription.id,
            user_id=subscription.user_id,
            amount=amount,
            status=PaymentStatus.PENDING,
            payment_method=payment_method,
            description=description or f"Assinatura {subscription.plan.value}",
        )

        self.db.add(payment)
        await self.db.commit()
        await self.db.refresh(payment)

        return payment

    async def mark_payment_paid(
        self,
        payment: Payment,
        card_last_four: str | None = None,
        card_brand: str | None = None,
    ) -> Payment:
        """Mark a payment as paid."""
        payment.status = PaymentStatus.PAID
        payment.paid_at = datetime.now(timezone.utc)
        payment.card_last_four = card_last_four
        payment.card_brand = card_brand

        await self.db.commit()
        await self.db.refresh(payment)

        logger.info("Payment marked as paid", payment_id=payment.id)

        return payment

    # ===========================================
    # Coupon Management
    # ===========================================

    async def validate_coupon(
        self,
        code: str,
        plan: SubscriptionPlan,
    ) -> Coupon | None:
        """Validate a coupon code."""
        result = await self.db.execute(
            select(Coupon)
            .where(Coupon.code == code.upper())
            .where(Coupon.is_active == True)
        )
        coupon = result.scalar_one_or_none()

        if not coupon:
            return None

        now = datetime.now(timezone.utc)

        # Check validity period
        if coupon.valid_from > now:
            return None
        if coupon.valid_until and coupon.valid_until < now:
            return None

        # Check usage limit
        if coupon.max_uses and coupon.times_used >= coupon.max_uses:
            return None

        # Check applicable plans
        if coupon.applicable_plans:
            applicable = [p.strip() for p in coupon.applicable_plans.split(",")]
            if plan.value not in applicable:
                return None

        return coupon

    async def apply_coupon(self, coupon: Coupon) -> None:
        """Increment coupon usage count."""
        coupon.times_used += 1
        await self.db.commit()

    def calculate_discounted_price(
        self,
        price: Decimal,
        coupon: Coupon,
    ) -> Decimal:
        """Calculate price after applying coupon discount."""
        if coupon.discount_percent:
            discount = price * Decimal(coupon.discount_percent) / 100
            return max(price - discount, Decimal("0"))
        elif coupon.discount_amount:
            return max(price - coupon.discount_amount, Decimal("0"))
        return price

    # ===========================================
    # Usage Tracking
    # ===========================================

    async def get_user_usage(self, user: User) -> dict:
        """Get user's current usage against plan limits."""
        plan_details = self.get_plan_details(user.subscription_plan)
        features = plan_details["features"]

        # Count current usage
        from sqlalchemy import func, and_

        # Count portfolios
        from src.models.portfolio import Portfolio
        portfolio_count = await self.db.execute(
            select(func.count(Portfolio.id))
            .where(Portfolio.user_id == user.id)
        )
        portfolios_used = portfolio_count.scalar() or 0

        # Count price alerts
        from src.models.alert import PriceAlert
        alert_count = await self.db.execute(
            select(func.count(PriceAlert.id))
            .where(PriceAlert.user_id == user.id)
            .where(PriceAlert.is_active == True)
        )
        alerts_used = alert_count.scalar() or 0

        # Count broker connections
        from src.models.broker import BrokerConnection, ConnectionStatus
        broker_count = await self.db.execute(
            select(func.count(BrokerConnection.id))
            .where(BrokerConnection.user_id == user.id)
            .where(BrokerConnection.status == ConnectionStatus.ACTIVE)
        )
        brokers_used = broker_count.scalar() or 0

        # Count AI messages this month
        from src.models.chat import ChatMessage, ChatSession
        from src.schemas.chat import MessageRole
        first_of_month = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        ai_messages_count = await self.db.execute(
            select(func.count(ChatMessage.id))
            .join(ChatSession, ChatMessage.session_id == ChatSession.id)
            .where(
                and_(
                    ChatSession.user_id == user.id,
                    ChatMessage.role == MessageRole.ASSISTANT,
                    ChatMessage.created_at >= first_of_month,
                )
            )
        )
        ai_messages_used = ai_messages_count.scalar() or 0

        usage = {
            "portfolios": portfolios_used,
            "price_alerts": alerts_used,
            "broker_connections": brokers_used,
            "ai_messages_this_month": ai_messages_used,
        }

        limits = {
            "portfolios": features["portfolios"],
            "price_alerts": features["price_alerts"],
            "broker_connections": features["broker_connections"],
            "ai_messages_per_month": features["ai_messages_per_month"],
        }

        # Calculate percentage used
        percentage_used = {}
        for key in usage:
            limit = limits.get(key, 0)
            if limit == -1:  # Unlimited
                percentage_used[key] = 0
            elif limit == 0:
                percentage_used[key] = 100 if usage[key] > 0 else 0
            else:
                percentage_used[key] = min(100, int(usage[key] / limit * 100))

        return {
            "plan": user.subscription_plan,
            "usage": usage,
            "limits": limits,
            "percentage_used": percentage_used,
        }

    async def check_limit(
        self,
        user: User,
        feature: str,
        increment: int = 1,
    ) -> bool:
        """Check if user can use more of a feature."""
        plan_details = self.get_plan_details(user.subscription_plan)
        limit = plan_details["features"].get(feature, 0)

        if limit == -1:  # Unlimited
            return True

        usage_data = await self.get_user_usage(user)
        current_usage = usage_data["usage"].get(feature, 0)

        return current_usage + increment <= limit
