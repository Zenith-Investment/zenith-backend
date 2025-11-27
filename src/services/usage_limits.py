"""
Usage Limits Service.

Enforces subscription-based usage limits across the platform.
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from src.core.exceptions import UsageLimitExceeded, SubscriptionRequired
from src.models.subscription import PLAN_DETAILS
from src.models.portfolio import Portfolio, PortfolioAsset
from src.models.alert import PriceAlert
from src.models.chat import ChatMessage, ChatSession
from src.models.api_key import APIKey
from src.models.user import User
from src.schemas.user import SubscriptionPlan

logger = structlog.get_logger()


class UsageLimitsService:
    """Service for checking and enforcing usage limits."""

    def __init__(self, db: AsyncSession):
        self.db = db

    def get_plan_limits(self, plan: SubscriptionPlan) -> dict:
        """Get limits for a subscription plan."""
        plan_info = PLAN_DETAILS.get(plan, PLAN_DETAILS[SubscriptionPlan.STARTER])
        return plan_info.get("features", {})

    async def check_portfolio_limit(self, user: User) -> tuple[bool, int, int]:
        """Check if user can create another portfolio."""
        limits = self.get_plan_limits(user.subscription_plan)
        max_portfolios = limits.get("portfolios", 1)

        if max_portfolios == -1:
            return True, 0, -1

        result = await self.db.execute(
            select(func.count(Portfolio.id)).where(Portfolio.user_id == user.id)
        )
        current_count = result.scalar() or 0

        can_create = current_count < max_portfolios
        return can_create, current_count, max_portfolios

    async def enforce_portfolio_limit(self, user: User) -> None:
        """Raise exception if portfolio limit exceeded."""
        can_create, current, limit = await self.check_portfolio_limit(user)
        if not can_create:
            raise UsageLimitExceeded(
                resource="carteiras",
                current=current,
                limit=limit,
                upgrade_message="Faca upgrade para criar mais carteiras.",
            )

    async def check_assets_limit(self, user: User, portfolio_id: int) -> tuple[bool, int, int]:
        """Check if user can add another asset to portfolio."""
        limits = self.get_plan_limits(user.subscription_plan)
        max_assets = limits.get("assets_per_portfolio", 10)

        if max_assets == -1:
            return True, 0, -1

        result = await self.db.execute(
            select(func.count(PortfolioAsset.id)).where(
                PortfolioAsset.portfolio_id == portfolio_id
            )
        )
        current_count = result.scalar() or 0

        can_add = current_count < max_assets
        return can_add, current_count, max_assets

    async def enforce_assets_limit(self, user: User, portfolio_id: int) -> None:
        """Raise exception if assets limit exceeded."""
        can_add, current, limit = await self.check_assets_limit(user, portfolio_id)
        if not can_add:
            raise UsageLimitExceeded(
                resource="ativos por carteira",
                current=current,
                limit=limit,
                upgrade_message="Faca upgrade para adicionar mais ativos.",
            )

    async def check_alerts_limit(self, user: User) -> tuple[bool, int, int]:
        """Check if user can create another price alert."""
        limits = self.get_plan_limits(user.subscription_plan)
        max_alerts = limits.get("price_alerts", 3)

        if max_alerts == -1:
            return True, 0, -1

        result = await self.db.execute(
            select(func.count(PriceAlert.id)).where(
                and_(
                    PriceAlert.user_id == user.id,
                    PriceAlert.is_active == True,
                )
            )
        )
        current_count = result.scalar() or 0

        can_create = current_count < max_alerts
        return can_create, current_count, max_alerts

    async def enforce_alerts_limit(self, user: User) -> None:
        """Raise exception if alerts limit exceeded."""
        can_create, current, limit = await self.check_alerts_limit(user)
        if not can_create:
            raise UsageLimitExceeded(
                resource="alertas de preco",
                current=current,
                limit=limit,
                upgrade_message="Faca upgrade para criar mais alertas.",
            )

    async def check_ai_messages_limit(self, user: User) -> tuple[bool, int, int]:
        """Check if user can send another AI message this month."""
        limits = self.get_plan_limits(user.subscription_plan)
        max_messages = limits.get("ai_messages_per_month", 20)

        if max_messages == -1:
            return True, 0, -1

        month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        result = await self.db.execute(
            select(func.count(ChatMessage.id))
            .join(ChatSession, ChatMessage.session_id == ChatSession.id)
            .where(
                and_(
                    ChatSession.user_id == user.id,
                    ChatMessage.role == "user",
                    ChatMessage.created_at >= month_start,
                )
            )
        )
        current_count = result.scalar() or 0

        can_send = current_count < max_messages
        return can_send, current_count, max_messages

    async def enforce_ai_messages_limit(self, user: User) -> None:
        """Raise exception if AI messages limit exceeded."""
        can_send, current, limit = await self.check_ai_messages_limit(user)
        if not can_send:
            raise UsageLimitExceeded(
                resource="mensagens de IA por mes",
                current=current,
                limit=limit,
                upgrade_message="Faca upgrade para enviar mais mensagens a IA.",
            )

    async def check_api_keys_limit(self, user: User) -> tuple[bool, int, int]:
        """Check if user can create another API key."""
        limits_map = {
            SubscriptionPlan.STARTER: 1,
            SubscriptionPlan.SMART: 3,
            SubscriptionPlan.PRO: 10,
            SubscriptionPlan.PREMIUM: -1,
        }
        max_keys = limits_map.get(user.subscription_plan, 1)

        if max_keys == -1:
            return True, 0, -1

        result = await self.db.execute(
            select(func.count(APIKey.id)).where(
                and_(
                    APIKey.user_id == user.id,
                    APIKey.is_active == True,
                )
            )
        )
        current_count = result.scalar() or 0

        can_create = current_count < max_keys
        return can_create, current_count, max_keys

    async def enforce_api_keys_limit(self, user: User) -> None:
        """Raise exception if API keys limit exceeded."""
        can_create, current, limit = await self.check_api_keys_limit(user)
        if not can_create:
            raise UsageLimitExceeded(
                resource="chaves de API",
                current=current,
                limit=limit,
                upgrade_message="Faca upgrade para criar mais chaves de API.",
            )

    def check_feature_access(self, user: User, feature: str) -> bool:
        """Check if user has access to a specific feature."""
        limits = self.get_plan_limits(user.subscription_plan)
        return limits.get(feature, False)

    def enforce_feature_access(self, user: User, feature: str, feature_name: str) -> None:
        """Raise exception if user doesn't have access to feature."""
        if not self.check_feature_access(user, feature):
            required_plan = "PREMIUM"
            for plan in [SubscriptionPlan.SMART, SubscriptionPlan.PRO, SubscriptionPlan.PREMIUM]:
                plan_limits = self.get_plan_limits(plan)
                if plan_limits.get(feature, False):
                    required_plan = plan.value.upper()
                    break

            raise SubscriptionRequired(
                feature=feature_name,
                required_plan=required_plan,
            )

    async def get_usage_summary(self, user: User) -> dict:
        """Get complete usage summary for user."""
        limits = self.get_plan_limits(user.subscription_plan)

        _, portfolios_current, portfolios_limit = await self.check_portfolio_limit(user)
        _, alerts_current, alerts_limit = await self.check_alerts_limit(user)
        _, messages_current, messages_limit = await self.check_ai_messages_limit(user)
        _, api_keys_current, api_keys_limit = await self.check_api_keys_limit(user)

        return {
            "plan": user.subscription_plan.value,
            "plan_name": PLAN_DETAILS[user.subscription_plan]["name"],
            "usage": {
                "portfolios": {
                    "current": portfolios_current,
                    "limit": portfolios_limit,
                    "unlimited": portfolios_limit == -1,
                },
                "alerts": {
                    "current": alerts_current,
                    "limit": alerts_limit,
                    "unlimited": alerts_limit == -1,
                },
                "ai_messages_this_month": {
                    "current": messages_current,
                    "limit": messages_limit,
                    "unlimited": messages_limit == -1,
                },
                "api_keys": {
                    "current": api_keys_current,
                    "limit": api_keys_limit,
                    "unlimited": api_keys_limit == -1,
                },
            },
            "features": {
                "email_reports": limits.get("email_reports", False),
                "priority_support": limits.get("priority_support", False),
                "broker_connections": limits.get("broker_connections", 0),
                "export_formats": limits.get("export_formats", ["csv"]),
                "market_data_delay_minutes": limits.get("market_data_delay_minutes", 15),
            },
        }


def get_usage_limits_service(db: AsyncSession) -> UsageLimitsService:
    """Get usage limits service instance."""
    return UsageLimitsService(db)
