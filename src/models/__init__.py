# Import all models here for Alembic to detect them
from src.models.user import User
from src.models.profile import InvestorProfile, AssessmentSession
from src.models.portfolio import Portfolio, PortfolioAsset, PortfolioType, Transaction
from src.models.chat import ChatSession, ChatMessage, ChatFeedback
from src.models.alert import PriceAlert
from src.models.market import PriceHistory, PortfolioSnapshot
from src.models.broker import BrokerConnection, BrokerSyncHistory, BrokerType, ConnectionStatus, SyncStatus
from src.models.subscription import (
    Subscription,
    Payment,
    Coupon,
    SubscriptionStatus,
    PaymentStatus,
    PaymentMethod,
    PLAN_DETAILS,
)
from src.models.analytics import (
    Backtest,
    BacktestStatus,
    PriceForecastHistory,
    StrategyRecommendationHistory,
)
from src.models.community import (
    CommunityStrategy,
    StrategyUse,
    StrategyMLFeatures,
    StrategyStatus,
)
from src.models.api_key import APIKey, APIKeyPermission, APIRequestLog
from src.models.notification import Notification, NotificationPreferences, NotificationType, NotificationPriority
from src.models.settings import UserSettings, LLMProvider
from src.models.newsletter import NewsletterSubscriber

__all__ = [
    "User",
    "InvestorProfile",
    "AssessmentSession",
    "Portfolio",
    "PortfolioAsset",
    "PortfolioType",
    "Transaction",
    "ChatSession",
    "ChatMessage",
    "ChatFeedback",
    "PriceAlert",
    "PriceHistory",
    "PortfolioSnapshot",
    "BrokerConnection",
    "BrokerSyncHistory",
    "BrokerType",
    "ConnectionStatus",
    "SyncStatus",
    "Subscription",
    "Payment",
    "Coupon",
    "SubscriptionStatus",
    "PaymentStatus",
    "PaymentMethod",
    "PLAN_DETAILS",
    "Backtest",
    "BacktestStatus",
    "PriceForecastHistory",
    "StrategyRecommendationHistory",
    "CommunityStrategy",
    "StrategyUse",
    "StrategyMLFeatures",
    "StrategyStatus",
    "APIKey",
    "APIKeyPermission",
    "APIRequestLog",
    "Notification",
    "NotificationPreferences",
    "NotificationType",
    "NotificationPriority",
    "UserSettings",
    "LLMProvider",
    "NewsletterSubscriber",
]
