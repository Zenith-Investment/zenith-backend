from fastapi import APIRouter

from src.api.v1.endpoints import auth, users, portfolio, market, chat, profile, recommendations, transactions, alerts, exports, ws, brokers, subscriptions, privacy, health, analytics, reports, community, public_api, dashboard, notifications, settings, newsletter

api_router = APIRouter()

# Authentication endpoints
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"],
)

# User management endpoints
api_router.include_router(
    users.router,
    prefix="/users",
    tags=["Users"],
)

# Investor profile endpoints
api_router.include_router(
    profile.router,
    prefix="/profile",
    tags=["Investor Profile"],
)

# Portfolio management endpoints
api_router.include_router(
    portfolio.router,
    prefix="/portfolio",
    tags=["Portfolio"],
)

# Market data endpoints
api_router.include_router(
    market.router,
    prefix="/market",
    tags=["Market Data"],
)

# AI Chat endpoints
api_router.include_router(
    chat.router,
    prefix="/chat",
    tags=["AI Chat"],
)

# AI Recommendations endpoints
api_router.include_router(
    recommendations.router,
    prefix="/recommendations",
    tags=["AI Recommendations"],
)

# Transaction history endpoints
api_router.include_router(
    transactions.router,
    prefix="/transactions",
    tags=["Transactions"],
)

# Price alerts endpoints
api_router.include_router(
    alerts.router,
    prefix="/alerts",
    tags=["Price Alerts"],
)

# Export endpoints
api_router.include_router(
    exports.router,
    prefix="/exports",
    tags=["Exports"],
)

# WebSocket endpoints
api_router.include_router(
    ws.router,
    prefix="/ws",
    tags=["WebSocket"],
)

# Broker integration endpoints
api_router.include_router(
    brokers.router,
    prefix="/brokers",
    tags=["Broker Integrations"],
)

# Subscription management endpoints
api_router.include_router(
    subscriptions.router,
    prefix="/subscriptions",
    tags=["Subscriptions"],
)

# Privacy and LGPD compliance endpoints
api_router.include_router(
    privacy.router,
    prefix="/privacy",
    tags=["Privacy & LGPD"],
)

# Health check endpoints
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health"],
)

# Advanced Analytics endpoints
api_router.include_router(
    analytics.router,
    prefix="/analytics",
    tags=["Advanced Analytics"],
)

# Custom Reports endpoints
api_router.include_router(
    reports.router,
    prefix="/reports",
    tags=["Reports"],
)

# Community Strategies endpoints
api_router.include_router(
    community.router,
    prefix="/community",
    tags=["Community Strategies"],
)

# Public API endpoints
api_router.include_router(
    public_api.router,
    prefix="/public",
    tags=["Public API"],
)

# Dashboard endpoints
api_router.include_router(
    dashboard.router,
    prefix="/dashboard",
    tags=["Dashboard"],
)

# Notifications endpoints
api_router.include_router(
    notifications.router,
    prefix="/notifications",
    tags=["Notifications"],
)

# Settings endpoints
api_router.include_router(
    settings.router,
    prefix="/settings",
    tags=["Settings"],
)

# Newsletter endpoints (public)
api_router.include_router(
    newsletter.router,
    prefix="/newsletter",
    tags=["Newsletter"],
)
