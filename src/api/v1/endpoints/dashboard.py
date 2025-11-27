"""Dashboard summary endpoint for frontend integration."""
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select, func, and_
import structlog

from src.core.deps import CurrentUser, DbSession
from src.models.portfolio import Portfolio, PortfolioAsset
from src.models.alert import PriceAlert
from src.models.analytics import Backtest, PriceForecastHistory
from src.models.community import CommunityStrategy, StrategyUse
from src.models.notification import Notification
from src.services.market import market_service
from src.services.usage_limits import get_usage_limits_service

router = APIRouter()
logger = structlog.get_logger()


@router.get("/summary")
async def get_dashboard_summary(
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """
    Get complete dashboard summary for the frontend.

    Returns all essential data in a single call for optimal performance.
    """
    # Get primary portfolio
    portfolio_query = select(Portfolio).where(
        and_(
            Portfolio.user_id == current_user.id,
            Portfolio.is_primary == True,
        )
    )
    result = await db.execute(portfolio_query)
    primary_portfolio = result.scalar_one_or_none()

    # Calculate portfolio value
    total_portfolio_value = Decimal("0")
    total_invested = Decimal("0")
    assets_data = []

    if primary_portfolio:
        for asset in primary_portfolio.assets:
            current_price = await market_service.get_current_price(asset.ticker)
            current_price_val = current_price or asset.average_price
            asset_value = asset.quantity * current_price_val
            invested = asset.quantity * asset.average_price
            profit_loss = asset_value - invested

            total_portfolio_value += asset_value
            total_invested += invested

            assets_data.append({
                "ticker": asset.ticker,
                "quantity": float(asset.quantity),
                "average_price": float(asset.average_price),
                "current_price": float(current_price_val),
                "value": float(asset_value),
                "profit_loss": float(profit_loss),
                "profit_loss_pct": float(profit_loss / invested * 100) if invested > 0 else 0,
            })

    # Sort by value descending
    assets_data.sort(key=lambda x: x["value"], reverse=True)

    # Calculate total P&L
    total_profit_loss = total_portfolio_value - total_invested
    total_profit_loss_pct = float(total_profit_loss / total_invested * 100) if total_invested > 0 else 0

    # Get portfolio count
    portfolios_count_result = await db.execute(
        select(func.count(Portfolio.id)).where(Portfolio.user_id == current_user.id)
    )
    portfolios_count = portfolios_count_result.scalar() or 0

    # Get active alerts count
    alerts_result = await db.execute(
        select(func.count(PriceAlert.id)).where(
            and_(
                PriceAlert.user_id == current_user.id,
                PriceAlert.is_active == True,
            )
        )
    )
    active_alerts = alerts_result.scalar() or 0

    # Get triggered alerts (last 7 days)
    week_ago = datetime.utcnow() - timedelta(days=7)
    triggered_result = await db.execute(
        select(func.count(PriceAlert.id)).where(
            and_(
                PriceAlert.user_id == current_user.id,
                PriceAlert.triggered_at >= week_ago,
            )
        )
    )
    triggered_alerts = triggered_result.scalar() or 0

    # Get unread notifications count
    unread_result = await db.execute(
        select(func.count(Notification.id)).where(
            and_(
                Notification.user_id == current_user.id,
                Notification.is_read == False,
            )
        )
    )
    unread_notifications = unread_result.scalar() or 0

    # Get recent backtests (last 5)
    backtests_query = (
        select(Backtest)
        .where(Backtest.user_id == current_user.id)
        .order_by(Backtest.created_at.desc())
        .limit(5)
    )
    backtests_result = await db.execute(backtests_query)
    recent_backtests = backtests_result.scalars().all()

    # Get recent forecasts (last 5)
    forecasts_query = (
        select(PriceForecastHistory)
        .where(PriceForecastHistory.user_id == current_user.id)
        .order_by(PriceForecastHistory.created_at.desc())
        .limit(5)
    )
    forecasts_result = await db.execute(forecasts_query)
    recent_forecasts = forecasts_result.scalars().all()

    # Get active community strategies being used
    strategies_query = (
        select(StrategyUse)
        .where(
            and_(
                StrategyUse.user_id == current_user.id,
                StrategyUse.is_active == True,
            )
        )
        .limit(5)
    )
    strategies_result = await db.execute(strategies_query)
    active_strategies = strategies_result.scalars().all()

    # Get market indices
    indices = await market_service.get_market_indices()

    # Get usage limits
    usage_service = get_usage_limits_service(db)
    usage_summary = await usage_service.get_usage_summary(current_user)

    return {
        "user": {
            "id": current_user.id,
            "name": current_user.full_name,
            "email": current_user.email,
            "plan": current_user.subscription_plan.value,
            "is_verified": current_user.is_verified,
        },
        "portfolio": {
            "total_value": float(total_portfolio_value),
            "total_invested": float(total_invested),
            "profit_loss": float(total_profit_loss),
            "profit_loss_pct": total_profit_loss_pct,
            "assets_count": len(assets_data),
            "portfolios_count": portfolios_count,
            "primary_portfolio_id": primary_portfolio.id if primary_portfolio else None,
            "top_assets": assets_data[:5],  # Top 5 by value
        },
        "alerts": {
            "active": active_alerts,
            "triggered_this_week": triggered_alerts,
        },
        "notifications": {
            "unread": unread_notifications,
        },
        "recent_activity": {
            "backtests": [
                {
                    "id": b.id,
                    "strategy": b.strategy_name,
                    "return": float(b.total_return) if b.total_return else None,
                    "date": b.created_at.isoformat(),
                }
                for b in recent_backtests
            ],
            "forecasts": [
                {
                    "id": f.id,
                    "ticker": f.ticker,
                    "predicted_change": float(f.predicted_change_pct),
                    "confidence": float(f.confidence),
                    "date": f.created_at.isoformat(),
                }
                for f in recent_forecasts
            ],
            "active_strategies": [
                {
                    "id": s.id,
                    "strategy_id": s.strategy_id,
                    "initial_value": float(s.initial_value),
                    "current_value": float(s.current_value) if s.current_value else None,
                    "return_pct": float(s.return_pct) if s.return_pct else None,
                }
                for s in active_strategies
            ],
        },
        "market": {
            "indices": [
                {
                    "name": idx.name,
                    "value": float(idx.current_price),
                    "change": float(idx.change),
                    "change_pct": float(idx.change_percent),
                }
                for idx in (indices or [])
            ],
        },
        "usage": usage_summary,
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.get("/quick-stats")
async def get_quick_stats(
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """
    Get quick stats for header/navbar display.

    Lightweight endpoint for frequent polling.
    """
    # Unread notifications
    unread_result = await db.execute(
        select(func.count(Notification.id)).where(
            and_(
                Notification.user_id == current_user.id,
                Notification.is_read == False,
            )
        )
    )
    unread = unread_result.scalar() or 0

    # Active alerts
    alerts_result = await db.execute(
        select(func.count(PriceAlert.id)).where(
            and_(
                PriceAlert.user_id == current_user.id,
                PriceAlert.is_active == True,
            )
        )
    )
    alerts = alerts_result.scalar() or 0

    return {
        "unread_notifications": unread,
        "active_alerts": alerts,
        "plan": current_user.subscription_plan.value,
    }
