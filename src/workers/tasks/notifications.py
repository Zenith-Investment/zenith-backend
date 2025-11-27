"""Notification background tasks."""
import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import structlog

from src.core.database import get_celery_async_session
from src.models.alert import PriceAlert
from src.models.portfolio import Portfolio, PortfolioAsset
from src.models.user import User
from src.services.email import (
    email_service,
    get_daily_report_email_html,
    get_price_alert_email_html,
    get_welcome_email_html,
    get_rebalance_suggestion_email_html,
)
from src.workers.celery_app import celery_app
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload

logger = structlog.get_logger()


def run_async(coro):
    """Helper to run async code in sync Celery tasks."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _check_price_alerts_async():
    """Check all active price alerts and trigger notifications."""
    from src.services.market import market_service

    async with get_celery_async_session()() as session:
        # Get all active, non-triggered alerts
        result = await session.execute(
            select(PriceAlert)
            .where(
                and_(
                    PriceAlert.is_active == True,
                    PriceAlert.is_triggered == False,
                )
            )
            .options(selectinload(PriceAlert.user))
        )
        alerts = result.scalars().all()

        if not alerts:
            return {"checked": 0, "triggered": 0}

        # Group alerts by ticker
        alerts_by_ticker: dict[str, list[PriceAlert]] = {}
        for alert in alerts:
            ticker = alert.ticker.upper()
            if ticker not in alerts_by_ticker:
                alerts_by_ticker[ticker] = []
            alerts_by_ticker[ticker].append(alert)

        # Fetch current prices
        tickers = list(alerts_by_ticker.keys())
        quotes = await market_service.get_quotes(tickers)

        triggered_count = 0

        for ticker, ticker_alerts in alerts_by_ticker.items():
            quote = quotes.get(ticker)
            if not quote:
                continue

            current_price = float(quote.current_price)

            for alert in ticker_alerts:
                target_price = float(alert.target_price)
                should_trigger = False

                if alert.condition.value == "above" and current_price >= target_price:
                    should_trigger = True
                elif alert.condition.value == "below" and current_price <= target_price:
                    should_trigger = True

                if should_trigger:
                    # Mark as triggered
                    alert.is_triggered = True
                    alert.triggered_at = datetime.now(timezone.utc)
                    alert.triggered_price = Decimal(str(current_price))

                    # Send notification
                    send_price_alert.delay(
                        user_id=alert.user_id,
                        user_email=alert.user.email,
                        user_name=alert.user.full_name,
                        ticker=ticker,
                        condition=alert.condition.value,
                        target_price=target_price,
                        current_price=current_price,
                    )

                    triggered_count += 1
                    logger.info(
                        "Price alert triggered",
                        alert_id=alert.id,
                        ticker=ticker,
                        condition=alert.condition.value,
                        target=target_price,
                        current=current_price,
                    )

        await session.commit()

    return {"checked": len(alerts), "triggered": triggered_count}


async def _send_daily_reports_async():
    """Send daily portfolio reports to all users."""
    from src.services.market import market_service

    async with get_celery_async_session()() as session:
        # Get all users with portfolios
        result = await session.execute(
            select(User)
            .where(User.is_active == True)
            .options(
                selectinload(User.portfolio).selectinload(Portfolio.assets)
            )
        )
        users = result.scalars().all()

        sent_count = 0

        for user in users:
            if not user.portfolio or not user.portfolio.assets:
                continue

            try:
                # Calculate portfolio values
                tickers = [asset.ticker for asset in user.portfolio.assets]
                quotes = await market_service.get_quotes(tickers)

                total_value = Decimal("0")
                assets_data = []

                for asset in user.portfolio.assets:
                    quote = quotes.get(asset.ticker.upper())
                    if quote:
                        current_value = asset.quantity * quote.current_price
                        total_value += current_value

                        # Calculate daily change (assuming quote has previous_close)
                        prev_close = getattr(quote, 'previous_close', quote.current_price)
                        if prev_close and prev_close > 0:
                            change_pct = ((quote.current_price - prev_close) / prev_close) * 100
                        else:
                            change_pct = 0

                        assets_data.append({
                            "ticker": asset.ticker,
                            "value": float(current_value),
                            "change_pct": float(change_pct),
                        })

                # Sort for top gainers/losers
                sorted_assets = sorted(assets_data, key=lambda x: x["change_pct"], reverse=True)
                top_gainers = [a for a in sorted_assets if a["change_pct"] > 0][:3]
                top_losers = [a for a in sorted_assets if a["change_pct"] < 0][-3:]

                # Calculate total daily change (simplified)
                daily_change = sum(a["value"] * a["change_pct"] / 100 for a in assets_data)
                daily_change_pct = (daily_change / float(total_value) * 100) if total_value > 0 else 0

                # Send email
                html_content = get_daily_report_email_html(
                    name=user.full_name,
                    total_value=float(total_value),
                    daily_change=daily_change,
                    daily_change_pct=daily_change_pct,
                    top_gainers=top_gainers,
                    top_losers=top_losers,
                )

                if email_service.send_email(
                    to_email=user.email,
                    subject="📊 Resumo Diário do Seu Portfólio - InvestAI",
                    html_content=html_content,
                ):
                    sent_count += 1

            except Exception as e:
                logger.warning(
                    "Failed to send daily report",
                    user_id=user.id,
                    error=str(e),
                )

    return {"sent": sent_count, "total_users": len(users)}


@celery_app.task
def check_price_alerts():
    """Check all price alerts and send notifications for triggered ones."""
    logger.info("Checking price alerts...")
    result = run_async(_check_price_alerts_async())
    logger.info("Price alerts checked", **result)
    return result


@celery_app.task
def send_daily_reports():
    """Send daily portfolio reports to users."""
    logger.info("Sending daily reports...")
    result = run_async(_send_daily_reports_async())
    logger.info("Daily reports sent", **result)
    return result


@celery_app.task
def send_price_alert(
    user_id: int,
    user_email: str,
    user_name: str,
    ticker: str,
    condition: str,
    target_price: float,
    current_price: float,
):
    """Send price alert notification to user."""
    logger.info(
        "Sending price alert notification",
        user_id=user_id,
        ticker=ticker,
        condition=condition,
        current_price=current_price,
    )

    html_content = get_price_alert_email_html(
        name=user_name,
        ticker=ticker,
        condition=condition,
        target_price=target_price,
        current_price=current_price,
    )

    condition_text = "acima de" if condition == "above" else "abaixo de"
    success = email_service.send_email(
        to_email=user_email,
        subject=f"🔔 Alerta: {ticker} está {condition_text} R$ {target_price:.2f}",
        html_content=html_content,
    )

    return {"sent": success, "user_id": user_id, "ticker": ticker}


async def _send_rebalance_suggestion_async(user_id: int, portfolio_id: int):
    """Send rebalancing suggestion asynchronously."""
    from src.services.market import market_service
    from src.models.profile import InvestorProfile

    async with get_celery_async_session()() as session:
        # Get user with portfolio and profile
        result = await session.execute(
            select(User)
            .where(User.id == user_id)
            .options(
                selectinload(User.portfolio).selectinload(Portfolio.assets),
                selectinload(User.investor_profile),
            )
        )
        user = result.scalar_one_or_none()

        if not user or not user.portfolio or not user.portfolio.assets:
            return {"sent": False, "reason": "no_portfolio"}

        portfolio = user.portfolio

        # Get target allocation from investor profile
        target_allocation = {}
        if user.investor_profile and user.investor_profile.suggested_allocation:
            target_allocation = user.investor_profile.suggested_allocation
        else:
            # Default allocation for moderate investor
            target_allocation = {
                "stocks": 40,
                "fiis": 20,
                "fixed_income": 30,
                "international": 10,
            }

        # Get current quotes
        tickers = [asset.ticker for asset in portfolio.assets]
        quotes = await market_service.get_quotes(tickers)

        # Calculate current values and allocations
        total_value = Decimal("0")
        assets_by_type = {"stocks": [], "fiis": [], "fixed_income": [], "international": []}

        for asset in portfolio.assets:
            quote = quotes.get(asset.ticker.upper())
            if quote:
                current_value = float(asset.quantity * quote.current_price)
            else:
                current_value = float(asset.quantity * asset.average_price)

            total_value += Decimal(str(current_value))

            # Classify asset
            if asset.ticker.endswith("11"):
                assets_by_type["fiis"].append({"ticker": asset.ticker, "value": current_value})
            elif asset.ticker.endswith("34") or asset.ticker.endswith("35"):
                assets_by_type["international"].append({"ticker": asset.ticker, "value": current_value})
            else:
                assets_by_type["stocks"].append({"ticker": asset.ticker, "value": current_value})

        if total_value == 0:
            return {"sent": False, "reason": "zero_value"}

        # Calculate current allocation percentages
        current_allocation = {}
        for asset_type, assets in assets_by_type.items():
            type_value = sum(a["value"] for a in assets)
            current_allocation[asset_type] = (type_value / float(total_value)) * 100

        # Calculate suggestions
        suggestions = []
        threshold = 5  # 5% deviation threshold

        for asset_type, target_pct in target_allocation.items():
            current_pct = current_allocation.get(asset_type, 0)
            deviation = target_pct - current_pct

            if abs(deviation) >= threshold:
                amount = abs(deviation / 100) * float(total_value)
                action = "comprar" if deviation > 0 else "vender"

                # Get representative ticker for this type
                if assets_by_type.get(asset_type):
                    ticker = assets_by_type[asset_type][0]["ticker"]
                else:
                    # Default tickers by type
                    default_tickers = {
                        "stocks": "PETR4",
                        "fiis": "HGLG11",
                        "fixed_income": "TESOURO",
                        "international": "IVVB11",
                    }
                    ticker = default_tickers.get(asset_type, asset_type.upper())

                suggestions.append({
                    "ticker": ticker,
                    "action": action,
                    "amount": amount,
                    "current_pct": current_pct,
                    "target_pct": target_pct,
                    "asset_type": asset_type,
                })

        if not suggestions:
            return {"sent": False, "reason": "no_rebalance_needed"}

        # Send email
        html_content = get_rebalance_suggestion_email_html(
            name=user.full_name,
            total_value=float(total_value),
            suggestions=suggestions,
            target_allocation=target_allocation,
            current_allocation=current_allocation,
        )

        success = email_service.send_email(
            to_email=user.email,
            subject="⚖️ Sugestão de Rebalanceamento - InvestAI",
            html_content=html_content,
        )

        return {"sent": success, "user_id": user_id, "suggestions_count": len(suggestions)}


@celery_app.task
def send_rebalance_suggestion(user_id: int, portfolio_id: int):
    """Send portfolio rebalancing suggestion to user."""
    logger.info(
        "Sending rebalance suggestion",
        user_id=user_id,
        portfolio_id=portfolio_id,
    )
    result = run_async(_send_rebalance_suggestion_async(user_id, portfolio_id))
    logger.info("Rebalance suggestion result", **result)
    return result


@celery_app.task
def send_welcome_email(user_id: int, email: str, name: str):
    """Send welcome email to new user."""
    logger.info("Sending welcome email", user_id=user_id, email=email)

    html_content = get_welcome_email_html(name=name)

    success = email_service.send_email(
        to_email=email,
        subject="🎉 Bem-vindo ao InvestAI!",
        html_content=html_content,
    )

    return {"sent": success, "user_id": user_id}
