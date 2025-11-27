"""Market data background tasks."""
import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import structlog

from src.core.database import get_celery_async_session
from src.models.market import PriceHistory, PortfolioSnapshot
from src.models.portfolio import Portfolio, PortfolioAsset
from src.workers.celery_app import celery_app
from sqlalchemy import select, distinct
from sqlalchemy.orm import selectinload

logger = structlog.get_logger()


def run_async(coro):
    """Helper to run async code in sync Celery tasks.

    Creates a fresh event loop for each task execution and properly
    cleans up async resources to avoid 'Event loop is closed' errors.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        # Clean up all pending tasks
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            # Give tasks a chance to clean up
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass

        # Close any remaining transports
        loop.run_until_complete(loop.shutdown_asyncgens())
        try:
            loop.run_until_complete(loop.shutdown_default_executor())
        except Exception:
            pass

        loop.close()


async def _get_tracked_tickers() -> list[str]:
    """Get all unique tickers from active portfolios."""
    async with get_celery_async_session()() as session:
        result = await session.execute(
            select(distinct(PortfolioAsset.ticker))
        )
        return [row[0] for row in result.fetchall()]


async def _update_prices_async():
    """Update market prices asynchronously."""
    from src.services.market import market_service

    # Get all tracked tickers
    tickers = await _get_tracked_tickers()

    if not tickers:
        logger.info("No tickers to update")
        return {"updated": 0, "tickers": []}

    logger.info("Fetching prices for tickers", count=len(tickers))

    # Fetch quotes in batches of 10
    batch_size = 10
    all_quotes = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            quotes = await market_service.get_quotes(batch)
            all_quotes.update(quotes)
        except Exception as e:
            logger.warning("Failed to fetch batch", batch=batch, error=str(e))

    # Store prices in database
    async with get_celery_async_session()() as session:
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        stored_count = 0

        for ticker, quote in all_quotes.items():
            try:
                # Check if already exists
                existing = await session.execute(
                    select(PriceHistory).where(
                        PriceHistory.ticker == ticker,
                        PriceHistory.date == today,
                    )
                )
                if existing.scalar_one_or_none():
                    continue

                # Create new price record
                price_record = PriceHistory(
                    ticker=ticker,
                    date=today,
                    open_price=quote.open_price if hasattr(quote, 'open_price') else None,
                    high_price=quote.high_price if hasattr(quote, 'high_price') else None,
                    low_price=quote.low_price if hasattr(quote, 'low_price') else None,
                    close_price=quote.current_price,
                    volume=quote.volume if hasattr(quote, 'volume') else None,
                )
                session.add(price_record)
                stored_count += 1
            except Exception as e:
                logger.warning("Failed to store price", ticker=ticker, error=str(e))

        await session.commit()

    logger.info("Prices stored in database", count=stored_count)

    return {
        "updated": len(all_quotes),
        "stored": stored_count,
        "tickers": list(all_quotes.keys()),
    }


async def _update_portfolio_snapshots_async():
    """Update portfolio snapshots for all active portfolios."""
    from src.services.market import market_service

    async with get_celery_async_session()() as session:
        # Get all portfolios with assets
        result = await session.execute(
            select(Portfolio).options(selectinload(Portfolio.assets))
        )
        portfolios = result.scalars().all()

        if not portfolios:
            return {"updated": 0}

        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        updated_count = 0

        for portfolio in portfolios:
            if not portfolio.assets:
                continue

            try:
                # Get all tickers for this portfolio
                tickers = [asset.ticker for asset in portfolio.assets]
                quotes = await market_service.get_quotes(tickers)

                # Calculate total values
                total_invested = Decimal("0")
                total_current = Decimal("0")

                for asset in portfolio.assets:
                    invested = asset.quantity * asset.average_price
                    total_invested += invested

                    quote = quotes.get(asset.ticker.upper())
                    if quote:
                        total_current += asset.quantity * quote.current_price
                    else:
                        total_current += invested

                # Get previous snapshot for daily return calculation
                prev_result = await session.execute(
                    select(PortfolioSnapshot)
                    .where(
                        PortfolioSnapshot.portfolio_id == portfolio.id,
                        PortfolioSnapshot.date < today,
                    )
                    .order_by(PortfolioSnapshot.date.desc())
                    .limit(1)
                )
                prev_snapshot = prev_result.scalar_one_or_none()

                daily_return = None
                if prev_snapshot and prev_snapshot.total_value > 0:
                    daily_return = (total_current - prev_snapshot.total_value) / prev_snapshot.total_value

                # Check if today's snapshot exists
                existing = await session.execute(
                    select(PortfolioSnapshot).where(
                        PortfolioSnapshot.portfolio_id == portfolio.id,
                        PortfolioSnapshot.date == today,
                    )
                )
                snapshot = existing.scalar_one_or_none()

                if snapshot:
                    snapshot.total_value = total_current
                    snapshot.total_invested = total_invested
                    snapshot.daily_return = daily_return
                else:
                    snapshot = PortfolioSnapshot(
                        portfolio_id=portfolio.id,
                        date=today,
                        total_value=total_current,
                        total_invested=total_invested,
                        daily_return=daily_return,
                    )
                    session.add(snapshot)

                updated_count += 1

            except Exception as e:
                logger.warning(
                    "Failed to update portfolio snapshot",
                    portfolio_id=portfolio.id,
                    error=str(e),
                )

        await session.commit()

    return {"updated": updated_count}


async def _fetch_historical_data_async(ticker: str, period: str = "1y"):
    """Fetch and store historical data for a ticker."""
    from src.services.market import market_service

    try:
        history = await market_service.get_historical_data(ticker, period)

        if not history:
            return {"ticker": ticker, "stored": 0}

        async with get_celery_async_session()() as session:
            stored_count = 0

            for data_point in history:
                try:
                    # Check if exists
                    existing = await session.execute(
                        select(PriceHistory).where(
                            PriceHistory.ticker == ticker.upper(),
                            PriceHistory.date == data_point.get("date"),
                        )
                    )
                    if existing.scalar_one_or_none():
                        continue

                    price_record = PriceHistory(
                        ticker=ticker.upper(),
                        date=data_point.get("date"),
                        open_price=data_point.get("open"),
                        high_price=data_point.get("high"),
                        low_price=data_point.get("low"),
                        close_price=data_point.get("close"),
                        volume=data_point.get("volume"),
                    )
                    session.add(price_record)
                    stored_count += 1
                except Exception as e:
                    logger.warning("Failed to store historical data point", error=str(e))

            await session.commit()

        return {"ticker": ticker, "stored": stored_count, "period": period}

    except Exception as e:
        logger.error("Failed to fetch historical data", ticker=ticker, error=str(e))
        return {"ticker": ticker, "stored": 0, "error": str(e)}


@celery_app.task(bind=True, max_retries=3)
def update_market_prices(self):
    """Update cached market prices for tracked assets."""
    try:
        logger.info("Starting market price update...")
        result = run_async(_update_prices_async())
        logger.info("Market prices updated successfully", **result)
        return result
    except Exception as exc:
        logger.error("Failed to update market prices", error=str(exc))
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, max_retries=3)
def update_portfolio_snapshots(self):
    """Update daily portfolio snapshots for all portfolios."""
    try:
        logger.info("Starting portfolio snapshots update...")
        result = run_async(_update_portfolio_snapshots_async())
        logger.info("Portfolio snapshots updated successfully", **result)
        return result
    except Exception as exc:
        logger.error("Failed to update portfolio snapshots", error=str(exc))
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, max_retries=3)
def update_market_indices(self):
    """Update major market indices (IBOV, IFIX, etc.)."""
    try:
        logger.info("Updating market indices...")

        indices = ["^BVSP", "IFIX.SA", "^GSPC", "^DJI"]
        result = run_async(_update_prices_for_tickers(indices))

        logger.info("Market indices updated successfully", **result)
        return result
    except Exception as exc:
        logger.error("Failed to update market indices", error=str(exc))
        raise self.retry(exc=exc, countdown=30)


async def _update_prices_for_tickers(tickers: list[str]):
    """Update prices for specific tickers."""
    from src.services.market import market_service

    quotes = await market_service.get_quotes(tickers)

    async with get_celery_async_session()() as session:
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        stored_count = 0

        for ticker, quote in quotes.items():
            try:
                existing = await session.execute(
                    select(PriceHistory).where(
                        PriceHistory.ticker == ticker,
                        PriceHistory.date == today,
                    )
                )
                if existing.scalar_one_or_none():
                    continue

                price_record = PriceHistory(
                    ticker=ticker,
                    date=today,
                    close_price=quote.current_price,
                )
                session.add(price_record)
                stored_count += 1
            except Exception as e:
                logger.warning("Failed to store index price", ticker=ticker, error=str(e))

        await session.commit()

    return {"updated": len(quotes), "stored": stored_count, "tickers": list(quotes.keys())}


@celery_app.task
def fetch_asset_history(ticker: str, period: str = "1y"):
    """Fetch historical data for a specific asset."""
    logger.info("Fetching asset history", ticker=ticker, period=period)
    result = run_async(_fetch_historical_data_async(ticker, period))
    return result


@celery_app.task
def analyze_market_sentiment():
    """Analyze market sentiment from news and social media."""
    logger.info("Analyzing market sentiment...")
    # TODO: Implement sentiment analysis with news API
    # For MVP, this is a placeholder
    return {"status": "not_implemented", "message": "Sentiment analysis coming soon"}
