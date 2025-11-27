"""Market data endpoints."""
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Query, status
import structlog

from src.schemas.market import (
    AssetSearchResponse,
    AssetSearchResult,
    AssetDetailResponse,
    AssetHistoryResponse,
    PriceDataPoint,
    MarketIndicesResponse,
    MarketIndex,
    MarketNewsResponse,
    NewsItem,
)
from src.services.market import market_service

router = APIRouter()
logger = structlog.get_logger()


@router.get("/search", response_model=AssetSearchResponse)
async def search_assets(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50),
) -> AssetSearchResponse:
    """Search for assets by name or ticker."""
    results = await market_service.search(q, limit)

    return AssetSearchResponse(
        query=q,
        results=[
            AssetSearchResult(
                ticker=r.ticker,
                name=r.name,
                asset_type=r.asset_type.value,
                exchange=r.exchange,
            )
            for r in results
        ],
        total=len(results),
    )


@router.get("/assets/{ticker}", response_model=AssetDetailResponse)
async def get_asset_detail(ticker: str) -> AssetDetailResponse:
    """Get detailed information about an asset."""
    quote = await market_service.get_quote(ticker)
    if not quote:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset not found: {ticker}",
        )

    # Try to get fundamentals
    fundamentals = await market_service.get_fundamentals(ticker)

    # Get 52-week range
    week_range = await market_service.get_52_week_range(ticker)

    return AssetDetailResponse(
        ticker=quote.ticker,
        name=quote.name,
        asset_type=quote.asset_type.value,
        exchange=quote.exchange,
        current_price=quote.current_price,
        currency=quote.currency,
        change=quote.change,
        change_percent=quote.change_percent,
        open_price=quote.open_price,
        high=quote.high,
        low=quote.low,
        volume=quote.volume,
        market_cap=quote.market_cap,
        pe_ratio=fundamentals.pe_ratio if fundamentals else None,
        dividend_yield=fundamentals.dividend_yield if fundamentals else None,
        week_52_high=Decimal(str(week_range["week_52_high"])) if week_range else None,
        week_52_low=Decimal(str(week_range["week_52_low"])) if week_range else None,
        description=fundamentals.description if fundamentals else None,
        sector=fundamentals.sector if fundamentals else None,
        industry=fundamentals.industry if fundamentals else None,
        last_updated=quote.last_updated,
    )


@router.get("/assets/{ticker}/history", response_model=AssetHistoryResponse)
async def get_asset_history(
    ticker: str,
    period: str = Query("1y", description="History period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max"),
    interval: str = Query("1d", description="Data interval: 1m, 5m, 15m, 1h, 1d, 1wk, 1mo"),
) -> AssetHistoryResponse:
    """Get price history for an asset."""
    # Validate period
    valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
    if period not in valid_periods:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid period. Valid values: {', '.join(valid_periods)}",
        )

    history = await market_service.get_history(ticker, period, interval)

    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No history found for: {ticker}",
        )

    return AssetHistoryResponse(
        ticker=ticker.upper(),
        period=period,
        data=[
            PriceDataPoint(
                date=h.date,
                open=h.open,
                high=h.high,
                low=h.low,
                close=h.close,
                volume=h.volume,
            )
            for h in history
        ],
    )


@router.get("/indices", response_model=MarketIndicesResponse)
async def get_market_indices() -> MarketIndicesResponse:
    """Get major market indices (IBOV, IFIX, S&P 500, etc.)."""
    quotes = await market_service.get_market_indices()

    # Map index symbols to readable names
    index_names = {
        "^BVSP": "Ibovespa",
        "^IFIX": "IFIX",
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "Nasdaq",
    }

    return MarketIndicesResponse(
        indices=[
            MarketIndex(
                symbol=q.ticker,
                name=index_names.get(q.ticker, q.name),
                value=q.current_price,
                change=q.change,
                change_percent=q.change_percent,
                last_updated=q.last_updated,
            )
            for q in quotes
        ]
    )


@router.get("/quotes")
async def get_multiple_quotes(
    tickers: str = Query(..., description="Comma-separated list of tickers"),
) -> dict:
    """Get quotes for multiple tickers at once."""
    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]

    if not ticker_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one ticker is required",
        )

    if len(ticker_list) > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 20 tickers allowed per request",
        )

    quotes = await market_service.get_quotes(ticker_list)

    return {
        "quotes": {
            ticker: {
                "ticker": q.ticker,
                "name": q.name,
                "current_price": float(q.current_price),
                "change": float(q.change),
                "change_percent": q.change_percent,
                "volume": q.volume,
            }
            for ticker, q in quotes.items()
        },
        "found": len(quotes),
        "requested": len(ticker_list),
    }


@router.get("/popular")
async def get_popular_assets() -> dict:
    """Get popular Brazilian assets for quick access."""
    from src.integrations.market_data.brapi import POPULAR_BR_STOCKS, POPULAR_FIIS

    return {
        "stocks": [{"ticker": t, "name": n} for t, n in POPULAR_BR_STOCKS],
        "fiis": [{"ticker": t, "name": n} for t, n in POPULAR_FIIS],
    }


@router.get("/news", response_model=MarketNewsResponse)
async def get_market_news(
    ticker: str | None = Query(None, description="Filter news by ticker"),
    limit: int = Query(10, ge=1, le=50),
) -> MarketNewsResponse:
    """Get market news, optionally filtered by ticker."""
    from datetime import datetime

    news_data = await market_service.get_market_news(ticker, limit)

    news_items = [
        NewsItem(
            id=item["id"],
            title=item["title"],
            summary=item.get("summary"),
            url=item["url"],
            source=item["source"],
            published_at=datetime.fromisoformat(item["published_at"]) if isinstance(item["published_at"], str) else item["published_at"],
            related_tickers=item.get("related_tickers", []),
            sentiment=item.get("sentiment"),
        )
        for item in news_data
    ]

    return MarketNewsResponse(news=news_items, total=len(news_items))
