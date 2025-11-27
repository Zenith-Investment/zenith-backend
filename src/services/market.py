"""Market data service with caching and provider aggregation."""
import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal

import redis.asyncio as redis
import structlog

from src.core.config import settings
from src.integrations.market_data.base import (
    AssetFundamentals,
    AssetQuote,
    AssetSearchResult,
    AssetType,
    PriceHistory,
)
from src.integrations.market_data.yahoo_finance import YahooFinanceProvider
from src.integrations.market_data.brapi import BRAPIProvider, POPULAR_BR_STOCKS, POPULAR_FIIS

logger = structlog.get_logger()


class MarketDataService:
    """Aggregated market data service with caching."""

    CACHE_TTL_QUOTE = 60  # 1 minute for quotes
    CACHE_TTL_HISTORY = 3600  # 1 hour for historical data
    CACHE_TTL_FUNDAMENTALS = 86400  # 24 hours for fundamentals

    def __init__(self):
        self.yahoo = YahooFinanceProvider()
        self.brapi = BRAPIProvider()
        self._redis: redis.Redis | None = None
        self._redis_loop_id: int | None = None

    async def _get_redis(self) -> redis.Redis:
        """Get Redis connection.

        Creates a new connection if the event loop has changed (e.g., in Celery workers).
        """
        try:
            current_loop = asyncio.get_running_loop()
            current_loop_id = id(current_loop)
        except RuntimeError:
            current_loop_id = None

        # Create new connection if loop changed or no connection exists
        if self._redis is None or self._redis_loop_id != current_loop_id:
            # Close old connection if exists
            if self._redis is not None:
                try:
                    await self._redis.close()
                except Exception:
                    pass
            self._redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
            self._redis_loop_id = current_loop_id

        return self._redis

    async def _get_cache(self, key: str) -> dict | None:
        """Get value from cache."""
        try:
            r = await self._get_redis()
            data = await r.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning("Cache read error", key=key, error=str(e))
        return None

    async def _set_cache(self, key: str, value: dict, ttl: int) -> None:
        """Set value in cache."""
        try:
            r = await self._get_redis()
            await r.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.warning("Cache write error", key=key, error=str(e))

    def _is_brazilian_ticker(self, ticker: str) -> bool:
        """Check if ticker is likely Brazilian."""
        ticker = ticker.upper()
        # Brazilian tickers are typically 4-6 chars ending with a number
        if len(ticker) >= 4 and ticker[-1].isdigit():
            return True
        # Known suffixes
        if any(ticker.endswith(s) for s in ["3", "4", "5", "6", "11", "34", "35"]):
            return True
        return False

    async def get_quote(self, ticker: str) -> AssetQuote | None:
        """Get real-time quote with caching."""
        cache_key = f"quote:{ticker.upper()}"

        # Try cache first
        cached = await self._get_cache(cache_key)
        if cached:
            logger.debug("Quote cache hit", ticker=ticker)
            return self._dict_to_quote(cached)

        # Fetch from providers
        quote = None
        if self._is_brazilian_ticker(ticker):
            # Try BRAPI first for Brazilian stocks
            quote = await self.brapi.get_quote(ticker)

        if not quote:
            # Fallback to Yahoo Finance
            quote = await self.yahoo.get_quote(ticker)

        if quote:
            # Cache the result
            await self._set_cache(cache_key, self._quote_to_dict(quote), self.CACHE_TTL_QUOTE)

        return quote

    async def get_quotes(self, tickers: list[str]) -> dict[str, AssetQuote]:
        """Get multiple quotes with caching."""
        results = {}
        uncached_tickers = []

        # Check cache for each ticker
        for ticker in tickers:
            cache_key = f"quote:{ticker.upper()}"
            cached = await self._get_cache(cache_key)
            if cached:
                results[ticker.upper()] = self._dict_to_quote(cached)
            else:
                uncached_tickers.append(ticker)

        if not uncached_tickers:
            return results

        # Separate Brazilian and international tickers
        br_tickers = [t for t in uncached_tickers if self._is_brazilian_ticker(t)]
        other_tickers = [t for t in uncached_tickers if not self._is_brazilian_ticker(t)]

        # Fetch Brazilian tickers from BRAPI
        if br_tickers:
            br_quotes = await self.brapi.get_quotes(br_tickers)
            for ticker, quote in br_quotes.items():
                results[ticker] = quote
                await self._set_cache(f"quote:{ticker}", self._quote_to_dict(quote), self.CACHE_TTL_QUOTE)

        # Fetch remaining from Yahoo
        remaining = [t for t in br_tickers if t.upper() not in results] + other_tickers
        if remaining:
            yahoo_quotes = await self.yahoo.get_quotes(remaining)
            for ticker, quote in yahoo_quotes.items():
                results[ticker] = quote
                await self._set_cache(f"quote:{ticker}", self._quote_to_dict(quote), self.CACHE_TTL_QUOTE)

        return results

    async def get_history(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> list[PriceHistory]:
        """Get historical data with caching."""
        cache_key = f"history:{ticker.upper()}:{period}:{interval}"

        # Try cache first
        cached = await self._get_cache(cache_key)
        if cached:
            logger.debug("History cache hit", ticker=ticker)
            return [self._dict_to_history(h) for h in cached]

        # Yahoo Finance is better for historical data
        history = await self.yahoo.get_history(ticker, period, interval)

        if history:
            await self._set_cache(
                cache_key,
                [self._history_to_dict(h) for h in history],
                self.CACHE_TTL_HISTORY,
            )

        return history

    async def search(self, query: str, limit: int = 10) -> list[AssetSearchResult]:
        """Search for assets."""
        query_upper = query.upper()
        results = []

        # Search in popular Brazilian stocks first
        for ticker, name in POPULAR_BR_STOCKS + POPULAR_FIIS:
            if query_upper in ticker or query_upper in name.upper():
                results.append(
                    AssetSearchResult(
                        ticker=ticker,
                        name=name,
                        asset_type=AssetType.FII if "11" in ticker else AssetType.STOCK,
                        exchange="B3",
                    )
                )
                if len(results) >= limit:
                    break

        # If not enough results, search BRAPI
        if len(results) < limit:
            brapi_results = await self.brapi.search(query, limit - len(results))
            existing_tickers = {r.ticker for r in results}
            for r in brapi_results:
                if r.ticker not in existing_tickers:
                    results.append(r)

        return results[:limit]

    async def get_fundamentals(self, ticker: str) -> AssetFundamentals | None:
        """Get fundamental data with caching."""
        cache_key = f"fundamentals:{ticker.upper()}"

        # Try cache first
        cached = await self._get_cache(cache_key)
        if cached:
            return self._dict_to_fundamentals(cached)

        # Use Yahoo Finance for fundamentals
        fundamentals = await self.yahoo.get_fundamentals(ticker)

        if fundamentals:
            await self._set_cache(
                cache_key,
                self._fundamentals_to_dict(fundamentals),
                self.CACHE_TTL_FUNDAMENTALS,
            )

        return fundamentals

    async def get_current_price(self, ticker: str) -> Decimal | None:
        """Get current price for a ticker."""
        quote = await self.get_quote(ticker)
        return quote.current_price if quote else None

    async def get_52_week_range(self, ticker: str) -> dict | None:
        """Get 52-week high and low for a ticker."""
        cache_key = f"52week:{ticker.upper()}"

        # Try cache first
        cached = await self._get_cache(cache_key)
        if cached:
            return cached

        # Fetch 1 year of historical data
        history = await self.get_history(ticker, period="1y", interval="1d")

        if not history or len(history) < 5:
            return None

        # Calculate 52-week high and low
        highs = [float(h.high) for h in history]
        lows = [float(h.low) for h in history]

        result = {
            "week_52_high": max(highs),
            "week_52_low": min(lows),
        }

        # Cache for 1 hour
        await self._set_cache(cache_key, result, 3600)

        return result

    async def get_market_news(self, ticker: str | None = None, limit: int = 10) -> list[dict]:
        """Get market news, optionally filtered by ticker."""
        cache_key = f"news:{ticker or 'general'}:{limit}"

        # Try cache first (5 minutes)
        cached = await self._get_cache(cache_key)
        if cached:
            return cached

        # For MVP, return simulated news based on market data
        # In production, this would integrate with a news API
        news = []

        if ticker:
            quote = await self.get_quote(ticker)
            if quote:
                change_text = "alta" if quote.change_percent > 0 else "queda"
                news.append({
                    "id": f"news_{ticker}_1",
                    "title": f"{quote.name} ({ticker}) registra {change_text} de {abs(quote.change_percent):.2f}%",
                    "summary": f"O ativo {ticker} fechou em R$ {quote.current_price:.2f}.",
                    "url": f"https://investai.com.br/noticias/{ticker.lower()}",
                    "source": "InvestAI",
                    "published_at": datetime.now().isoformat(),
                    "related_tickers": [ticker.upper()],
                    "sentiment": "positive" if quote.change_percent > 0 else "negative",
                })
        else:
            # General market news
            indices = await self.get_market_indices()
            for idx in indices[:3]:
                change_text = "alta" if idx.change_percent > 0 else "queda"
                news.append({
                    "id": f"news_{idx.ticker}",
                    "title": f"{idx.name} opera em {change_text} de {abs(idx.change_percent):.2f}%",
                    "summary": f"Índice marca {idx.current_price:.0f} pontos.",
                    "url": f"https://investai.com.br/mercado",
                    "source": "InvestAI",
                    "published_at": datetime.now().isoformat(),
                    "related_tickers": [idx.ticker],
                    "sentiment": "positive" if idx.change_percent > 0 else "negative",
                })

        # Cache for 5 minutes
        await self._set_cache(cache_key, news, 300)

        return news[:limit]

    async def get_market_indices(self) -> list[AssetQuote]:
        """Get major market indices."""
        indices = [
            "^BVSP",  # Ibovespa
            "^IFIX",  # IFIX (FIIs)
            "^GSPC",  # S&P 500
            "^DJI",   # Dow Jones
        ]

        quotes = await self.yahoo.get_quotes(indices)
        return list(quotes.values())

    # Serialization helpers
    def _quote_to_dict(self, quote: AssetQuote) -> dict:
        return {
            "ticker": quote.ticker,
            "name": quote.name,
            "asset_type": quote.asset_type.value,
            "current_price": str(quote.current_price),
            "change": str(quote.change),
            "change_percent": quote.change_percent,
            "open_price": str(quote.open_price) if quote.open_price else None,
            "high": str(quote.high) if quote.high else None,
            "low": str(quote.low) if quote.low else None,
            "previous_close": str(quote.previous_close) if quote.previous_close else None,
            "volume": quote.volume,
            "market_cap": str(quote.market_cap) if quote.market_cap else None,
            "currency": quote.currency,
            "exchange": quote.exchange,
            "last_updated": quote.last_updated.isoformat() if quote.last_updated else None,
        }

    def _dict_to_quote(self, data: dict) -> AssetQuote:
        return AssetQuote(
            ticker=data["ticker"],
            name=data["name"],
            asset_type=AssetType(data["asset_type"]),
            current_price=Decimal(data["current_price"]),
            change=Decimal(data["change"]),
            change_percent=data["change_percent"],
            open_price=Decimal(data["open_price"]) if data.get("open_price") else None,
            high=Decimal(data["high"]) if data.get("high") else None,
            low=Decimal(data["low"]) if data.get("low") else None,
            previous_close=Decimal(data["previous_close"]) if data.get("previous_close") else None,
            volume=data.get("volume"),
            market_cap=Decimal(data["market_cap"]) if data.get("market_cap") else None,
            currency=data.get("currency", "BRL"),
            exchange=data.get("exchange"),
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else None,
        )

    def _history_to_dict(self, h: PriceHistory) -> dict:
        return {
            "date": h.date.isoformat(),
            "open": str(h.open),
            "high": str(h.high),
            "low": str(h.low),
            "close": str(h.close),
            "volume": h.volume,
        }

    def _dict_to_history(self, data: dict) -> PriceHistory:
        return PriceHistory(
            date=datetime.fromisoformat(data["date"]),
            open=Decimal(data["open"]),
            high=Decimal(data["high"]),
            low=Decimal(data["low"]),
            close=Decimal(data["close"]),
            volume=data["volume"],
        )

    def _fundamentals_to_dict(self, f: AssetFundamentals) -> dict:
        return {
            "ticker": f.ticker,
            "pe_ratio": f.pe_ratio,
            "pb_ratio": f.pb_ratio,
            "dividend_yield": f.dividend_yield,
            "roe": f.roe,
            "roa": f.roa,
            "debt_to_equity": f.debt_to_equity,
            "sector": f.sector,
            "industry": f.industry,
            "description": f.description,
        }

    def _dict_to_fundamentals(self, data: dict) -> AssetFundamentals:
        return AssetFundamentals(
            ticker=data["ticker"],
            pe_ratio=data.get("pe_ratio"),
            pb_ratio=data.get("pb_ratio"),
            dividend_yield=data.get("dividend_yield"),
            roe=data.get("roe"),
            roa=data.get("roa"),
            debt_to_equity=data.get("debt_to_equity"),
            sector=data.get("sector"),
            industry=data.get("industry"),
            description=data.get("description"),
        )


# Singleton instance
market_service = MarketDataService()
