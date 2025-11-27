"""Yahoo Finance market data provider."""
import asyncio
from datetime import datetime
from decimal import Decimal

import yfinance as yf
import structlog

from src.integrations.market_data.base import (
    AssetFundamentals,
    AssetQuote,
    AssetSearchResult,
    AssetType,
    MarketDataProvider,
    PriceHistory,
)

logger = structlog.get_logger()


class YahooFinanceProvider(MarketDataProvider):
    """Market data provider using Yahoo Finance."""

    # Mapping of Brazilian tickers to Yahoo format
    BR_SUFFIX = ".SA"

    def _format_ticker(self, ticker: str) -> str:
        """Format ticker for Yahoo Finance."""
        ticker = ticker.upper().strip()
        # Add .SA suffix for Brazilian stocks if not present
        if not ticker.endswith(".SA") and not ticker.startswith("^"):
            # Check if it's likely a Brazilian ticker (4-6 chars, ends with number)
            if len(ticker) >= 4 and ticker[-1].isdigit():
                return f"{ticker}.SA"
        return ticker

    def _determine_asset_type(self, ticker: str, info: dict) -> AssetType:
        """Determine asset type from ticker and info."""
        ticker_upper = ticker.upper()
        quote_type = info.get("quoteType", "").upper()

        if quote_type == "ETF":
            return AssetType.ETF
        if "11" in ticker_upper:  # FIIs usually end with 11
            return AssetType.FII
        if ticker_upper.endswith("34") or ticker_upper.endswith("35"):
            return AssetType.BDR
        if ticker_upper.startswith("^"):
            return AssetType.INDEX
        return AssetType.STOCK

    async def get_quote(self, ticker: str) -> AssetQuote | None:
        """Get real-time quote for a ticker."""
        try:
            formatted_ticker = self._format_ticker(ticker)

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            stock = await loop.run_in_executor(None, yf.Ticker, formatted_ticker)
            info = await loop.run_in_executor(None, lambda: stock.info)

            if not info or "regularMarketPrice" not in info:
                logger.warning("No data found for ticker", ticker=ticker)
                return None

            current_price = Decimal(str(info.get("regularMarketPrice", 0)))
            previous_close = Decimal(str(info.get("previousClose", 0)))
            change = current_price - previous_close if previous_close else Decimal(0)
            change_percent = float(info.get("regularMarketChangePercent", 0))

            return AssetQuote(
                ticker=ticker.upper(),
                name=info.get("shortName") or info.get("longName") or ticker,
                asset_type=self._determine_asset_type(ticker, info),
                current_price=current_price,
                change=change,
                change_percent=change_percent,
                open_price=Decimal(str(info.get("regularMarketOpen", 0))) if info.get("regularMarketOpen") else None,
                high=Decimal(str(info.get("regularMarketDayHigh", 0))) if info.get("regularMarketDayHigh") else None,
                low=Decimal(str(info.get("regularMarketDayLow", 0))) if info.get("regularMarketDayLow") else None,
                previous_close=previous_close if previous_close else None,
                volume=info.get("regularMarketVolume"),
                market_cap=Decimal(str(info.get("marketCap", 0))) if info.get("marketCap") else None,
                currency=info.get("currency", "BRL"),
                exchange=info.get("exchange"),
                last_updated=datetime.now(),
            )

        except Exception as e:
            logger.error("Error fetching quote", ticker=ticker, error=str(e))
            return None

    async def get_quotes(self, tickers: list[str]) -> dict[str, AssetQuote]:
        """Get real-time quotes for multiple tickers."""
        results = {}
        tasks = [self.get_quote(ticker) for ticker in tickers]
        quotes = await asyncio.gather(*tasks, return_exceptions=True)

        for ticker, quote in zip(tickers, quotes):
            if isinstance(quote, AssetQuote):
                results[ticker.upper()] = quote

        return results

    async def get_history(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> list[PriceHistory]:
        """Get historical price data."""
        try:
            formatted_ticker = self._format_ticker(ticker)

            loop = asyncio.get_event_loop()
            stock = await loop.run_in_executor(None, yf.Ticker, formatted_ticker)

            def fetch_history():
                return stock.history(period=period, interval=interval)

            hist = await loop.run_in_executor(None, fetch_history)

            if hist.empty:
                return []

            history = []
            for date, row in hist.iterrows():
                history.append(
                    PriceHistory(
                        date=date.to_pydatetime(),
                        open=Decimal(str(row["Open"])),
                        high=Decimal(str(row["High"])),
                        low=Decimal(str(row["Low"])),
                        close=Decimal(str(row["Close"])),
                        volume=int(row["Volume"]),
                    )
                )

            return history

        except Exception as e:
            logger.error("Error fetching history", ticker=ticker, error=str(e))
            return []

    async def search(self, query: str, limit: int = 10) -> list[AssetSearchResult]:
        """Search for assets - limited support in yfinance."""
        # yfinance doesn't have native search, return empty
        # We'll use BRAPI for Brazilian stocks search
        return []

    async def get_fundamentals(self, ticker: str) -> AssetFundamentals | None:
        """Get fundamental data for an asset."""
        try:
            formatted_ticker = self._format_ticker(ticker)

            loop = asyncio.get_event_loop()
            stock = await loop.run_in_executor(None, yf.Ticker, formatted_ticker)
            info = await loop.run_in_executor(None, lambda: stock.info)

            if not info:
                return None

            return AssetFundamentals(
                ticker=ticker.upper(),
                pe_ratio=info.get("trailingPE"),
                pb_ratio=info.get("priceToBook"),
                dividend_yield=info.get("dividendYield"),
                roe=info.get("returnOnEquity"),
                roa=info.get("returnOnAssets"),
                debt_to_equity=info.get("debtToEquity"),
                current_ratio=info.get("currentRatio"),
                profit_margin=info.get("profitMargins"),
                revenue=Decimal(str(info.get("totalRevenue", 0))) if info.get("totalRevenue") else None,
                net_income=Decimal(str(info.get("netIncomeToCommon", 0))) if info.get("netIncomeToCommon") else None,
                eps=info.get("trailingEps"),
                sector=info.get("sector"),
                industry=info.get("industry"),
                description=info.get("longBusinessSummary"),
                website=info.get("website"),
                employees=info.get("fullTimeEmployees"),
            )

        except Exception as e:
            logger.error("Error fetching fundamentals", ticker=ticker, error=str(e))
            return None
