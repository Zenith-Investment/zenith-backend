"""BRAPI market data provider for Brazilian stocks."""
import asyncio
from datetime import datetime
from decimal import Decimal

import httpx
import structlog

from src.core.config import settings
from src.integrations.market_data.base import (
    AssetFundamentals,
    AssetQuote,
    AssetSearchResult,
    AssetType,
    MarketDataProvider,
    PriceHistory,
)

logger = structlog.get_logger()


class BRAPIProvider(MarketDataProvider):
    """Market data provider using BRAPI (Brazilian stocks API)."""

    BASE_URL = "https://brapi.dev/api"

    def __init__(self):
        self.token = settings.BRAPI_TOKEN

    def _get_headers(self) -> dict:
        """Get request headers."""
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _determine_asset_type(self, ticker: str) -> AssetType:
        """Determine asset type from ticker."""
        ticker_upper = ticker.upper()
        if "11" in ticker_upper:
            return AssetType.FII
        if ticker_upper.endswith("34") or ticker_upper.endswith("35"):
            return AssetType.BDR
        return AssetType.STOCK

    async def get_quote(self, ticker: str) -> AssetQuote | None:
        """Get real-time quote for a ticker."""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.BASE_URL}/quote/{ticker.upper()}"
                params = {"token": self.token} if self.token else {}

                response = await client.get(url, params=params, headers=self._get_headers())
                response.raise_for_status()
                data = response.json()

                if "results" not in data or not data["results"]:
                    return None

                result = data["results"][0]

                current_price = Decimal(str(result.get("regularMarketPrice", 0)))
                change = Decimal(str(result.get("regularMarketChange", 0)))
                change_percent = float(result.get("regularMarketChangePercent", 0))

                return AssetQuote(
                    ticker=ticker.upper(),
                    name=result.get("shortName") or result.get("longName") or ticker,
                    asset_type=self._determine_asset_type(ticker),
                    current_price=current_price,
                    change=change,
                    change_percent=change_percent,
                    open_price=Decimal(str(result.get("regularMarketOpen", 0))) if result.get("regularMarketOpen") else None,
                    high=Decimal(str(result.get("regularMarketDayHigh", 0))) if result.get("regularMarketDayHigh") else None,
                    low=Decimal(str(result.get("regularMarketDayLow", 0))) if result.get("regularMarketDayLow") else None,
                    previous_close=Decimal(str(result.get("regularMarketPreviousClose", 0))) if result.get("regularMarketPreviousClose") else None,
                    volume=result.get("regularMarketVolume"),
                    market_cap=Decimal(str(result.get("marketCap", 0))) if result.get("marketCap") else None,
                    currency=result.get("currency", "BRL"),
                    exchange="B3",
                    last_updated=datetime.now(),
                )

        except httpx.HTTPStatusError as e:
            logger.error("BRAPI HTTP error", ticker=ticker, status=e.response.status_code)
            return None
        except Exception as e:
            logger.error("Error fetching BRAPI quote", ticker=ticker, error=str(e))
            return None

    async def get_quotes(self, tickers: list[str]) -> dict[str, AssetQuote]:
        """Get real-time quotes for multiple tickers."""
        try:
            async with httpx.AsyncClient() as client:
                tickers_str = ",".join([t.upper() for t in tickers])
                url = f"{self.BASE_URL}/quote/{tickers_str}"
                params = {"token": self.token} if self.token else {}

                response = await client.get(url, params=params, headers=self._get_headers())
                response.raise_for_status()
                data = response.json()

                results = {}
                for result in data.get("results", []):
                    ticker = result.get("symbol", "").upper()
                    if ticker:
                        current_price = Decimal(str(result.get("regularMarketPrice", 0)))
                        change = Decimal(str(result.get("regularMarketChange", 0)))

                        results[ticker] = AssetQuote(
                            ticker=ticker,
                            name=result.get("shortName") or ticker,
                            asset_type=self._determine_asset_type(ticker),
                            current_price=current_price,
                            change=change,
                            change_percent=float(result.get("regularMarketChangePercent", 0)),
                            volume=result.get("regularMarketVolume"),
                            currency="BRL",
                            exchange="B3",
                            last_updated=datetime.now(),
                        )

                return results

        except Exception as e:
            logger.error("Error fetching BRAPI quotes", error=str(e))
            return {}

    async def get_history(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> list[PriceHistory]:
        """Get historical price data - limited in free BRAPI."""
        # BRAPI free tier has limited historical data
        # Fall back to Yahoo Finance for history
        return []

    async def search(self, query: str, limit: int = 10) -> list[AssetSearchResult]:
        """Search for Brazilian assets."""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.BASE_URL}/available"
                params = {"search": query}
                if self.token:
                    params["token"] = self.token

                response = await client.get(url, params=params, headers=self._get_headers())
                response.raise_for_status()
                data = response.json()

                results = []
                for stock in data.get("stocks", [])[:limit]:
                    results.append(
                        AssetSearchResult(
                            ticker=stock,
                            name=stock,  # BRAPI doesn't return names in search
                            asset_type=self._determine_asset_type(stock),
                            exchange="B3",
                        )
                    )

                return results

        except Exception as e:
            logger.error("Error searching BRAPI", query=query, error=str(e))
            return []

    async def get_fundamentals(self, ticker: str) -> AssetFundamentals | None:
        """Get fundamental data - limited in BRAPI."""
        # BRAPI has limited fundamental data in free tier
        return None


# List of popular Brazilian stocks for autocomplete
POPULAR_BR_STOCKS = [
    ("PETR4", "Petrobras PN"),
    ("VALE3", "Vale ON"),
    ("ITUB4", "Itaú Unibanco PN"),
    ("BBDC4", "Bradesco PN"),
    ("ABEV3", "Ambev ON"),
    ("B3SA3", "B3 ON"),
    ("WEGE3", "WEG ON"),
    ("RENT3", "Localiza ON"),
    ("BBAS3", "Banco do Brasil ON"),
    ("ITSA4", "Itaúsa PN"),
    ("MGLU3", "Magazine Luiza ON"),
    ("LREN3", "Lojas Renner ON"),
    ("SUZB3", "Suzano ON"),
    ("JBSS3", "JBS ON"),
    ("GGBR4", "Gerdau PN"),
    ("RADL3", "RaiaDrogasil ON"),
    ("RAIL3", "Rumo ON"),
    ("VIVT3", "Telefônica Brasil ON"),
    ("CSAN3", "Cosan ON"),
    ("ELET3", "Eletrobras ON"),
]

POPULAR_FIIS = [
    ("HGLG11", "CSHG Logística FII"),
    ("XPLG11", "XP Log FII"),
    ("KNRI11", "Kinea Renda Imobiliária FII"),
    ("MXRF11", "Maxi Renda FII"),
    ("VISC11", "Vinci Shopping Centers FII"),
    ("XPML11", "XP Malls FII"),
    ("HGRE11", "CSHG Real Estate FII"),
    ("BTLG11", "BTG Pactual Logística FII"),
    ("PVBI11", "VBI Prime Properties FII"),
    ("IRDM11", "Iridium Recebíveis FII"),
]
