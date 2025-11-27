"""Base interface for market data providers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum


class AssetType(str, Enum):
    STOCK = "stock"
    ETF = "etf"
    FII = "fii"
    BDR = "bdr"
    INDEX = "index"
    CRYPTO = "crypto"
    FUND = "fund"


@dataclass
class AssetQuote:
    """Real-time asset quote."""
    ticker: str
    name: str
    asset_type: AssetType
    current_price: Decimal
    change: Decimal
    change_percent: float
    open_price: Decimal | None = None
    high: Decimal | None = None
    low: Decimal | None = None
    previous_close: Decimal | None = None
    volume: int | None = None
    market_cap: Decimal | None = None
    currency: str = "BRL"
    exchange: str | None = None
    last_updated: datetime | None = None


@dataclass
class AssetFundamentals:
    """Fundamental data for an asset."""
    ticker: str
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    dividend_yield: float | None = None
    roe: float | None = None
    roa: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    profit_margin: float | None = None
    revenue: Decimal | None = None
    net_income: Decimal | None = None
    eps: float | None = None
    sector: str | None = None
    industry: str | None = None
    description: str | None = None
    website: str | None = None
    employees: int | None = None


@dataclass
class PriceHistory:
    """Historical price data point."""
    date: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    adjusted_close: Decimal | None = None


@dataclass
class AssetSearchResult:
    """Search result for an asset."""
    ticker: str
    name: str
    asset_type: AssetType
    exchange: str | None = None


class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    async def get_quote(self, ticker: str) -> AssetQuote | None:
        """Get real-time quote for a ticker."""
        pass

    @abstractmethod
    async def get_quotes(self, tickers: list[str]) -> dict[str, AssetQuote]:
        """Get real-time quotes for multiple tickers."""
        pass

    @abstractmethod
    async def get_history(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> list[PriceHistory]:
        """Get historical price data."""
        pass

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> list[AssetSearchResult]:
        """Search for assets by name or ticker."""
        pass

    @abstractmethod
    async def get_fundamentals(self, ticker: str) -> AssetFundamentals | None:
        """Get fundamental data for an asset."""
        pass
