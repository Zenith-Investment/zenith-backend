from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel


class AssetSearchResult(BaseModel):
    ticker: str
    name: str
    asset_type: str
    exchange: str | None = None


class AssetSearchResponse(BaseModel):
    query: str
    results: list[AssetSearchResult]
    total: int


class AssetDetailResponse(BaseModel):
    ticker: str
    name: str
    asset_type: str
    exchange: str | None = None
    current_price: Decimal
    currency: str = "BRL"
    change: Decimal
    change_percent: float
    open_price: Decimal | None = None
    high: Decimal | None = None
    low: Decimal | None = None
    volume: int | None = None
    market_cap: Decimal | None = None
    pe_ratio: float | None = None
    dividend_yield: float | None = None
    week_52_high: Decimal | None = None
    week_52_low: Decimal | None = None
    description: str | None = None
    sector: str | None = None
    industry: str | None = None
    last_updated: datetime


class PriceDataPoint(BaseModel):
    date: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


class AssetHistoryResponse(BaseModel):
    ticker: str
    period: str
    data: list[PriceDataPoint]


class MarketIndex(BaseModel):
    symbol: str
    name: str
    value: Decimal
    change: Decimal
    change_percent: float
    last_updated: datetime


class MarketIndicesResponse(BaseModel):
    indices: list[MarketIndex]


class NewsItem(BaseModel):
    id: str
    title: str
    summary: str | None = None
    url: str
    source: str
    published_at: datetime
    related_tickers: list[str] = []
    sentiment: str | None = None  # positive, negative, neutral


class MarketNewsResponse(BaseModel):
    news: list[NewsItem]
    total: int
