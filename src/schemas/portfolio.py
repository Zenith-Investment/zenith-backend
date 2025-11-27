from datetime import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field


class AssetClass(str, Enum):
    STOCKS = "stocks"
    FIIS = "fiis"  # Fundos Imobiliários
    FIXED_INCOME = "fixed_income"  # Renda Fixa geral
    CDB = "cdb"  # Certificado de Depósito Bancário
    LCI = "lci"  # Letra de Crédito Imobiliário
    LCA = "lca"  # Letra de Crédito do Agronegócio
    TESOURO = "tesouro"  # Tesouro Direto
    CRYPTO = "crypto"
    ETF = "etf"
    BDR = "bdr"
    FUNDS = "funds"
    CASH = "cash"
    OTHER = "other"


class TransactionType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"
    SPLIT = "split"


class PortfolioType(str, Enum):
    """Portfolio type classification."""
    REAL = "real"
    SIMULATED = "simulated"
    WATCHLIST = "watchlist"


# ===========================================
# Portfolio Schemas (Multiple Portfolios)
# ===========================================

class PortfolioCreate(BaseModel):
    """Schema for creating a new portfolio."""
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    portfolio_type: PortfolioType = PortfolioType.REAL
    is_primary: bool = False
    color: str | None = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$")
    icon: str | None = None
    target_value: Decimal | None = Field(None, ge=0)
    risk_profile: str | None = None


class PortfolioUpdate(BaseModel):
    """Schema for updating a portfolio."""
    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = None
    portfolio_type: PortfolioType | None = None
    is_primary: bool | None = None
    color: str | None = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$")
    icon: str | None = None
    target_value: Decimal | None = Field(None, ge=0)
    risk_profile: str | None = None


class PortfolioBasicResponse(BaseModel):
    """Basic portfolio info without assets."""
    id: int
    name: str
    description: str | None
    portfolio_type: PortfolioType
    is_primary: bool
    color: str | None
    icon: str | None
    target_value: Decimal | None
    risk_profile: str | None
    total_value: Decimal | None = None
    assets_count: int = 0
    created_at: datetime
    updated_at: datetime | None

    class Config:
        from_attributes = True


class PortfolioListResponse(BaseModel):
    """Response for listing all user portfolios."""
    portfolios: list[PortfolioBasicResponse]
    total: int


class PortfolioAssetBase(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20)
    asset_class: AssetClass
    quantity: Decimal = Field(..., gt=0)
    average_price: Decimal = Field(..., gt=0)
    broker: str | None = None


class PortfolioAssetCreate(PortfolioAssetBase):
    purchase_date: datetime | None = None


class PortfolioAssetResponse(PortfolioAssetBase):
    id: int
    current_price: Decimal | None = None
    current_value: Decimal | None = None
    total_invested: Decimal
    profit_loss: Decimal | None = None
    profit_loss_percentage: float | None = None
    weight_in_portfolio: float | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class PortfolioSummary(BaseModel):
    total_invested: Decimal
    current_value: Decimal
    total_profit_loss: Decimal
    total_profit_loss_percentage: float
    assets_count: int


class AllocationByClass(BaseModel):
    asset_class: AssetClass
    value: Decimal
    percentage: float
    count: int


class PortfolioResponse(BaseModel):
    summary: PortfolioSummary
    assets: list[PortfolioAssetResponse]
    allocation_by_class: list[AllocationByClass]


class PerformanceDataPoint(BaseModel):
    date: datetime
    value: Decimal
    invested: Decimal


class PortfolioPerformanceResponse(BaseModel):
    period: str
    start_value: Decimal
    end_value: Decimal
    total_return: Decimal
    total_return_percentage: float
    annualized_return: float | None = None
    volatility: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    history: list[PerformanceDataPoint]


class RebalanceSuggestion(BaseModel):
    ticker: str
    action: str  # "buy" or "sell"
    quantity: Decimal
    reason: str


class RebalanceSimulationResponse(BaseModel):
    current_allocation: list[AllocationByClass]
    target_allocation: list[AllocationByClass]
    suggestions: list[RebalanceSuggestion]
    estimated_cost: Decimal


# Transaction Schemas
class TransactionCreate(BaseModel):
    asset_id: int
    transaction_type: TransactionType
    quantity: Decimal = Field(..., gt=0)
    price: Decimal = Field(..., gt=0)
    fees: Decimal = Field(default=Decimal("0"), ge=0)
    transaction_date: datetime
    notes: str | None = None


class TransactionResponse(BaseModel):
    id: int
    asset_id: int
    ticker: str
    asset_class: AssetClass
    transaction_type: TransactionType
    quantity: Decimal
    price: Decimal
    total_value: Decimal
    fees: Decimal
    transaction_date: datetime
    notes: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class TransactionListResponse(BaseModel):
    transactions: list[TransactionResponse]
    total: int
    page: int
    page_size: int


# CSV Import Schemas
class CSVImportRow(BaseModel):
    ticker: str
    asset_class: AssetClass
    quantity: Decimal
    average_price: Decimal
    broker: str | None = None
    purchase_date: datetime | None = None


class CSVImportResult(BaseModel):
    success: bool
    total_rows: int
    imported: int
    updated: int
    errors: list[str]
    assets: list[PortfolioAssetResponse]


class CSVTemplateInfo(BaseModel):
    columns: list[str]
    example_row: dict[str, str]
    asset_classes: list[str]
    notes: str
