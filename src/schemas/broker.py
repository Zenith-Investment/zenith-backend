"""Broker integration schemas."""
from datetime import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field


class BrokerType(str, Enum):
    """Supported broker types."""
    XP = "xp"
    RICO = "rico"
    CLEAR = "clear"
    BTG = "btg"
    NUINVEST = "nuinvest"
    INTER = "inter"


class ConnectionStatus(str, Enum):
    """Broker connection status."""
    PENDING = "pending"
    ACTIVE = "active"
    EXPIRED = "expired"
    ERROR = "error"
    REVOKED = "revoked"


class SyncStatus(str, Enum):
    """Synchronization status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


# Request schemas
class BrokerConnectRequest(BaseModel):
    """Request to start broker connection flow."""
    broker_type: BrokerType


class BrokerCallbackRequest(BaseModel):
    """OAuth callback data."""
    code: str
    state: str


# Response schemas
class BrokerAuthUrlResponse(BaseModel):
    """OAuth authorization URL response."""
    authorization_url: str
    state: str


class BrokerConnectionResponse(BaseModel):
    """Broker connection details."""
    id: int
    broker_type: BrokerType
    status: ConnectionStatus
    broker_account_id: str | None = None
    broker_account_name: str | None = None
    last_sync_at: datetime | None = None
    last_error: str | None = None
    created_at: datetime
    updated_at: datetime | None = None

    class Config:
        from_attributes = True


class BrokerConnectionListResponse(BaseModel):
    """List of broker connections."""
    connections: list[BrokerConnectionResponse]
    total: int


class BrokerSyncHistoryResponse(BaseModel):
    """Sync history entry."""
    id: int
    sync_type: str
    status: SyncStatus
    records_synced: int
    records_created: int
    records_updated: int
    error_message: str | None = None
    started_at: datetime
    completed_at: datetime | None = None

    class Config:
        from_attributes = True


class BrokerSyncResponse(BaseModel):
    """Sync operation result."""
    sync_id: int
    status: SyncStatus
    message: str


# Portfolio data from broker
class BrokerPositionResponse(BaseModel):
    """Position from broker."""
    ticker: str
    name: str
    quantity: Decimal
    average_price: Decimal
    current_price: Decimal | None = None
    current_value: Decimal | None = None
    profit_loss: Decimal | None = None
    asset_class: str | None = None


class BrokerAccountResponse(BaseModel):
    """Account from broker."""
    account_id: str
    account_name: str
    account_type: str
    balance: Decimal
    currency: str = "BRL"


class BrokerTransactionResponse(BaseModel):
    """Transaction from broker."""
    transaction_id: str
    ticker: str
    transaction_type: str
    quantity: Decimal
    price: Decimal
    total_value: Decimal
    fees: Decimal
    transaction_date: datetime
    settlement_date: datetime | None = None


class BrokerPortfolioResponse(BaseModel):
    """Import operation result."""
    imported: int
    updated: int
    skipped: int
    message: str


class BrokerFullPortfolioResponse(BaseModel):
    """Complete portfolio data from broker."""
    connection_id: int
    broker_type: BrokerType
    accounts: list[BrokerAccountResponse]
    positions: list[BrokerPositionResponse]
    total_value: Decimal
    last_sync_at: datetime | None = None


class SupportedBrokerResponse(BaseModel):
    """Information about a supported broker."""
    broker_type: BrokerType
    name: str
    description: str
    documentation_url: str
    features: list[str]
    requires_mtls: bool = False
    auth_type: str = "oauth2"


class SupportedBrokersListResponse(BaseModel):
    """List of all supported brokers."""
    brokers: list[SupportedBrokerResponse]
