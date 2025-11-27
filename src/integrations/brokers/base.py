"""Base broker client interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

import structlog

logger = structlog.get_logger()


class BrokerType(str, Enum):
    """Supported broker types."""
    XP = "xp"
    RICO = "rico"
    CLEAR = "clear"
    BTG = "btg"
    NUINVEST = "nuinvest"
    INTER = "inter"


@dataclass
class BrokerConfig:
    """Configuration for broker API connection."""
    broker_type: BrokerType
    client_id: str
    client_secret: str
    redirect_uri: Optional[str] = None
    sandbox: bool = True
    api_base_url: Optional[str] = None


@dataclass
class BrokerPosition:
    """A position held at a broker."""
    ticker: str
    name: str
    quantity: Decimal
    average_price: Decimal
    current_price: Optional[Decimal] = None
    current_value: Optional[Decimal] = None
    profit_loss: Optional[Decimal] = None
    asset_class: Optional[str] = None
    broker_id: Optional[str] = None  # Broker's internal ID


@dataclass
class BrokerTransaction:
    """A transaction from a broker."""
    transaction_id: str
    ticker: str
    transaction_type: str  # buy, sell, dividend, etc.
    quantity: Decimal
    price: Decimal
    total_value: Decimal
    fees: Decimal
    transaction_date: datetime
    settlement_date: Optional[datetime] = None


@dataclass
class BrokerAccount:
    """Broker account information."""
    account_id: str
    account_name: str
    account_type: str  # individual, joint, corporate
    balance: Decimal
    currency: str = "BRL"


class BaseBrokerClient(ABC):
    """
    Abstract base class for broker integrations.

    All broker clients should inherit from this class and implement
    the required methods.
    """

    def __init__(self, config: BrokerConfig):
        self.config = config
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None

    @property
    @abstractmethod
    def broker_type(self) -> BrokerType:
        """Return the broker type."""
        pass

    @property
    @abstractmethod
    def api_base_url(self) -> str:
        """Return the API base URL."""
        pass

    @property
    @abstractmethod
    def auth_url(self) -> str:
        """Return the OAuth authorization URL."""
        pass

    @property
    @abstractmethod
    def token_url(self) -> str:
        """Return the OAuth token URL."""
        pass

    # Authentication methods
    @abstractmethod
    async def get_authorization_url(self, state: str) -> str:
        """
        Get the OAuth authorization URL for user consent.

        Args:
            state: A random state string for CSRF protection

        Returns:
            The authorization URL to redirect the user to
        """
        pass

    @abstractmethod
    async def exchange_code_for_token(self, code: str) -> dict:
        """
        Exchange an authorization code for access tokens.

        Args:
            code: The authorization code from the OAuth callback

        Returns:
            Dict with access_token, refresh_token, expires_in
        """
        pass

    @abstractmethod
    async def refresh_access_token(self) -> dict:
        """
        Refresh the access token using the refresh token.

        Returns:
            Dict with new access_token, refresh_token, expires_in
        """
        pass

    # Account methods
    @abstractmethod
    async def get_accounts(self) -> list[BrokerAccount]:
        """
        Get all accounts for the authenticated user.

        Returns:
            List of BrokerAccount objects
        """
        pass

    @abstractmethod
    async def get_account_balance(self, account_id: str) -> Decimal:
        """
        Get the balance of a specific account.

        Args:
            account_id: The account identifier

        Returns:
            The account balance
        """
        pass

    # Portfolio methods
    @abstractmethod
    async def get_positions(self, account_id: Optional[str] = None) -> list[BrokerPosition]:
        """
        Get all positions in the portfolio.

        Args:
            account_id: Optional account ID (if multiple accounts)

        Returns:
            List of BrokerPosition objects
        """
        pass

    @abstractmethod
    async def get_position(self, ticker: str, account_id: Optional[str] = None) -> Optional[BrokerPosition]:
        """
        Get a specific position by ticker.

        Args:
            ticker: The asset ticker symbol
            account_id: Optional account ID

        Returns:
            BrokerPosition if found, None otherwise
        """
        pass

    # Transaction methods
    @abstractmethod
    async def get_transactions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        account_id: Optional[str] = None,
    ) -> list[BrokerTransaction]:
        """
        Get transaction history.

        Args:
            start_date: Start of date range
            end_date: End of date range
            account_id: Optional account ID

        Returns:
            List of BrokerTransaction objects
        """
        pass

    # Helper methods
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> Any:
        """Make an authenticated API request."""
        import httpx

        url = f"{self.api_base_url}{endpoint}"
        request_headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
            **(headers or {}),
        }

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method,
                url=url,
                json=data,
                params=params,
                headers=request_headers,
            )
            response.raise_for_status()
            return response.json()

    def set_tokens(
        self,
        access_token: str,
        refresh_token: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ):
        """Set the authentication tokens."""
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._token_expires_at = expires_at

    def is_token_expired(self) -> bool:
        """Check if the access token is expired."""
        if not self._token_expires_at:
            return True
        return datetime.now() >= self._token_expires_at
