"""
XP Investimentos API client.

This client supports:
- XP Investimentos
- Rico (parte do grupo XP)
- Clear (parte do grupo XP)

Documentation: https://developer.xpinc.com/
Open Finance: https://developer.xpinc.com/open-finance

Authentication: OAuth 2.0
"""
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional
from urllib.parse import urlencode

import httpx
import structlog

from src.integrations.brokers.base import (
    BaseBrokerClient,
    BrokerAccount,
    BrokerConfig,
    BrokerPosition,
    BrokerTransaction,
    BrokerType,
)

logger = structlog.get_logger()


class XPClient(BaseBrokerClient):
    """
    XP Investimentos API client.

    Supports XP, Rico, and Clear through the XP Inc Developer Portal.

    API Features:
    - Portfolio positions and values
    - Transaction history
    - Account information
    - Real-time quotes (via partner integrations)

    Note: Full API access requires partnership agreement with XP.
    """

    # API URLs
    PRODUCTION_API_URL = "https://api.xpinc.com"
    SANDBOX_API_URL = "https://sandbox.api.xpinc.com"
    AUTH_URL = "https://auth.xpinc.com/oauth2/authorize"
    TOKEN_URL = "https://auth.xpinc.com/oauth2/token"

    @property
    def broker_type(self) -> BrokerType:
        return BrokerType.XP

    @property
    def api_base_url(self) -> str:
        if self.config.api_base_url:
            return self.config.api_base_url
        return self.SANDBOX_API_URL if self.config.sandbox else self.PRODUCTION_API_URL

    @property
    def auth_url(self) -> str:
        return self.AUTH_URL

    @property
    def token_url(self) -> str:
        return self.TOKEN_URL

    async def get_authorization_url(self, state: str) -> str:
        """Get OAuth authorization URL."""
        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "response_type": "code",
            "scope": "openid profile positions transactions accounts",
            "state": state,
        }
        return f"{self.auth_url}?{urlencode(params)}"

    async def exchange_code_for_token(self, code: str) -> dict:
        """Exchange authorization code for tokens."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_url,
                data={
                    "grant_type": "authorization_code",
                    "client_id": self.config.client_id,
                    "client_secret": self.config.client_secret,
                    "code": code,
                    "redirect_uri": self.config.redirect_uri,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Set tokens
            expires_at = datetime.now() + timedelta(seconds=data.get("expires_in", 3600))
            self.set_tokens(
                access_token=data["access_token"],
                refresh_token=data.get("refresh_token"),
                expires_at=expires_at,
            )

            return data

    async def refresh_access_token(self) -> dict:
        """Refresh the access token."""
        if not self._refresh_token:
            raise ValueError("No refresh token available")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_url,
                data={
                    "grant_type": "refresh_token",
                    "client_id": self.config.client_id,
                    "client_secret": self.config.client_secret,
                    "refresh_token": self._refresh_token,
                },
            )
            response.raise_for_status()
            data = response.json()

            expires_at = datetime.now() + timedelta(seconds=data.get("expires_in", 3600))
            self.set_tokens(
                access_token=data["access_token"],
                refresh_token=data.get("refresh_token", self._refresh_token),
                expires_at=expires_at,
            )

            return data

    async def get_accounts(self) -> list[BrokerAccount]:
        """Get all accounts for the user."""
        try:
            data = await self._make_request("GET", "/v1/accounts")

            accounts = []
            for acc in data.get("accounts", []):
                accounts.append(
                    BrokerAccount(
                        account_id=acc["id"],
                        account_name=acc.get("name", "Conta XP"),
                        account_type=acc.get("type", "individual"),
                        balance=Decimal(str(acc.get("balance", 0))),
                        currency="BRL",
                    )
                )
            return accounts

        except Exception as e:
            logger.error("Failed to get XP accounts", error=str(e))
            raise

    async def get_account_balance(self, account_id: str) -> Decimal:
        """Get balance for a specific account."""
        try:
            data = await self._make_request("GET", f"/v1/accounts/{account_id}/balance")
            return Decimal(str(data.get("available_balance", 0)))
        except Exception as e:
            logger.error("Failed to get XP account balance", account_id=account_id, error=str(e))
            raise

    async def get_positions(self, account_id: Optional[str] = None) -> list[BrokerPosition]:
        """Get all positions in the portfolio."""
        try:
            endpoint = "/v1/positions"
            if account_id:
                endpoint = f"/v1/accounts/{account_id}/positions"

            data = await self._make_request("GET", endpoint)

            positions = []
            for pos in data.get("positions", []):
                positions.append(
                    BrokerPosition(
                        ticker=pos["ticker"],
                        name=pos.get("name", pos["ticker"]),
                        quantity=Decimal(str(pos["quantity"])),
                        average_price=Decimal(str(pos["average_price"])),
                        current_price=Decimal(str(pos["current_price"])) if pos.get("current_price") else None,
                        current_value=Decimal(str(pos["current_value"])) if pos.get("current_value") else None,
                        profit_loss=Decimal(str(pos["profit_loss"])) if pos.get("profit_loss") else None,
                        asset_class=pos.get("asset_class"),
                        broker_id=pos.get("id"),
                    )
                )
            return positions

        except Exception as e:
            logger.error("Failed to get XP positions", error=str(e))
            raise

    async def get_position(self, ticker: str, account_id: Optional[str] = None) -> Optional[BrokerPosition]:
        """Get a specific position by ticker."""
        positions = await self.get_positions(account_id)
        for pos in positions:
            if pos.ticker.upper() == ticker.upper():
                return pos
        return None

    async def get_transactions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        account_id: Optional[str] = None,
    ) -> list[BrokerTransaction]:
        """Get transaction history."""
        try:
            endpoint = "/v1/transactions"
            if account_id:
                endpoint = f"/v1/accounts/{account_id}/transactions"

            params = {}
            if start_date:
                params["start_date"] = start_date.strftime("%Y-%m-%d")
            if end_date:
                params["end_date"] = end_date.strftime("%Y-%m-%d")

            data = await self._make_request("GET", endpoint, params=params)

            transactions = []
            for tx in data.get("transactions", []):
                transactions.append(
                    BrokerTransaction(
                        transaction_id=tx["id"],
                        ticker=tx["ticker"],
                        transaction_type=tx["type"],
                        quantity=Decimal(str(tx["quantity"])),
                        price=Decimal(str(tx["price"])),
                        total_value=Decimal(str(tx["total_value"])),
                        fees=Decimal(str(tx.get("fees", 0))),
                        transaction_date=datetime.fromisoformat(tx["date"]),
                        settlement_date=datetime.fromisoformat(tx["settlement_date"]) if tx.get("settlement_date") else None,
                    )
                )
            return transactions

        except Exception as e:
            logger.error("Failed to get XP transactions", error=str(e))
            raise


# Aliases for Rico and Clear (same API, different branding)
class RicoClient(XPClient):
    """Rico API client (uses XP infrastructure)."""

    @property
    def broker_type(self) -> BrokerType:
        return BrokerType.RICO


class ClearClient(XPClient):
    """Clear API client (uses XP infrastructure)."""

    @property
    def broker_type(self) -> BrokerType:
        return BrokerType.CLEAR
