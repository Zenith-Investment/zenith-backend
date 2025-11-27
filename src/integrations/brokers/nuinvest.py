"""
NuInvest (Nubank) API client.

Uses Open Finance Brasil APIs for data access.
Documentation: https://openfinance.dev.br/provider/Nubank

Authentication: OAuth 2.0 with FAPI compliance
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


class NuInvestClient(BaseBrokerClient):
    """
    NuInvest (Nubank) API client via Open Finance.

    Uses Open Finance Brasil standard APIs.
    Requires Open Finance registration and certification.

    API Features:
    - Investment positions
    - Transaction history
    - Account data

    Note: Requires Open Finance certification for production access.
    """

    # Open Finance Brasil endpoints (Nubank specific)
    PRODUCTION_API_URL = "https://open-finance.nubank.com.br"
    SANDBOX_API_URL = "https://sandbox.open-finance.nubank.com.br"
    AUTH_URL = "https://auth.nubank.com.br/oauth2/authorize"
    TOKEN_URL = "https://auth.nubank.com.br/oauth2/token"

    @property
    def broker_type(self) -> BrokerType:
        return BrokerType.NUINVEST

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
        # Open Finance uses FAPI (Financial-grade API) security profile
        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "response_type": "code",
            "scope": "openid investments accounts",
            "state": state,
            # FAPI requires PKCE
            "code_challenge_method": "S256",
        }
        return f"{self.auth_url}?{urlencode(params)}"

    async def exchange_code_for_token(self, code: str) -> dict:
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
            expires_at = datetime.now() + timedelta(seconds=data.get("expires_in", 3600))
            self.set_tokens(data["access_token"], data.get("refresh_token"), expires_at)
            return data

    async def refresh_access_token(self) -> dict:
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
            self.set_tokens(data["access_token"], data.get("refresh_token", self._refresh_token), expires_at)
            return data

    async def get_accounts(self) -> list[BrokerAccount]:
        # Open Finance standard endpoint
        data = await self._make_request("GET", "/investments/v1/accounts")
        return [
            BrokerAccount(
                account_id=acc["accountId"],
                account_name=acc.get("brandName", "NuInvest"),
                account_type=acc.get("type", "individual"),
                balance=Decimal(str(acc.get("availableAmount", {}).get("amount", 0))),
            )
            for acc in data.get("data", [])
        ]

    async def get_account_balance(self, account_id: str) -> Decimal:
        data = await self._make_request("GET", f"/investments/v1/accounts/{account_id}/balances")
        balance_data = data.get("data", {})
        return Decimal(str(balance_data.get("availableAmount", {}).get("amount", 0)))

    async def get_positions(self, account_id: Optional[str] = None) -> list[BrokerPosition]:
        # Open Finance investments endpoint
        endpoint = "/investments/v1/investments"
        if account_id:
            endpoint = f"/investments/v1/accounts/{account_id}/investments"

        data = await self._make_request("GET", endpoint)
        positions = []

        for inv in data.get("data", []):
            positions.append(
                BrokerPosition(
                    ticker=inv.get("ticker", inv.get("investmentId", "")),
                    name=inv.get("brandName", inv.get("ticker", "")),
                    quantity=Decimal(str(inv.get("quantity", {}).get("amount", 0))),
                    average_price=Decimal(str(inv.get("acquisitionUnitPrice", {}).get("amount", 0))),
                    current_value=Decimal(str(inv.get("grossAmount", {}).get("amount", 0))),
                    asset_class=inv.get("investmentType"),
                    broker_id=inv.get("investmentId"),
                )
            )
        return positions

    async def get_position(self, ticker: str, account_id: Optional[str] = None) -> Optional[BrokerPosition]:
        positions = await self.get_positions(account_id)
        return next((p for p in positions if p.ticker.upper() == ticker.upper()), None)

    async def get_transactions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        account_id: Optional[str] = None,
    ) -> list[BrokerTransaction]:
        endpoint = "/investments/v1/transactions"
        if account_id:
            endpoint = f"/investments/v1/accounts/{account_id}/transactions"

        params = {}
        if start_date:
            params["fromDate"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["toDate"] = end_date.strftime("%Y-%m-%d")

        data = await self._make_request("GET", endpoint, params=params)
        return [
            BrokerTransaction(
                transaction_id=tx.get("transactionId", ""),
                ticker=tx.get("ticker", ""),
                transaction_type=tx.get("type", ""),
                quantity=Decimal(str(tx.get("quantity", {}).get("amount", 0))),
                price=Decimal(str(tx.get("unitPrice", {}).get("amount", 0))),
                total_value=Decimal(str(tx.get("transactionValue", {}).get("amount", 0))),
                fees=Decimal(str(tx.get("transactionTax", {}).get("amount", 0))),
                transaction_date=datetime.fromisoformat(tx.get("transactionDate", datetime.now().isoformat())),
            )
            for tx in data.get("data", [])
        ]
