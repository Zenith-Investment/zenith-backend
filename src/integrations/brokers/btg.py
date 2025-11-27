"""
BTG Pactual API client.

Documentation: https://developer.btgpactual.com/
Empresas: https://developers.empresas.btgpactual.com/docs

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


class BTGClient(BaseBrokerClient):
    """
    BTG Pactual API client.

    API Features:
    - Account balance and statement
    - Investment positions
    - Transaction history
    - Boleto and Pix payments (empresas)

    Note: Full API access requires BTG Pactual account and API credentials.
    """

    PRODUCTION_API_URL = "https://api.btgpactual.com"
    SANDBOX_API_URL = "https://sandbox.api.btgpactual.com"
    AUTH_URL = "https://auth.btgpactual.com/oauth2/authorize"
    TOKEN_URL = "https://auth.btgpactual.com/oauth2/token"

    @property
    def broker_type(self) -> BrokerType:
        return BrokerType.BTG

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
        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "response_type": "code",
            "scope": "openid profile investments accounts",
            "state": state,
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
        data = await self._make_request("GET", "/v1/accounts")
        return [
            BrokerAccount(
                account_id=acc["id"],
                account_name=acc.get("name", "Conta BTG"),
                account_type=acc.get("type", "individual"),
                balance=Decimal(str(acc.get("balance", 0))),
            )
            for acc in data.get("accounts", [])
        ]

    async def get_account_balance(self, account_id: str) -> Decimal:
        data = await self._make_request("GET", f"/v1/accounts/{account_id}/balance")
        return Decimal(str(data.get("available", 0)))

    async def get_positions(self, account_id: Optional[str] = None) -> list[BrokerPosition]:
        endpoint = f"/v1/accounts/{account_id}/investments" if account_id else "/v1/investments"
        data = await self._make_request("GET", endpoint)
        return [
            BrokerPosition(
                ticker=pos["ticker"],
                name=pos.get("name", pos["ticker"]),
                quantity=Decimal(str(pos["quantity"])),
                average_price=Decimal(str(pos["average_price"])),
                current_price=Decimal(str(pos["current_price"])) if pos.get("current_price") else None,
                current_value=Decimal(str(pos["current_value"])) if pos.get("current_value") else None,
                asset_class=pos.get("asset_class"),
                broker_id=pos.get("id"),
            )
            for pos in data.get("investments", [])
        ]

    async def get_position(self, ticker: str, account_id: Optional[str] = None) -> Optional[BrokerPosition]:
        positions = await self.get_positions(account_id)
        return next((p for p in positions if p.ticker.upper() == ticker.upper()), None)

    async def get_transactions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        account_id: Optional[str] = None,
    ) -> list[BrokerTransaction]:
        endpoint = f"/v1/accounts/{account_id}/transactions" if account_id else "/v1/transactions"
        params = {}
        if start_date:
            params["start_date"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["end_date"] = end_date.strftime("%Y-%m-%d")

        data = await self._make_request("GET", endpoint, params=params)
        return [
            BrokerTransaction(
                transaction_id=tx["id"],
                ticker=tx["ticker"],
                transaction_type=tx["type"],
                quantity=Decimal(str(tx["quantity"])),
                price=Decimal(str(tx["price"])),
                total_value=Decimal(str(tx["total_value"])),
                fees=Decimal(str(tx.get("fees", 0))),
                transaction_date=datetime.fromisoformat(tx["date"]),
            )
            for tx in data.get("transactions", [])
        ]
