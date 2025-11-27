"""
Banco Inter API client.

Documentation: https://developers.inter.co/
Help: https://ajuda.bancointer.com.br/pt-BR/articles/4284886

Authentication: OAuth 2.0 with mTLS (mutual TLS)
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


class InterClient(BaseBrokerClient):
    """
    Banco Inter API client.

    API Features:
    - Account balance and statement
    - Investment positions
    - Transaction history
    - PIX and TED transfers
    - Boleto payments

    Note: Requires Inter business account and API credentials.
    mTLS (mutual TLS) authentication required for production.
    """

    PRODUCTION_API_URL = "https://cdpj.partners.bancointer.com.br"
    SANDBOX_API_URL = "https://cdpj-sandbox.partners.bancointer.com.br"
    AUTH_URL = "https://cdpj.partners.bancointer.com.br/oauth/v2/authorize"
    TOKEN_URL = "https://cdpj.partners.bancointer.com.br/oauth/v2/token"

    @property
    def broker_type(self) -> BrokerType:
        return BrokerType.INTER

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
            "scope": "extrato.read boleto-cobranca.read boleto-cobranca.write",
            "state": state,
        }
        return f"{self.auth_url}?{urlencode(params)}"

    async def exchange_code_for_token(self, code: str) -> dict:
        # Inter requires mTLS for production
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
        # Inter uses a different endpoint structure
        data = await self._make_request("GET", "/banking/v2/extrato")
        return [
            BrokerAccount(
                account_id="main",
                account_name="Conta Inter",
                account_type="checking",
                balance=Decimal(str(data.get("saldo", 0))),
            )
        ]

    async def get_account_balance(self, account_id: str) -> Decimal:
        data = await self._make_request("GET", "/banking/v2/saldo")
        return Decimal(str(data.get("disponivel", 0)))

    async def get_positions(self, account_id: Optional[str] = None) -> list[BrokerPosition]:
        # Inter investments endpoint
        try:
            data = await self._make_request("GET", "/investimentos/v1/posicao")
            positions = []

            for inv in data.get("investimentos", []):
                positions.append(
                    BrokerPosition(
                        ticker=inv.get("codigo", inv.get("nome", "")),
                        name=inv.get("nome", ""),
                        quantity=Decimal(str(inv.get("quantidade", 0))),
                        average_price=Decimal(str(inv.get("precoMedio", 0))),
                        current_price=Decimal(str(inv.get("precoAtual", 0))) if inv.get("precoAtual") else None,
                        current_value=Decimal(str(inv.get("valorAtual", 0))) if inv.get("valorAtual") else None,
                        asset_class=inv.get("tipoInvestimento"),
                        broker_id=inv.get("id"),
                    )
                )
            return positions

        except Exception as e:
            logger.warning("Failed to get Inter positions", error=str(e))
            return []

    async def get_position(self, ticker: str, account_id: Optional[str] = None) -> Optional[BrokerPosition]:
        positions = await self.get_positions(account_id)
        return next((p for p in positions if p.ticker.upper() == ticker.upper()), None)

    async def get_transactions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        account_id: Optional[str] = None,
    ) -> list[BrokerTransaction]:
        params = {}
        if start_date:
            params["dataInicio"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["dataFim"] = end_date.strftime("%Y-%m-%d")

        data = await self._make_request("GET", "/banking/v2/extrato", params=params)
        transactions = []

        for tx in data.get("transacoes", []):
            transactions.append(
                BrokerTransaction(
                    transaction_id=tx.get("idTransacao", ""),
                    ticker=tx.get("descricao", ""),
                    transaction_type=tx.get("tipoTransacao", ""),
                    quantity=Decimal("1"),  # Inter doesn't provide quantity for banking transactions
                    price=Decimal(str(abs(tx.get("valor", 0)))),
                    total_value=Decimal(str(abs(tx.get("valor", 0)))),
                    fees=Decimal("0"),
                    transaction_date=datetime.fromisoformat(tx.get("dataEntrada", datetime.now().isoformat())),
                )
            )
        return transactions

    async def get_statement(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict:
        """Get bank statement (extrato)."""
        params = {}
        if start_date:
            params["dataInicio"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["dataFim"] = end_date.strftime("%Y-%m-%d")

        return await self._make_request("GET", "/banking/v2/extrato", params=params)
