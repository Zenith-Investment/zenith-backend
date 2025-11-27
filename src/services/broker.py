"""Broker connection service."""
import secrets
from datetime import datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from src.core.config import settings
from src.models.broker import (
    BrokerConnection,
    BrokerSyncHistory,
    BrokerType,
    ConnectionStatus,
    SyncStatus,
)
from src.integrations.brokers import (
    get_broker_client,
    BrokerConfig,
    BrokerType as IntegrationBrokerType,
    BaseBrokerClient,
)

logger = structlog.get_logger()


# Broker metadata
BROKER_INFO = {
    BrokerType.XP: {
        "name": "XP Investimentos",
        "description": "Maior corretora independente do Brasil",
        "documentation_url": "https://developer.xpinc.com/",
        "features": ["positions", "transactions", "accounts", "quotes"],
        "requires_mtls": False,
        "auth_type": "oauth2",
    },
    BrokerType.RICO: {
        "name": "Rico",
        "description": "Corretora do grupo XP com foco em investidores iniciantes",
        "documentation_url": "https://developer.xpinc.com/",
        "features": ["positions", "transactions", "accounts"],
        "requires_mtls": False,
        "auth_type": "oauth2",
    },
    BrokerType.CLEAR: {
        "name": "Clear",
        "description": "Corretora do grupo XP com foco em traders",
        "documentation_url": "https://developer.xpinc.com/",
        "features": ["positions", "transactions", "accounts"],
        "requires_mtls": False,
        "auth_type": "oauth2",
    },
    BrokerType.BTG: {
        "name": "BTG Pactual",
        "description": "Maior banco de investimentos da America Latina",
        "documentation_url": "https://developer.btgpactual.com/",
        "features": ["positions", "transactions", "accounts", "payments"],
        "requires_mtls": False,
        "auth_type": "oauth2",
    },
    BrokerType.NUINVEST: {
        "name": "NuInvest (Nubank)",
        "description": "Plataforma de investimentos do Nubank via Open Finance",
        "documentation_url": "https://openfinance.dev.br/provider/Nubank",
        "features": ["positions", "transactions", "accounts"],
        "requires_mtls": False,
        "auth_type": "oauth2_fapi",
    },
    BrokerType.INTER: {
        "name": "Banco Inter",
        "description": "Banco digital com plataforma de investimentos",
        "documentation_url": "https://developers.inter.co/",
        "features": ["positions", "transactions", "accounts", "pix", "boleto"],
        "requires_mtls": True,
        "auth_type": "oauth2_mtls",
    },
}


class BrokerService:
    """Service for managing broker connections."""

    def __init__(self, db: AsyncSession):
        self.db = db

    def get_supported_brokers(self) -> list[dict]:
        """Get list of all supported brokers with their info."""
        return [
            {
                "broker_type": broker_type,
                **info,
            }
            for broker_type, info in BROKER_INFO.items()
        ]

    async def get_user_connections(self, user_id: int) -> list[BrokerConnection]:
        """Get all broker connections for a user."""
        result = await self.db.execute(
            select(BrokerConnection)
            .where(BrokerConnection.user_id == user_id)
            .order_by(BrokerConnection.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_connection(
        self, user_id: int, connection_id: int
    ) -> BrokerConnection | None:
        """Get a specific broker connection."""
        result = await self.db.execute(
            select(BrokerConnection).where(
                BrokerConnection.id == connection_id,
                BrokerConnection.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_connection_by_broker(
        self, user_id: int, broker_type: BrokerType
    ) -> BrokerConnection | None:
        """Get connection for a specific broker."""
        result = await self.db.execute(
            select(BrokerConnection).where(
                BrokerConnection.user_id == user_id,
                BrokerConnection.broker_type == broker_type,
            )
        )
        return result.scalar_one_or_none()

    async def start_connection_flow(
        self, user_id: int, broker_type: BrokerType
    ) -> tuple[str, str]:
        """
        Start OAuth connection flow for a broker.

        Returns:
            Tuple of (authorization_url, state)
        """
        # Check if connection already exists
        existing = await self.get_connection_by_broker(user_id, broker_type)

        # Generate state for CSRF protection
        state = secrets.token_urlsafe(32)

        if existing:
            # Update existing connection
            existing.status = ConnectionStatus.PENDING
            existing.oauth_state = state
            existing.last_error = None
        else:
            # Create new connection
            existing = BrokerConnection(
                user_id=user_id,
                broker_type=broker_type,
                status=ConnectionStatus.PENDING,
                oauth_state=state,
            )
            self.db.add(existing)

        await self.db.commit()

        # Get authorization URL from broker client
        client = self._get_broker_client(broker_type)
        auth_url = await client.get_authorization_url(state)

        logger.info(
            "Started broker connection flow",
            user_id=user_id,
            broker_type=broker_type.value,
        )

        return auth_url, state

    async def complete_connection(
        self, user_id: int, code: str, state: str
    ) -> BrokerConnection:
        """
        Complete OAuth connection after callback.

        Args:
            user_id: User ID
            code: Authorization code from OAuth callback
            state: State from OAuth callback

        Returns:
            Updated BrokerConnection
        """
        # Find connection by state
        result = await self.db.execute(
            select(BrokerConnection).where(
                BrokerConnection.user_id == user_id,
                BrokerConnection.oauth_state == state,
                BrokerConnection.status == ConnectionStatus.PENDING,
            )
        )
        connection = result.scalar_one_or_none()

        if not connection:
            raise ValueError("Invalid state or connection not found")

        try:
            # Exchange code for tokens
            client = self._get_broker_client(connection.broker_type)
            token_data = await client.exchange_code_for_token(code)

            # Store tokens (encrypted in production)
            connection.access_token_encrypted = token_data.get("access_token")
            connection.refresh_token_encrypted = token_data.get("refresh_token")

            if "expires_in" in token_data:
                from datetime import timedelta
                connection.token_expires_at = datetime.now() + timedelta(
                    seconds=token_data["expires_in"]
                )

            # Get account info
            client.set_tokens(
                access_token=token_data.get("access_token"),
                refresh_token=token_data.get("refresh_token"),
            )
            accounts = await client.get_accounts()

            if accounts:
                connection.broker_account_id = accounts[0].account_id
                connection.broker_account_name = accounts[0].account_name

            connection.status = ConnectionStatus.ACTIVE
            connection.oauth_state = None
            connection.last_error = None

            await self.db.commit()
            await self.db.refresh(connection)

            logger.info(
                "Broker connection completed",
                user_id=user_id,
                connection_id=connection.id,
                broker_type=connection.broker_type.value,
            )

            return connection

        except Exception as e:
            connection.status = ConnectionStatus.ERROR
            connection.last_error = str(e)
            await self.db.commit()

            logger.error(
                "Failed to complete broker connection",
                user_id=user_id,
                error=str(e),
            )
            raise

    async def disconnect_broker(self, user_id: int, connection_id: int) -> None:
        """Disconnect a broker connection."""
        connection = await self.get_connection(user_id, connection_id)
        if not connection:
            raise ValueError("Connection not found")

        connection.status = ConnectionStatus.REVOKED
        connection.access_token_encrypted = None
        connection.refresh_token_encrypted = None
        connection.token_expires_at = None

        await self.db.commit()

        logger.info(
            "Broker disconnected",
            user_id=user_id,
            connection_id=connection_id,
        )

    async def sync_positions(
        self, user_id: int, connection_id: int
    ) -> BrokerSyncHistory:
        """Sync positions from broker."""
        connection = await self.get_connection(user_id, connection_id)
        if not connection:
            raise ValueError("Connection not found")

        if connection.status != ConnectionStatus.ACTIVE:
            raise ValueError("Connection is not active")

        # Create sync history entry
        sync_history = BrokerSyncHistory(
            connection_id=connection_id,
            sync_type="positions",
            status=SyncStatus.RUNNING,
        )
        self.db.add(sync_history)
        await self.db.commit()

        try:
            # Get positions from broker
            client = self._get_broker_client(connection.broker_type)
            client.set_tokens(
                access_token=connection.access_token_encrypted,
                refresh_token=connection.refresh_token_encrypted,
            )

            positions = await client.get_positions(connection.broker_account_id)

            # Update sync history
            sync_history.status = SyncStatus.SUCCESS
            sync_history.records_synced = len(positions)
            sync_history.completed_at = datetime.now()

            connection.last_sync_at = datetime.now()

            await self.db.commit()
            await self.db.refresh(sync_history)

            logger.info(
                "Positions synced",
                user_id=user_id,
                connection_id=connection_id,
                positions_count=len(positions),
            )

            return sync_history

        except Exception as e:
            sync_history.status = SyncStatus.FAILED
            sync_history.error_message = str(e)
            sync_history.completed_at = datetime.now()

            connection.last_error = str(e)

            await self.db.commit()

            logger.error(
                "Failed to sync positions",
                user_id=user_id,
                connection_id=connection_id,
                error=str(e),
            )
            raise

    async def get_broker_positions(
        self, user_id: int, connection_id: int
    ) -> list[dict]:
        """Get current positions from broker."""
        connection = await self.get_connection(user_id, connection_id)
        if not connection:
            raise ValueError("Connection not found")

        if connection.status != ConnectionStatus.ACTIVE:
            raise ValueError("Connection is not active")

        client = self._get_broker_client(connection.broker_type)
        client.set_tokens(
            access_token=connection.access_token_encrypted,
            refresh_token=connection.refresh_token_encrypted,
        )

        positions = await client.get_positions(connection.broker_account_id)

        return [
            {
                "ticker": p.ticker,
                "name": p.name,
                "quantity": p.quantity,
                "average_price": p.average_price,
                "current_price": p.current_price,
                "current_value": p.current_value,
                "profit_loss": p.profit_loss,
                "asset_class": p.asset_class,
            }
            for p in positions
        ]

    async def get_broker_transactions(
        self,
        user_id: int,
        connection_id: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict]:
        """Get transactions from broker."""
        connection = await self.get_connection(user_id, connection_id)
        if not connection:
            raise ValueError("Connection not found")

        if connection.status != ConnectionStatus.ACTIVE:
            raise ValueError("Connection is not active")

        client = self._get_broker_client(connection.broker_type)
        client.set_tokens(
            access_token=connection.access_token_encrypted,
            refresh_token=connection.refresh_token_encrypted,
        )

        transactions = await client.get_transactions(
            start_date=start_date,
            end_date=end_date,
            account_id=connection.broker_account_id,
        )

        return [
            {
                "transaction_id": t.transaction_id,
                "ticker": t.ticker,
                "transaction_type": t.transaction_type,
                "quantity": t.quantity,
                "price": t.price,
                "total_value": t.total_value,
                "fees": t.fees,
                "transaction_date": t.transaction_date,
                "settlement_date": t.settlement_date,
            }
            for t in transactions
        ]

    async def get_sync_history(
        self, user_id: int, connection_id: int, limit: int = 10
    ) -> list[BrokerSyncHistory]:
        """Get sync history for a connection."""
        # First verify user owns the connection
        connection = await self.get_connection(user_id, connection_id)
        if not connection:
            raise ValueError("Connection not found")

        result = await self.db.execute(
            select(BrokerSyncHistory)
            .where(BrokerSyncHistory.connection_id == connection_id)
            .order_by(BrokerSyncHistory.started_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    def _get_broker_client(self, broker_type: BrokerType) -> BaseBrokerClient:
        """Get broker client instance."""
        # Map model enum to integration enum
        integration_type = IntegrationBrokerType(broker_type.value)

        # Get credentials from settings (in production these would be from env)
        config = BrokerConfig(
            broker_type=integration_type,
            client_id=getattr(settings, f"{broker_type.value.upper()}_CLIENT_ID", ""),
            client_secret=getattr(settings, f"{broker_type.value.upper()}_CLIENT_SECRET", ""),
            redirect_uri=f"{settings.API_BASE_URL}/api/v1/brokers/callback/{broker_type.value}",
            sandbox=settings.BROKER_SANDBOX_MODE,
        )

        return get_broker_client(integration_type, config)
