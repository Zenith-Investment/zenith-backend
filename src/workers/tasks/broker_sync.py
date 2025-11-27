"""Broker synchronization background tasks."""
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import structlog
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.core.database import get_celery_async_session
from src.models.broker import (
    BrokerConnection,
    BrokerSyncHistory,
    BrokerType,
    ConnectionStatus,
    SyncStatus,
)
from src.models.portfolio import Portfolio, PortfolioAsset
from src.models.user import User
from src.workers.celery_app import celery_app

logger = structlog.get_logger()


def run_async(coro):
    """Helper to run async code in sync Celery tasks.

    Creates a fresh event loop for each task execution and properly
    cleans up async resources to avoid 'Event loop is closed' errors.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        # Clean up all pending tasks
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass

        loop.run_until_complete(loop.shutdown_asyncgens())
        try:
            loop.run_until_complete(loop.shutdown_default_executor())
        except Exception:
            pass

        loop.close()


async def _get_active_connections() -> list[dict]:
    """Get all active broker connections."""
    session_factory = get_celery_async_session()
    async with session_factory() as session:
        result = await session.execute(
            select(BrokerConnection)
            .where(BrokerConnection.status == ConnectionStatus.ACTIVE)
            .options(selectinload(BrokerConnection.user))
        )
        connections = result.scalars().all()

        return [
            {
                "id": conn.id,
                "user_id": conn.user_id,
                "broker_type": conn.broker_type,
                "access_token": conn.access_token_encrypted,
                "refresh_token": conn.refresh_token_encrypted,
                "token_expires_at": conn.token_expires_at,
                "broker_account_id": conn.broker_account_id,
                "last_sync_at": conn.last_sync_at,
            }
            for conn in connections
        ]


async def _refresh_token_if_needed(connection: dict) -> dict | None:
    """Refresh token if it's expired or about to expire."""
    from src.core.config import settings
    from src.integrations.brokers import get_broker_client, BrokerConfig, BrokerType as IntegrationBrokerType

    token_expires_at = connection.get("token_expires_at")

    # Check if token needs refresh (expires in less than 5 minutes)
    if token_expires_at:
        if isinstance(token_expires_at, str):
            token_expires_at = datetime.fromisoformat(token_expires_at)

        if token_expires_at > datetime.now(timezone.utc) + timedelta(minutes=5):
            # Token is still valid
            return None

    # Token needs refresh
    broker_type = connection["broker_type"]
    if isinstance(broker_type, str):
        broker_type = BrokerType(broker_type)

    try:
        integration_type = IntegrationBrokerType(broker_type.value)
        config = BrokerConfig(
            broker_type=integration_type,
            client_id=getattr(settings, f"{broker_type.value.upper()}_CLIENT_ID", ""),
            client_secret=getattr(settings, f"{broker_type.value.upper()}_CLIENT_SECRET", ""),
            redirect_uri=f"{settings.API_BASE_URL}/api/v1/brokers/callback/{broker_type.value}",
            sandbox=settings.BROKER_SANDBOX_MODE,
        )

        client = get_broker_client(integration_type, config)
        client.set_tokens(
            access_token=connection["access_token"],
            refresh_token=connection["refresh_token"],
        )

        new_tokens = await client.refresh_access_token()

        if new_tokens:
            logger.info(
                "Token refreshed",
                connection_id=connection["id"],
                broker_type=broker_type.value,
            )
            return new_tokens

    except Exception as e:
        logger.warning(
            "Failed to refresh token",
            connection_id=connection["id"],
            error=str(e),
        )

    return None


async def _sync_connection_positions(connection: dict) -> dict:
    """Sync positions from a single broker connection."""
    from src.core.config import settings
    from src.integrations.brokers import get_broker_client, BrokerConfig, BrokerType as IntegrationBrokerType

    connection_id = connection["id"]
    user_id = connection["user_id"]
    broker_type = connection["broker_type"]

    if isinstance(broker_type, str):
        broker_type = BrokerType(broker_type)

    session_factory = get_celery_async_session()
    async with session_factory() as session:
        # Create sync history entry
        sync_history = BrokerSyncHistory(
            connection_id=connection_id,
            sync_type="positions",
            status=SyncStatus.RUNNING,
        )
        session.add(sync_history)
        await session.commit()

        try:
            # Check and refresh token if needed
            new_tokens = await _refresh_token_if_needed(connection)

            access_token = connection["access_token"]
            refresh_token = connection["refresh_token"]

            if new_tokens:
                access_token = new_tokens.get("access_token", access_token)
                refresh_token = new_tokens.get("refresh_token", refresh_token)

                # Update tokens in database
                conn_result = await session.execute(
                    select(BrokerConnection).where(BrokerConnection.id == connection_id)
                )
                db_connection = conn_result.scalar_one_or_none()
                if db_connection:
                    db_connection.access_token_encrypted = access_token
                    db_connection.refresh_token_encrypted = refresh_token
                    if "expires_in" in new_tokens:
                        db_connection.token_expires_at = datetime.now(timezone.utc) + timedelta(
                            seconds=new_tokens["expires_in"]
                        )

            # Get broker client
            integration_type = IntegrationBrokerType(broker_type.value)
            config = BrokerConfig(
                broker_type=integration_type,
                client_id=getattr(settings, f"{broker_type.value.upper()}_CLIENT_ID", ""),
                client_secret=getattr(settings, f"{broker_type.value.upper()}_CLIENT_SECRET", ""),
                redirect_uri=f"{settings.API_BASE_URL}/api/v1/brokers/callback/{broker_type.value}",
                sandbox=settings.BROKER_SANDBOX_MODE,
            )

            client = get_broker_client(integration_type, config)
            client.set_tokens(access_token=access_token, refresh_token=refresh_token)

            # Fetch positions from broker
            positions = await client.get_positions(connection["broker_account_id"])

            # Get user's portfolio
            portfolio_result = await session.execute(
                select(Portfolio)
                .where(Portfolio.user_id == user_id)
                .options(selectinload(Portfolio.assets))
            )
            portfolio = portfolio_result.scalar_one_or_none()

            positions_synced = 0
            positions_updated = 0
            positions_added = 0

            if portfolio and positions:
                # Create a map of existing assets by ticker
                existing_assets = {
                    asset.ticker.upper(): asset
                    for asset in portfolio.assets
                }

                for position in positions:
                    ticker = position.ticker.upper()

                    if ticker in existing_assets:
                        # Update existing asset
                        asset = existing_assets[ticker]

                        # Only update if broker has more recent data
                        if position.quantity != asset.quantity or position.average_price != asset.average_price:
                            asset.quantity = Decimal(str(position.quantity))
                            asset.average_price = Decimal(str(position.average_price))
                            asset.updated_at = datetime.now(timezone.utc)
                            positions_updated += 1
                    else:
                        # Add new asset from broker
                        new_asset = PortfolioAsset(
                            portfolio_id=portfolio.id,
                            ticker=ticker,
                            quantity=Decimal(str(position.quantity)),
                            average_price=Decimal(str(position.average_price)),
                            asset_type=_map_asset_class(position.asset_class),
                            notes=f"Importado via {broker_type.value}",
                        )
                        session.add(new_asset)
                        positions_added += 1

                    positions_synced += 1

            # Update sync history
            sync_history.status = SyncStatus.SUCCESS
            sync_history.records_synced = positions_synced
            sync_history.completed_at = datetime.now(timezone.utc)

            # Update connection
            conn_result = await session.execute(
                select(BrokerConnection).where(BrokerConnection.id == connection_id)
            )
            db_connection = conn_result.scalar_one_or_none()
            if db_connection:
                db_connection.last_sync_at = datetime.now(timezone.utc)
                db_connection.last_error = None

            await session.commit()

            logger.info(
                "Broker sync completed",
                connection_id=connection_id,
                user_id=user_id,
                broker_type=broker_type.value,
                positions_synced=positions_synced,
                positions_updated=positions_updated,
                positions_added=positions_added,
            )

            return {
                "connection_id": connection_id,
                "user_id": user_id,
                "broker_type": broker_type.value,
                "status": "success",
                "positions_synced": positions_synced,
                "positions_updated": positions_updated,
                "positions_added": positions_added,
            }

        except Exception as e:
            # Update sync history with error
            sync_history.status = SyncStatus.FAILED
            sync_history.error_message = str(e)
            sync_history.completed_at = datetime.now(timezone.utc)

            # Update connection with error
            conn_result = await session.execute(
                select(BrokerConnection).where(BrokerConnection.id == connection_id)
            )
            db_connection = conn_result.scalar_one_or_none()
            if db_connection:
                db_connection.last_error = str(e)

            await session.commit()

            logger.error(
                "Broker sync failed",
                connection_id=connection_id,
                user_id=user_id,
                broker_type=broker_type.value,
                error=str(e),
            )

            return {
                "connection_id": connection_id,
                "user_id": user_id,
                "broker_type": broker_type.value,
                "status": "failed",
                "error": str(e),
            }


def _map_asset_class(broker_asset_class: str | None) -> str:
    """Map broker asset class to our asset type."""
    mapping = {
        "stock": "stock",
        "stocks": "stock",
        "equity": "stock",
        "fii": "fii",
        "real_estate": "fii",
        "etf": "etf",
        "fund": "fund",
        "fixed_income": "fixed_income",
        "renda_fixa": "fixed_income",
        "crypto": "crypto",
        "bdr": "bdr",
        "option": "option",
    }

    if broker_asset_class:
        return mapping.get(broker_asset_class.lower(), "stock")
    return "stock"


async def _sync_all_brokers_async() -> dict:
    """Sync all active broker connections."""
    connections = await _get_active_connections()

    if not connections:
        logger.info("No active broker connections to sync")
        return {"synced": 0, "connections": []}

    logger.info("Starting broker sync", connections_count=len(connections))

    results = []
    successful = 0
    failed = 0

    for connection in connections:
        result = await _sync_connection_positions(connection)
        results.append(result)

        if result.get("status") == "success":
            successful += 1
        else:
            failed += 1

    return {
        "synced": successful,
        "failed": failed,
        "total": len(connections),
        "results": results,
    }


@celery_app.task(bind=True, max_retries=2)
def sync_all_broker_connections(self):
    """
    Sync positions from all active broker connections.

    This task runs periodically to keep portfolio data
    synchronized with broker accounts.
    """
    try:
        logger.info("Starting scheduled broker sync...")
        result = run_async(_sync_all_brokers_async())
        logger.info(
            "Broker sync completed",
            synced=result["synced"],
            failed=result["failed"],
            total=result["total"],
        )
        return result
    except Exception as exc:
        logger.error("Failed to sync broker connections", error=str(exc))
        raise self.retry(exc=exc, countdown=300)  # Retry in 5 minutes


@celery_app.task(bind=True, max_retries=3)
def sync_single_broker_connection(self, connection_id: int, user_id: int):
    """
    Sync positions from a single broker connection.

    This task can be triggered manually by users or after
    OAuth callback completes.
    """
    try:
        logger.info(
            "Starting single broker sync",
            connection_id=connection_id,
            user_id=user_id,
        )

        async def _sync_single():
            session_factory = get_celery_async_session()
            async with session_factory() as session:
                result = await session.execute(
                    select(BrokerConnection).where(
                        BrokerConnection.id == connection_id,
                        BrokerConnection.user_id == user_id,
                    )
                )
                connection = result.scalar_one_or_none()

                if not connection:
                    return {"status": "error", "error": "Connection not found"}

                if connection.status != ConnectionStatus.ACTIVE:
                    return {"status": "error", "error": "Connection is not active"}

                conn_dict = {
                    "id": connection.id,
                    "user_id": connection.user_id,
                    "broker_type": connection.broker_type,
                    "access_token": connection.access_token_encrypted,
                    "refresh_token": connection.refresh_token_encrypted,
                    "token_expires_at": connection.token_expires_at,
                    "broker_account_id": connection.broker_account_id,
                    "last_sync_at": connection.last_sync_at,
                }

                return await _sync_connection_positions(conn_dict)

        result = run_async(_sync_single())

        logger.info(
            "Single broker sync completed",
            connection_id=connection_id,
            result_status=result.get("status"),
        )
        return result

    except Exception as exc:
        logger.error(
            "Failed to sync single broker connection",
            connection_id=connection_id,
            error=str(exc),
        )
        raise self.retry(exc=exc, countdown=60)


@celery_app.task
def check_broker_token_expiry():
    """
    Check for broker connections with expiring tokens
    and attempt to refresh them proactively.
    """
    async def _check_tokens():
        session_factory = get_celery_async_session()
        async with session_factory() as session:
            # Find connections with tokens expiring in the next 30 minutes
            expiry_threshold = datetime.now(timezone.utc) + timedelta(minutes=30)

            result = await session.execute(
                select(BrokerConnection).where(
                    BrokerConnection.status == ConnectionStatus.ACTIVE,
                    BrokerConnection.token_expires_at <= expiry_threshold,
                    BrokerConnection.token_expires_at > datetime.now(timezone.utc),
                )
            )
            connections = result.scalars().all()

            refreshed = 0
            failed = 0

            for conn in connections:
                conn_dict = {
                    "id": conn.id,
                    "user_id": conn.user_id,
                    "broker_type": conn.broker_type,
                    "access_token": conn.access_token_encrypted,
                    "refresh_token": conn.refresh_token_encrypted,
                    "token_expires_at": conn.token_expires_at,
                }

                try:
                    new_tokens = await _refresh_token_if_needed(conn_dict)

                    if new_tokens:
                        conn.access_token_encrypted = new_tokens.get("access_token")
                        conn.refresh_token_encrypted = new_tokens.get("refresh_token")
                        if "expires_in" in new_tokens:
                            conn.token_expires_at = datetime.now(timezone.utc) + timedelta(
                                seconds=new_tokens["expires_in"]
                            )
                        refreshed += 1
                except Exception as e:
                    logger.warning(
                        "Failed to refresh expiring token",
                        connection_id=conn.id,
                        error=str(e),
                    )
                    failed += 1

            await session.commit()

            return {"refreshed": refreshed, "failed": failed, "checked": len(connections)}

    logger.info("Checking broker token expiry...")
    result = run_async(_check_tokens())
    logger.info("Token expiry check completed", **result)
    return result
