"""Broker integration endpoints."""
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status
import structlog

from src.core.deps import CurrentUser, DbSession
from src.models.broker import BrokerType, ConnectionStatus, SyncStatus
from src.schemas.broker import (
    BrokerConnectRequest,
    BrokerCallbackRequest,
    BrokerAuthUrlResponse,
    BrokerConnectionResponse,
    BrokerConnectionListResponse,
    BrokerSyncHistoryResponse,
    BrokerSyncResponse,
    BrokerPositionResponse,
    BrokerTransactionResponse,
    BrokerPortfolioResponse,
    SupportedBrokerResponse,
    SupportedBrokersListResponse,
)
from src.services.broker import BrokerService

router = APIRouter()
logger = structlog.get_logger()


@router.get("/supported", response_model=SupportedBrokersListResponse)
async def list_supported_brokers(
    db: DbSession,
) -> SupportedBrokersListResponse:
    """List all supported brokers."""
    service = BrokerService(db)
    brokers = service.get_supported_brokers()

    return SupportedBrokersListResponse(
        brokers=[
            SupportedBrokerResponse(
                broker_type=b["broker_type"],
                name=b["name"],
                description=b["description"],
                documentation_url=b["documentation_url"],
                features=b["features"],
                requires_mtls=b["requires_mtls"],
                auth_type=b["auth_type"],
            )
            for b in brokers
        ]
    )


@router.get("/connections", response_model=BrokerConnectionListResponse)
async def list_connections(
    current_user: CurrentUser,
    db: DbSession,
) -> BrokerConnectionListResponse:
    """List all broker connections for current user."""
    service = BrokerService(db)
    connections = await service.get_user_connections(current_user.id)

    return BrokerConnectionListResponse(
        connections=[
            BrokerConnectionResponse(
                id=c.id,
                broker_type=c.broker_type,
                status=c.status,
                broker_account_id=c.broker_account_id,
                broker_account_name=c.broker_account_name,
                last_sync_at=c.last_sync_at,
                last_error=c.last_error,
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
            for c in connections
        ],
        total=len(connections),
    )


@router.get("/connections/{connection_id}", response_model=BrokerConnectionResponse)
async def get_connection(
    connection_id: int,
    current_user: CurrentUser,
    db: DbSession,
) -> BrokerConnectionResponse:
    """Get a specific broker connection."""
    service = BrokerService(db)
    connection = await service.get_connection(current_user.id, connection_id)

    if not connection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conexao nao encontrada.",
        )

    return BrokerConnectionResponse(
        id=connection.id,
        broker_type=connection.broker_type,
        status=connection.status,
        broker_account_id=connection.broker_account_id,
        broker_account_name=connection.broker_account_name,
        last_sync_at=connection.last_sync_at,
        last_error=connection.last_error,
        created_at=connection.created_at,
        updated_at=connection.updated_at,
    )


@router.post("/connect", response_model=BrokerAuthUrlResponse)
async def connect_broker(
    request: BrokerConnectRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> BrokerAuthUrlResponse:
    """Start OAuth flow to connect a broker."""
    service = BrokerService(db)

    try:
        auth_url, state = await service.start_connection_flow(
            current_user.id, BrokerType(request.broker_type.value)
        )

        logger.info(
            "Broker connection flow started",
            user_id=current_user.id,
            broker_type=request.broker_type.value,
        )

        return BrokerAuthUrlResponse(
            authorization_url=auth_url,
            state=state,
        )

    except Exception as e:
        logger.error(
            "Failed to start broker connection",
            user_id=current_user.id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao iniciar conexao: {str(e)}",
        )


@router.post("/callback/{broker_type}", response_model=BrokerConnectionResponse)
async def broker_callback(
    broker_type: str,
    request: BrokerCallbackRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> BrokerConnectionResponse:
    """Handle OAuth callback from broker."""
    service = BrokerService(db)

    try:
        connection = await service.complete_connection(
            user_id=current_user.id,
            code=request.code,
            state=request.state,
        )

        return BrokerConnectionResponse(
            id=connection.id,
            broker_type=connection.broker_type,
            status=connection.status,
            broker_account_id=connection.broker_account_id,
            broker_account_name=connection.broker_account_name,
            last_sync_at=connection.last_sync_at,
            last_error=connection.last_error,
            created_at=connection.created_at,
            updated_at=connection.updated_at,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            "Failed to complete broker connection",
            user_id=current_user.id,
            broker_type=broker_type,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao completar conexao: {str(e)}",
        )


@router.delete("/connections/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def disconnect_broker(
    connection_id: int,
    current_user: CurrentUser,
    db: DbSession,
) -> None:
    """Disconnect a broker."""
    service = BrokerService(db)

    try:
        await service.disconnect_broker(current_user.id, connection_id)

        logger.info(
            "Broker disconnected",
            user_id=current_user.id,
            connection_id=connection_id,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post("/connections/{connection_id}/sync", response_model=BrokerSyncResponse)
async def sync_broker_positions(
    connection_id: int,
    current_user: CurrentUser,
    db: DbSession,
) -> BrokerSyncResponse:
    """Trigger a sync of positions from broker."""
    service = BrokerService(db)

    try:
        sync_history = await service.sync_positions(current_user.id, connection_id)

        return BrokerSyncResponse(
            sync_id=sync_history.id,
            status=sync_history.status,
            message=f"Sincronizacao concluida: {sync_history.records_synced} posicoes sincronizadas.",
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            "Failed to sync positions",
            user_id=current_user.id,
            connection_id=connection_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao sincronizar posicoes: {str(e)}",
        )


@router.get(
    "/connections/{connection_id}/sync-history",
    response_model=list[BrokerSyncHistoryResponse],
)
async def get_sync_history(
    connection_id: int,
    current_user: CurrentUser,
    db: DbSession,
    limit: int = Query(10, ge=1, le=100),
) -> list[BrokerSyncHistoryResponse]:
    """Get sync history for a connection."""
    service = BrokerService(db)

    try:
        history = await service.get_sync_history(
            current_user.id, connection_id, limit
        )

        return [
            BrokerSyncHistoryResponse(
                id=h.id,
                sync_type=h.sync_type,
                status=h.status,
                records_synced=h.records_synced,
                records_created=h.records_created,
                records_updated=h.records_updated,
                error_message=h.error_message,
                started_at=h.started_at,
                completed_at=h.completed_at,
            )
            for h in history
        ]

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.get(
    "/connections/{connection_id}/positions",
    response_model=list[BrokerPositionResponse],
)
async def get_broker_positions(
    connection_id: int,
    current_user: CurrentUser,
    db: DbSession,
) -> list[BrokerPositionResponse]:
    """Get current positions from connected broker."""
    service = BrokerService(db)

    try:
        positions = await service.get_broker_positions(current_user.id, connection_id)

        return [
            BrokerPositionResponse(
                ticker=p["ticker"],
                name=p["name"],
                quantity=p["quantity"],
                average_price=p["average_price"],
                current_price=p["current_price"],
                current_value=p["current_value"],
                profit_loss=p["profit_loss"],
                asset_class=p["asset_class"],
            )
            for p in positions
        ]

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            "Failed to get broker positions",
            user_id=current_user.id,
            connection_id=connection_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao buscar posicoes: {str(e)}",
        )


@router.get(
    "/connections/{connection_id}/transactions",
    response_model=list[BrokerTransactionResponse],
)
async def get_broker_transactions(
    connection_id: int,
    current_user: CurrentUser,
    db: DbSession,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> list[BrokerTransactionResponse]:
    """Get transactions from connected broker."""
    service = BrokerService(db)

    try:
        transactions = await service.get_broker_transactions(
            user_id=current_user.id,
            connection_id=connection_id,
            start_date=start_date,
            end_date=end_date,
        )

        return [
            BrokerTransactionResponse(
                transaction_id=t["transaction_id"],
                ticker=t["ticker"],
                transaction_type=t["transaction_type"],
                quantity=t["quantity"],
                price=t["price"],
                total_value=t["total_value"],
                fees=t["fees"],
                transaction_date=t["transaction_date"],
                settlement_date=t["settlement_date"],
            )
            for t in transactions
        ]

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            "Failed to get broker transactions",
            user_id=current_user.id,
            connection_id=connection_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao buscar transacoes: {str(e)}",
        )


@router.post("/connections/{connection_id}/import-to-portfolio", response_model=BrokerPortfolioResponse)
async def import_broker_to_portfolio(
    connection_id: int,
    current_user: CurrentUser,
    db: DbSession,
) -> BrokerPortfolioResponse:
    """Import positions from broker into InvestAI portfolio."""
    from decimal import Decimal
    from sqlalchemy import select
    from src.models.portfolio import Portfolio, PortfolioAsset

    service = BrokerService(db)

    try:
        # Get positions from broker
        positions = await service.get_broker_positions(current_user.id, connection_id)

        if not positions:
            return BrokerPortfolioResponse(
                imported=0,
                updated=0,
                skipped=0,
                message="Nenhuma posicao encontrada na corretora.",
            )

        # Get or create portfolio
        result = await db.execute(
            select(Portfolio).where(Portfolio.user_id == current_user.id)
        )
        portfolio = result.scalar_one_or_none()

        if not portfolio:
            portfolio = Portfolio(user_id=current_user.id, name="Meu Portfolio")
            db.add(portfolio)
            await db.commit()
            await db.refresh(portfolio)

        imported_count = 0
        updated_count = 0
        skipped_count = 0

        for pos in positions:
            # Check if asset already exists
            existing = await db.execute(
                select(PortfolioAsset).where(
                    PortfolioAsset.portfolio_id == portfolio.id,
                    PortfolioAsset.ticker == pos["ticker"].upper(),
                )
            )
            existing_asset = existing.scalar_one_or_none()

            if existing_asset:
                # Update existing asset
                existing_asset.quantity = Decimal(str(pos["quantity"]))
                existing_asset.average_price = Decimal(str(pos["average_price"]))
                updated_count += 1
            else:
                # Create new asset
                if pos["quantity"] > 0:
                    new_asset = PortfolioAsset(
                        portfolio_id=portfolio.id,
                        ticker=pos["ticker"].upper(),
                        quantity=Decimal(str(pos["quantity"])),
                        average_price=Decimal(str(pos["average_price"])),
                        asset_type=pos.get("asset_class", "stock"),
                    )
                    db.add(new_asset)
                    imported_count += 1
                else:
                    skipped_count += 1

        await db.commit()

        logger.info(
            "Broker positions imported to portfolio",
            user_id=current_user.id,
            connection_id=connection_id,
            imported=imported_count,
            updated=updated_count,
            skipped=skipped_count,
        )

        return BrokerPortfolioResponse(
            imported=imported_count,
            updated=updated_count,
            skipped=skipped_count,
            message=f"Importacao concluida: {imported_count} novos ativos, {updated_count} atualizados, {skipped_count} ignorados.",
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            "Failed to import broker positions",
            user_id=current_user.id,
            connection_id=connection_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao importar posicoes: {str(e)}",
        )
