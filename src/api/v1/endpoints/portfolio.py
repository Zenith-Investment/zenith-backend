"""Portfolio management endpoints with multiple portfolios support."""
import csv
import io
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
import structlog

from src.core.deps import CurrentUser, DbSession
from src.models.portfolio import Portfolio, PortfolioType
from src.schemas.portfolio import (
    AssetClass,
    CSVImportResult,
    CSVTemplateInfo,
    PortfolioAssetCreate,
    PortfolioAssetResponse,
    PortfolioBasicResponse,
    PortfolioCreate,
    PortfolioListResponse,
    PortfolioPerformanceResponse,
    PortfolioResponse,
    PortfolioUpdate,
    RebalanceSimulationResponse,
)
from src.services.portfolio import PortfolioService

router = APIRouter()
logger = structlog.get_logger()


# ===========================================
# Multiple Portfolios Management
# ===========================================

@router.get("/list", response_model=PortfolioListResponse)
async def list_portfolios(
    current_user: CurrentUser,
    db: DbSession,
    portfolio_type: Optional[str] = None,
) -> PortfolioListResponse:
    """List all user's portfolios."""
    query = select(Portfolio).where(Portfolio.user_id == current_user.id)

    if portfolio_type:
        try:
            ptype = PortfolioType(portfolio_type)
            query = query.where(Portfolio.portfolio_type == ptype)
        except ValueError:
            pass

    query = query.order_by(Portfolio.is_primary.desc(), Portfolio.created_at.desc())

    result = await db.execute(query)
    portfolios = result.scalars().all()

    portfolio_responses = []
    for p in portfolios:
        # Calculate total value
        total_value = Decimal("0")
        for asset in p.assets:
            total_value += asset.quantity * asset.average_price

        portfolio_responses.append(PortfolioBasicResponse(
            id=p.id,
            name=p.name,
            description=p.description,
            portfolio_type=p.portfolio_type,
            is_primary=p.is_primary,
            color=p.color,
            icon=p.icon,
            target_value=p.target_value,
            risk_profile=p.risk_profile,
            total_value=total_value,
            assets_count=len(p.assets),
            created_at=p.created_at,
            updated_at=p.updated_at,
        ))

    return PortfolioListResponse(
        portfolios=portfolio_responses,
        total=len(portfolio_responses),
    )


@router.post("/create", response_model=PortfolioBasicResponse, status_code=status.HTTP_201_CREATED)
async def create_portfolio(
    portfolio_data: PortfolioCreate,
    current_user: CurrentUser,
    db: DbSession,
) -> PortfolioBasicResponse:
    """Create a new portfolio."""
    # Check portfolio limit based on subscription
    existing_count = await db.execute(
        select(Portfolio).where(Portfolio.user_id == current_user.id)
    )
    portfolios = existing_count.scalars().all()

    # Get limit based on subscription
    from src.schemas.user import SubscriptionPlan
    limits = {
        SubscriptionPlan.STARTER: 1,
        SubscriptionPlan.SMART: 3,
        SubscriptionPlan.PRO: 10,
        SubscriptionPlan.PREMIUM: 20,
    }
    limit = limits.get(current_user.subscription_plan, 1)

    if len(portfolios) >= limit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Limite de {limit} carteiras atingido para seu plano. Faça upgrade para criar mais.",
        )

    # If this is the first portfolio, make it primary
    is_primary = portfolio_data.is_primary or len(portfolios) == 0

    # If setting as primary, unset other portfolios
    if is_primary:
        for p in portfolios:
            if p.is_primary:
                p.is_primary = False

    new_portfolio = Portfolio(
        user_id=current_user.id,
        name=portfolio_data.name,
        description=portfolio_data.description,
        portfolio_type=portfolio_data.portfolio_type,
        is_primary=is_primary,
        color=portfolio_data.color,
        icon=portfolio_data.icon,
        target_value=portfolio_data.target_value,
        risk_profile=portfolio_data.risk_profile,
    )

    db.add(new_portfolio)
    await db.commit()
    await db.refresh(new_portfolio)

    logger.info("Portfolio created", user_id=current_user.id, portfolio_id=new_portfolio.id)

    return PortfolioBasicResponse(
        id=new_portfolio.id,
        name=new_portfolio.name,
        description=new_portfolio.description,
        portfolio_type=new_portfolio.portfolio_type,
        is_primary=new_portfolio.is_primary,
        color=new_portfolio.color,
        icon=new_portfolio.icon,
        target_value=new_portfolio.target_value,
        risk_profile=new_portfolio.risk_profile,
        total_value=Decimal("0"),
        assets_count=0,
        created_at=new_portfolio.created_at,
        updated_at=new_portfolio.updated_at,
    )


@router.get("/{portfolio_id}", response_model=PortfolioResponse)
async def get_portfolio_by_id(
    portfolio_id: int,
    current_user: CurrentUser,
    db: DbSession,
) -> PortfolioResponse:
    """Get a specific portfolio with full details."""
    portfolio_service = PortfolioService(db)

    try:
        portfolio = await portfolio_service.get_full_portfolio(current_user, portfolio_id)
        logger.info("Portfolio retrieved", user_id=current_user.id, portfolio_id=portfolio_id)
        return portfolio
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Failed to get portfolio", user_id=current_user.id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve portfolio",
        )


@router.put("/{portfolio_id}", response_model=PortfolioBasicResponse)
async def update_portfolio(
    portfolio_id: int,
    portfolio_data: PortfolioUpdate,
    current_user: CurrentUser,
    db: DbSession,
) -> PortfolioBasicResponse:
    """Update a portfolio."""
    result = await db.execute(
        select(Portfolio).where(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == current_user.id,
        )
    )
    portfolio = result.scalar_one_or_none()

    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Carteira não encontrada.",
        )

    # Update fields
    if portfolio_data.name is not None:
        portfolio.name = portfolio_data.name
    if portfolio_data.description is not None:
        portfolio.description = portfolio_data.description
    if portfolio_data.portfolio_type is not None:
        portfolio.portfolio_type = portfolio_data.portfolio_type
    if portfolio_data.color is not None:
        portfolio.color = portfolio_data.color
    if portfolio_data.icon is not None:
        portfolio.icon = portfolio_data.icon
    if portfolio_data.target_value is not None:
        portfolio.target_value = portfolio_data.target_value
    if portfolio_data.risk_profile is not None:
        portfolio.risk_profile = portfolio_data.risk_profile

    # Handle primary flag
    if portfolio_data.is_primary is True and not portfolio.is_primary:
        # Unset other portfolios as primary
        all_portfolios = await db.execute(
            select(Portfolio).where(
                Portfolio.user_id == current_user.id,
                Portfolio.is_primary == True,
            )
        )
        for p in all_portfolios.scalars().all():
            p.is_primary = False
        portfolio.is_primary = True

    await db.commit()
    await db.refresh(portfolio)

    # Calculate total value
    total_value = Decimal("0")
    for asset in portfolio.assets:
        total_value += asset.quantity * asset.average_price

    return PortfolioBasicResponse(
        id=portfolio.id,
        name=portfolio.name,
        description=portfolio.description,
        portfolio_type=portfolio.portfolio_type,
        is_primary=portfolio.is_primary,
        color=portfolio.color,
        icon=portfolio.icon,
        target_value=portfolio.target_value,
        risk_profile=portfolio.risk_profile,
        total_value=total_value,
        assets_count=len(portfolio.assets),
        created_at=portfolio.created_at,
        updated_at=portfolio.updated_at,
    )


@router.delete("/{portfolio_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_portfolio(
    portfolio_id: int,
    current_user: CurrentUser,
    db: DbSession,
) -> None:
    """Delete a portfolio."""
    result = await db.execute(
        select(Portfolio).where(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == current_user.id,
        )
    )
    portfolio = result.scalar_one_or_none()

    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Carteira não encontrada.",
        )

    # Don't allow deleting primary if other portfolios exist
    if portfolio.is_primary:
        other = await db.execute(
            select(Portfolio).where(
                Portfolio.user_id == current_user.id,
                Portfolio.id != portfolio_id,
            )
        )
        other_portfolios = other.scalars().all()
        if other_portfolios:
            # Make another portfolio primary
            other_portfolios[0].is_primary = True

    await db.delete(portfolio)
    await db.commit()

    logger.info("Portfolio deleted", user_id=current_user.id, portfolio_id=portfolio_id)


@router.get("/", response_model=PortfolioResponse)
async def get_portfolio(
    current_user: CurrentUser,
    db: DbSession,
) -> PortfolioResponse:
    """Get user's complete portfolio with current values and allocation."""
    portfolio_service = PortfolioService(db)

    try:
        portfolio = await portfolio_service.get_full_portfolio(current_user)
        logger.info("Portfolio retrieved", user_id=current_user.id)
        return portfolio
    except Exception as e:
        logger.error("Failed to get portfolio", user_id=current_user.id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve portfolio",
        )


@router.post("/assets", response_model=PortfolioAssetResponse, status_code=status.HTTP_201_CREATED)
async def add_asset(
    asset: PortfolioAssetCreate,
    current_user: CurrentUser,
    db: DbSession,
) -> PortfolioAssetResponse:
    """Add asset to portfolio.

    If the asset already exists, it will be updated with a weighted average price.
    """
    portfolio_service = PortfolioService(db)

    try:
        new_asset = await portfolio_service.add_asset(current_user, asset)

        # Get full response with current price
        portfolio = await portfolio_service.get_full_portfolio(current_user)
        total_value = portfolio.summary.current_value

        asset_response = await portfolio_service.get_asset_response(new_asset, total_value)

        logger.info(
            "Asset added to portfolio",
            user_id=current_user.id,
            ticker=asset.ticker,
        )

        return asset_response

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            "Failed to add asset",
            user_id=current_user.id,
            ticker=asset.ticker,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add asset to portfolio",
        )


@router.delete("/assets/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_asset(
    asset_id: int,
    current_user: CurrentUser,
    db: DbSession,
) -> None:
    """Remove asset from portfolio."""
    portfolio_service = PortfolioService(db)

    try:
        removed = await portfolio_service.remove_asset(current_user, asset_id)

        if not removed:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Asset not found in portfolio",
            )

        logger.info(
            "Asset removed from portfolio",
            user_id=current_user.id,
            asset_id=asset_id,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to remove asset",
            user_id=current_user.id,
            asset_id=asset_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove asset from portfolio",
        )


@router.get("/performance", response_model=PortfolioPerformanceResponse)
async def get_performance(
    current_user: CurrentUser,
    db: DbSession,
    period: str = "1y",
) -> PortfolioPerformanceResponse:
    """Get portfolio performance metrics.

    Args:
        period: Time period for performance calculation.
            Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max
    """
    portfolio_service = PortfolioService(db)

    try:
        performance = await portfolio_service.get_performance(current_user, period)
        logger.info(
            "Portfolio performance retrieved",
            user_id=current_user.id,
            period=period,
        )
        return performance

    except Exception as e:
        logger.error(
            "Failed to get portfolio performance",
            user_id=current_user.id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve portfolio performance",
        )


@router.post("/rebalance/simulate", response_model=RebalanceSimulationResponse)
async def simulate_rebalance(
    current_user: CurrentUser,
    db: DbSession,
) -> RebalanceSimulationResponse:
    """Simulate portfolio rebalancing based on target allocation.

    Returns suggestions for buying/selling assets to achieve the target allocation.
    The target allocation is determined by the user's investor profile risk level.
    """
    portfolio_service = PortfolioService(db)

    try:
        simulation = await portfolio_service.simulate_rebalance(current_user)
        logger.info(
            "Rebalance simulation completed",
            user_id=current_user.id,
            suggestions_count=len(simulation.suggestions),
        )
        return simulation

    except Exception as e:
        logger.error(
            "Failed to simulate rebalance",
            user_id=current_user.id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to simulate portfolio rebalancing",
        )


@router.get("/import/template", response_model=CSVTemplateInfo)
async def get_csv_template() -> CSVTemplateInfo:
    """Get CSV template information for portfolio import."""
    return CSVTemplateInfo(
        columns=["ticker", "asset_class", "quantity", "average_price", "broker", "purchase_date"],
        example_row={
            "ticker": "PETR4",
            "asset_class": "stocks",
            "quantity": "100",
            "average_price": "32.50",
            "broker": "XP",
            "purchase_date": "2024-01-15",
        },
        asset_classes=[ac.value for ac in AssetClass],
        notes=(
            "O arquivo CSV deve conter as colunas: ticker, asset_class, quantity, average_price. "
            "As colunas broker e purchase_date sao opcionais. "
            "Use ponto (.) como separador decimal. "
            "Formato de data: YYYY-MM-DD"
        ),
    )


@router.get("/import/template/download")
async def download_csv_template() -> StreamingResponse:
    """Download a CSV template file for portfolio import."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(["ticker", "asset_class", "quantity", "average_price", "broker", "purchase_date"])

    # Example rows
    writer.writerow(["PETR4", "stocks", "100", "32.50", "XP", "2024-01-15"])
    writer.writerow(["VALE3", "stocks", "50", "68.00", "Rico", "2024-02-20"])
    writer.writerow(["HGLG11", "fiis", "10", "160.00", "Clear", "2024-03-10"])
    writer.writerow(["IVVB11", "etf", "20", "280.00", "XP", "2024-04-01"])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=portfolio_template.csv"},
    )


@router.post("/import/csv", response_model=CSVImportResult)
async def import_csv(
    current_user: CurrentUser,
    db: DbSession,
    file: UploadFile = File(...),
) -> CSVImportResult:
    """Import portfolio assets from CSV file.

    Expected columns:
    - ticker: Asset ticker (required)
    - asset_class: Type of asset - stocks, fiis, fixed_income, etf, bdr, crypto, funds, cash, other (required)
    - quantity: Number of shares/units (required)
    - average_price: Average purchase price (required)
    - broker: Broker name (optional)
    - purchase_date: Purchase date in YYYY-MM-DD format (optional)

    If an asset already exists in the portfolio, it will be updated with a weighted average price.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="O arquivo deve ser um CSV (.csv)",
        )

    portfolio_service = PortfolioService(db)
    errors: list[str] = []
    imported = 0
    updated = 0
    imported_assets: list[PortfolioAssetResponse] = []

    try:
        contents = await file.read()
        decoded = contents.decode("utf-8-sig")  # Handle BOM
        reader = csv.DictReader(io.StringIO(decoded))

        rows = list(reader)
        total_rows = len(rows)

        if total_rows == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="O arquivo CSV esta vazio",
            )

        # Validate headers
        required_columns = {"ticker", "asset_class", "quantity", "average_price"}
        if reader.fieldnames:
            missing = required_columns - set(reader.fieldnames)
            if missing:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Colunas obrigatorias ausentes: {', '.join(missing)}",
                )

        for row_num, row in enumerate(rows, start=2):  # Start at 2 (header is row 1)
            try:
                # Parse and validate row
                ticker = row.get("ticker", "").strip().upper()
                if not ticker:
                    errors.append(f"Linha {row_num}: Ticker vazio")
                    continue

                # Parse asset class
                asset_class_str = row.get("asset_class", "").strip().lower()
                try:
                    asset_class = AssetClass(asset_class_str)
                except ValueError:
                    errors.append(f"Linha {row_num}: Classe de ativo invalida '{asset_class_str}'")
                    continue

                # Parse quantity
                try:
                    quantity = Decimal(row.get("quantity", "0").strip().replace(",", "."))
                    if quantity <= 0:
                        errors.append(f"Linha {row_num}: Quantidade deve ser maior que zero")
                        continue
                except InvalidOperation:
                    errors.append(f"Linha {row_num}: Quantidade invalida")
                    continue

                # Parse average price
                try:
                    average_price = Decimal(row.get("average_price", "0").strip().replace(",", "."))
                    if average_price <= 0:
                        errors.append(f"Linha {row_num}: Preco medio deve ser maior que zero")
                        continue
                except InvalidOperation:
                    errors.append(f"Linha {row_num}: Preco medio invalido")
                    continue

                # Parse optional fields
                broker = row.get("broker", "").strip() or None
                purchase_date = None
                date_str = row.get("purchase_date", "").strip()
                if date_str:
                    try:
                        purchase_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        errors.append(f"Linha {row_num}: Formato de data invalido (use YYYY-MM-DD)")
                        continue

                # Create asset
                asset_create = PortfolioAssetCreate(
                    ticker=ticker,
                    asset_class=asset_class,
                    quantity=quantity,
                    average_price=average_price,
                    broker=broker,
                    purchase_date=purchase_date,
                )

                # Check if it's an update or new
                existing = await portfolio_service.get_asset_by_ticker(current_user, ticker)
                is_update = existing is not None

                new_asset = await portfolio_service.add_asset(current_user, asset_create)

                # Get response
                portfolio = await portfolio_service.get_full_portfolio(current_user)
                total_value = portfolio.summary.current_value
                asset_response = await portfolio_service.get_asset_response(new_asset, total_value)
                imported_assets.append(asset_response)

                if is_update:
                    updated += 1
                else:
                    imported += 1

            except Exception as e:
                errors.append(f"Linha {row_num}: {str(e)}")

        logger.info(
            "CSV import completed",
            user_id=current_user.id,
            total=total_rows,
            imported=imported,
            updated=updated,
            errors=len(errors),
        )

        return CSVImportResult(
            success=len(errors) == 0,
            total_rows=total_rows,
            imported=imported,
            updated=updated,
            errors=errors,
            assets=imported_assets,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "CSV import failed",
            user_id=current_user.id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar arquivo CSV: {str(e)}",
        )
