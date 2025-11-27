"""Portfolio service."""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import structlog
from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.portfolio import Portfolio, PortfolioAsset
from src.models.market import PortfolioSnapshot, PriceHistory
from src.models.user import User
from src.schemas.portfolio import (
    AllocationByClass,
    AssetClass,
    PerformanceDataPoint,
    PortfolioAssetCreate,
    PortfolioAssetResponse,
    PortfolioPerformanceResponse,
    PortfolioResponse,
    PortfolioSummary,
    RebalanceSimulationResponse,
    RebalanceSuggestion,
)
from src.services.market import market_service
from src.utils.financial_calculations import (
    calculate_returns,
    calculate_annualized_return,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    get_period_days,
    get_period_start_date,
)

logger = structlog.get_logger()


class PortfolioService:
    """Service for portfolio operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_portfolio_by_user(self, user: User) -> Portfolio | None:
        """Get user's portfolio with assets loaded."""
        result = await self.db.execute(
            select(Portfolio)
            .where(Portfolio.user_id == user.id)
            .options(selectinload(Portfolio.assets))
        )
        return result.scalar_one_or_none()

    async def get_full_portfolio(self, user: User) -> PortfolioResponse:
        """Get complete portfolio with current values and allocation."""
        portfolio = await self.get_portfolio_by_user(user)

        if not portfolio or not portfolio.assets:
            # Return empty portfolio
            return PortfolioResponse(
                summary=PortfolioSummary(
                    total_invested=Decimal("0"),
                    current_value=Decimal("0"),
                    total_profit_loss=Decimal("0"),
                    total_profit_loss_percentage=0.0,
                    assets_count=0,
                ),
                assets=[],
                allocation_by_class=[],
            )

        # Get current prices for all tickers
        tickers = [asset.ticker for asset in portfolio.assets]
        quotes = await market_service.get_quotes(tickers)

        # Calculate values for each asset
        assets_response: list[PortfolioAssetResponse] = []
        total_invested = Decimal("0")
        total_current_value = Decimal("0")
        allocation_by_class: dict[AssetClass, dict] = defaultdict(
            lambda: {"value": Decimal("0"), "count": 0}
        )

        for asset in portfolio.assets:
            invested = asset.quantity * asset.average_price
            total_invested += invested

            # Get current price from quotes
            quote = quotes.get(asset.ticker.upper())
            current_price = quote.current_price if quote else None
            current_value = (
                asset.quantity * current_price if current_price else None
            )

            if current_value:
                total_current_value += current_value

            profit_loss = current_value - invested if current_value else None
            profit_loss_pct = (
                float((profit_loss / invested) * 100) if profit_loss and invested > 0 else None
            )

            assets_response.append(
                PortfolioAssetResponse(
                    id=asset.id,
                    ticker=asset.ticker,
                    asset_class=asset.asset_class,
                    quantity=asset.quantity,
                    average_price=asset.average_price,
                    broker=asset.broker,
                    current_price=current_price,
                    current_value=current_value,
                    total_invested=invested,
                    profit_loss=profit_loss,
                    profit_loss_percentage=profit_loss_pct,
                    weight_in_portfolio=None,  # Calculated below
                    created_at=asset.created_at,
                )
            )

            # Track allocation by class
            class_value = current_value if current_value else invested
            allocation_by_class[asset.asset_class]["value"] += class_value
            allocation_by_class[asset.asset_class]["count"] += 1

        # Calculate weights and allocation percentages
        if total_current_value > 0:
            for asset_resp in assets_response:
                if asset_resp.current_value:
                    asset_resp.weight_in_portfolio = float(
                        (asset_resp.current_value / total_current_value) * 100
                    )

        # Build allocation list
        allocation_list: list[AllocationByClass] = []
        total_for_allocation = (
            total_current_value if total_current_value > 0 else total_invested
        )
        for asset_class, data in allocation_by_class.items():
            percentage = (
                float((data["value"] / total_for_allocation) * 100)
                if total_for_allocation > 0
                else 0.0
            )
            allocation_list.append(
                AllocationByClass(
                    asset_class=asset_class,
                    value=data["value"],
                    percentage=percentage,
                    count=data["count"],
                )
            )

        # Sort by value descending
        allocation_list.sort(key=lambda x: x.value, reverse=True)

        # Calculate summary
        total_profit_loss = total_current_value - total_invested
        total_profit_loss_pct = (
            float((total_profit_loss / total_invested) * 100)
            if total_invested > 0
            else 0.0
        )

        return PortfolioResponse(
            summary=PortfolioSummary(
                total_invested=total_invested,
                current_value=total_current_value if total_current_value > 0 else total_invested,
                total_profit_loss=total_profit_loss,
                total_profit_loss_percentage=total_profit_loss_pct,
                assets_count=len(portfolio.assets),
            ),
            assets=assets_response,
            allocation_by_class=allocation_list,
        )

    async def add_asset(
        self, user: User, asset_data: PortfolioAssetCreate
    ) -> PortfolioAsset:
        """Add asset to user's portfolio."""
        portfolio = await self.get_portfolio_by_user(user)

        if not portfolio:
            raise ValueError("Portfolio not found")

        # Check if asset already exists in portfolio
        existing_asset = await self._get_asset_by_ticker(
            portfolio.id, asset_data.ticker
        )

        if existing_asset:
            # Update existing asset with weighted average price
            total_quantity = existing_asset.quantity + asset_data.quantity
            total_value = (
                existing_asset.quantity * existing_asset.average_price
                + asset_data.quantity * asset_data.average_price
            )
            new_average_price = total_value / total_quantity

            existing_asset.quantity = total_quantity
            existing_asset.average_price = new_average_price
            existing_asset.updated_at = datetime.now(timezone.utc)

            await self.db.commit()
            await self.db.refresh(existing_asset)

            logger.info(
                "Asset updated in portfolio",
                user_id=user.id,
                ticker=asset_data.ticker,
                new_quantity=float(total_quantity),
            )

            return existing_asset

        # Try to get asset name from market data
        asset_name = None
        try:
            quote = await market_service.get_quote(asset_data.ticker)
            if quote:
                asset_name = quote.name
        except Exception as e:
            logger.warning("Could not fetch asset name", ticker=asset_data.ticker, error=str(e))

        # Create new asset
        new_asset = PortfolioAsset(
            portfolio_id=portfolio.id,
            ticker=asset_data.ticker.upper(),
            name=asset_name,
            asset_class=asset_data.asset_class,
            quantity=asset_data.quantity,
            average_price=asset_data.average_price,
            broker=asset_data.broker,
        )

        self.db.add(new_asset)
        await self.db.commit()
        await self.db.refresh(new_asset)

        logger.info(
            "Asset added to portfolio",
            user_id=user.id,
            ticker=asset_data.ticker,
            quantity=float(asset_data.quantity),
        )

        return new_asset

    async def remove_asset(self, user: User, asset_id: int) -> bool:
        """Remove asset from user's portfolio."""
        portfolio = await self.get_portfolio_by_user(user)

        if not portfolio:
            raise ValueError("Portfolio not found")

        # Find the asset
        result = await self.db.execute(
            select(PortfolioAsset).where(
                PortfolioAsset.id == asset_id,
                PortfolioAsset.portfolio_id == portfolio.id,
            )
        )
        asset = result.scalar_one_or_none()

        if not asset:
            return False

        await self.db.delete(asset)
        await self.db.commit()

        logger.info(
            "Asset removed from portfolio",
            user_id=user.id,
            asset_id=asset_id,
            ticker=asset.ticker,
        )

        return True

    async def get_performance(
        self, user: User, period: str = "1y"
    ) -> PortfolioPerformanceResponse:
        """Get portfolio performance metrics with financial calculations."""
        portfolio = await self.get_portfolio_by_user(user)

        if not portfolio or not portfolio.assets:
            return PortfolioPerformanceResponse(
                period=period,
                start_value=Decimal("0"),
                end_value=Decimal("0"),
                total_return=Decimal("0"),
                total_return_percentage=0.0,
                annualized_return=None,
                volatility=None,
                sharpe_ratio=None,
                max_drawdown=None,
                history=[],
            )

        # Get current portfolio value
        tickers = [asset.ticker for asset in portfolio.assets]
        quotes = await market_service.get_quotes(tickers)

        total_invested = Decimal("0")
        total_current_value = Decimal("0")

        for asset in portfolio.assets:
            invested = asset.quantity * asset.average_price
            total_invested += invested

            quote = quotes.get(asset.ticker.upper())
            if quote:
                total_current_value += asset.quantity * quote.current_price
            else:
                total_current_value += invested  # Fallback to invested value

        total_return = total_current_value - total_invested
        total_return_pct = (
            float((total_return / total_invested) * 100)
            if total_invested > 0
            else 0.0
        )

        # Get portfolio snapshots for the period
        period_days = get_period_days(period)
        start_date = get_period_start_date(period)

        snapshots = await self._get_portfolio_snapshots(portfolio.id, start_date)

        # Calculate metrics from snapshots
        annualized_return = None
        volatility = None
        sharpe_ratio = None
        max_drawdown = None
        history: list[PerformanceDataPoint] = []

        if snapshots and len(snapshots) >= 2:
            # Extract values for calculations
            values = [float(s.total_value) for s in snapshots]
            returns = calculate_returns(values)

            # Calculate portfolio age in days
            first_snapshot = snapshots[0]
            days_active = (datetime.now(timezone.utc) - first_snapshot.date.replace(tzinfo=timezone.utc)).days

            # Calculate all metrics
            if days_active > 0:
                annualized_return = calculate_annualized_return(
                    snapshots[0].total_value,
                    snapshots[-1].total_value,
                    days_active
                )

            if returns:
                volatility = calculate_volatility(returns)
                sharpe_ratio = calculate_sharpe_ratio(returns)

            max_drawdown = calculate_max_drawdown(values)

            # Build history
            history = [
                PerformanceDataPoint(
                    date=s.date,
                    value=s.total_value,
                    invested=s.total_invested,
                )
                for s in snapshots
            ]
        else:
            # No historical data - use portfolio creation date for basic calculation
            if portfolio.created_at:
                days_active = (datetime.now(timezone.utc) - portfolio.created_at.replace(tzinfo=timezone.utc)).days
                if days_active > 0:
                    annualized_return = calculate_annualized_return(
                        total_invested,
                        total_current_value,
                        days_active
                    )

            # Add current point to history
            history = [
                PerformanceDataPoint(
                    date=datetime.now(timezone.utc),
                    value=total_current_value,
                    invested=total_invested,
                )
            ]

        return PortfolioPerformanceResponse(
            period=period,
            start_value=total_invested,
            end_value=total_current_value,
            total_return=total_return,
            total_return_percentage=total_return_pct,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            history=history,
        )

    async def _get_portfolio_snapshots(
        self, portfolio_id: int, start_date: datetime
    ) -> list[PortfolioSnapshot]:
        """Get portfolio snapshots from a start date."""
        result = await self.db.execute(
            select(PortfolioSnapshot)
            .where(
                and_(
                    PortfolioSnapshot.portfolio_id == portfolio_id,
                    PortfolioSnapshot.date >= start_date,
                )
            )
            .order_by(PortfolioSnapshot.date)
        )
        return list(result.scalars().all())

    async def create_snapshot(self, portfolio_id: int) -> PortfolioSnapshot | None:
        """Create a daily snapshot of portfolio value."""
        # Get portfolio with assets
        result = await self.db.execute(
            select(Portfolio)
            .where(Portfolio.id == portfolio_id)
            .options(selectinload(Portfolio.assets))
        )
        portfolio = result.scalar_one_or_none()

        if not portfolio or not portfolio.assets:
            return None

        # Calculate current values
        tickers = [asset.ticker for asset in portfolio.assets]
        quotes = await market_service.get_quotes(tickers)

        total_invested = Decimal("0")
        total_current_value = Decimal("0")

        for asset in portfolio.assets:
            invested = asset.quantity * asset.average_price
            total_invested += invested

            quote = quotes.get(asset.ticker.upper())
            if quote:
                total_current_value += asset.quantity * quote.current_price
            else:
                total_current_value += invested

        # Calculate daily return if previous snapshot exists
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday = today - timedelta(days=1)

        prev_result = await self.db.execute(
            select(PortfolioSnapshot)
            .where(
                and_(
                    PortfolioSnapshot.portfolio_id == portfolio_id,
                    PortfolioSnapshot.date >= yesterday,
                    PortfolioSnapshot.date < today,
                )
            )
            .order_by(desc(PortfolioSnapshot.date))
            .limit(1)
        )
        prev_snapshot = prev_result.scalar_one_or_none()

        daily_return = None
        if prev_snapshot and prev_snapshot.total_value > 0:
            daily_return = (total_current_value - prev_snapshot.total_value) / prev_snapshot.total_value

        # Create or update today's snapshot
        existing_result = await self.db.execute(
            select(PortfolioSnapshot)
            .where(
                and_(
                    PortfolioSnapshot.portfolio_id == portfolio_id,
                    PortfolioSnapshot.date >= today,
                )
            )
        )
        existing = existing_result.scalar_one_or_none()

        if existing:
            existing.total_value = total_current_value
            existing.total_invested = total_invested
            existing.daily_return = daily_return
            snapshot = existing
        else:
            snapshot = PortfolioSnapshot(
                portfolio_id=portfolio_id,
                date=today,
                total_value=total_current_value,
                total_invested=total_invested,
                daily_return=daily_return,
            )
            self.db.add(snapshot)

        await self.db.commit()
        await self.db.refresh(snapshot)

        logger.info(
            "Portfolio snapshot created",
            portfolio_id=portfolio_id,
            value=float(total_current_value),
            daily_return=float(daily_return) if daily_return else None,
        )

        return snapshot

    async def simulate_rebalance(
        self, user: User, target_allocation: dict[AssetClass, float] | None = None
    ) -> RebalanceSimulationResponse:
        """Simulate portfolio rebalancing."""
        portfolio_data = await self.get_full_portfolio(user)

        if not portfolio_data.assets:
            return RebalanceSimulationResponse(
                current_allocation=portfolio_data.allocation_by_class,
                target_allocation=[],
                suggestions=[],
                estimated_cost=Decimal("0"),
            )

        # Default target allocation based on risk profile
        # In a full implementation, this would come from the investor profile
        if target_allocation is None:
            target_allocation = {
                AssetClass.STOCKS: 40.0,
                AssetClass.FIIS: 30.0,
                AssetClass.FIXED_INCOME: 20.0,
                AssetClass.CASH: 10.0,
            }

        # Build target allocation list
        total_value = portfolio_data.summary.current_value
        target_allocation_list: list[AllocationByClass] = []

        for asset_class, target_pct in target_allocation.items():
            target_value = total_value * Decimal(str(target_pct / 100))
            target_allocation_list.append(
                AllocationByClass(
                    asset_class=asset_class,
                    value=target_value,
                    percentage=target_pct,
                    count=0,
                )
            )

        # Calculate suggestions
        suggestions: list[RebalanceSuggestion] = []
        estimated_cost = Decimal("0")

        current_by_class = {
            alloc.asset_class: alloc for alloc in portfolio_data.allocation_by_class
        }

        for target in target_allocation_list:
            current = current_by_class.get(target.asset_class)
            current_pct = current.percentage if current else 0.0
            diff_pct = target.percentage - current_pct

            if abs(diff_pct) >= 5:  # Only suggest if difference is >= 5%
                action = "buy" if diff_pct > 0 else "sell"
                diff_value = total_value * Decimal(str(abs(diff_pct) / 100))

                suggestions.append(
                    RebalanceSuggestion(
                        ticker=target.asset_class.value,  # Placeholder - would need specific ticker
                        action=action,
                        quantity=diff_value,  # Value to invest/divest
                        reason=f"Ajustar alocação de {current_pct:.1f}% para {target.percentage:.1f}%",
                    )
                )

                if action == "buy":
                    estimated_cost += diff_value

        return RebalanceSimulationResponse(
            current_allocation=portfolio_data.allocation_by_class,
            target_allocation=target_allocation_list,
            suggestions=suggestions,
            estimated_cost=estimated_cost,
        )

    async def _get_asset_by_ticker(
        self, portfolio_id: int, ticker: str
    ) -> PortfolioAsset | None:
        """Get asset by ticker from portfolio."""
        result = await self.db.execute(
            select(PortfolioAsset).where(
                PortfolioAsset.portfolio_id == portfolio_id,
                PortfolioAsset.ticker == ticker.upper(),
            )
        )
        return result.scalar_one_or_none()

    async def get_asset_by_ticker(
        self, user: User, ticker: str
    ) -> PortfolioAsset | None:
        """Get asset by ticker for user's portfolio."""
        portfolio = await self.get_portfolio_by_user(user)
        if not portfolio:
            return None
        return await self._get_asset_by_ticker(portfolio.id, ticker)

    async def get_asset_response(
        self, asset: PortfolioAsset, total_portfolio_value: Decimal | None = None
    ) -> PortfolioAssetResponse:
        """Convert PortfolioAsset to PortfolioAssetResponse with current prices."""
        invested = asset.quantity * asset.average_price

        # Get current price
        quote = await market_service.get_quote(asset.ticker)
        current_price = quote.current_price if quote else None
        current_value = asset.quantity * current_price if current_price else None

        profit_loss = current_value - invested if current_value else None
        profit_loss_pct = (
            float((profit_loss / invested) * 100) if profit_loss and invested > 0 else None
        )

        weight = None
        if total_portfolio_value and current_value:
            weight = float((current_value / total_portfolio_value) * 100)

        return PortfolioAssetResponse(
            id=asset.id,
            ticker=asset.ticker,
            asset_class=asset.asset_class,
            quantity=asset.quantity,
            average_price=asset.average_price,
            broker=asset.broker,
            current_price=current_price,
            current_value=current_value,
            total_invested=invested,
            profit_loss=profit_loss,
            profit_loss_percentage=profit_loss_pct,
            weight_in_portfolio=weight,
            created_at=asset.created_at,
        )
