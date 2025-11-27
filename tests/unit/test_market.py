"""Tests for market data functionality."""
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.market import PriceHistory, PortfolioSnapshot
from src.models.portfolio import Portfolio


class TestPriceHistory:
    """Test price history model and queries."""

    @pytest.mark.asyncio
    async def test_create_price_history(self, db_session: AsyncSession):
        """Test creating a price history record."""
        price = PriceHistory(
            ticker="PETR4",
            date=datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0),
            open_price=Decimal("35.00"),
            high_price=Decimal("36.50"),
            low_price=Decimal("34.50"),
            close_price=Decimal("36.00"),
            volume=1000000,
        )
        db_session.add(price)
        await db_session.commit()
        await db_session.refresh(price)

        assert price.id is not None
        assert price.ticker == "PETR4"
        assert price.close_price == Decimal("36.00")

    @pytest.mark.asyncio
    async def test_price_history_multiple_days(self, db_session: AsyncSession):
        """Test storing multiple days of price history."""
        from datetime import timedelta

        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        prices = [
            PriceHistory(
                ticker="VALE3",
                date=today - timedelta(days=i),
                close_price=Decimal(str(65 + i)),
            )
            for i in range(5)
        ]

        for price in prices:
            db_session.add(price)
        await db_session.commit()

        from sqlalchemy import select

        result = await db_session.execute(
            select(PriceHistory).where(PriceHistory.ticker == "VALE3")
        )
        stored_prices = result.scalars().all()

        assert len(stored_prices) == 5


class TestPortfolioSnapshot:
    """Test portfolio snapshot functionality."""

    @pytest.mark.asyncio
    async def test_create_portfolio_snapshot(
        self, db_session: AsyncSession, test_portfolio: Portfolio
    ):
        """Test creating a portfolio snapshot."""
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        snapshot = PortfolioSnapshot(
            portfolio_id=test_portfolio.id,
            date=today,
            total_value=Decimal("10000.00"),
            total_invested=Decimal("9500.00"),
            daily_return=Decimal("0.0526"),
        )
        db_session.add(snapshot)
        await db_session.commit()
        await db_session.refresh(snapshot)

        assert snapshot.id is not None
        assert snapshot.total_value == Decimal("10000.00")
        assert snapshot.daily_return == Decimal("0.0526")

    @pytest.mark.asyncio
    async def test_snapshot_history(
        self, db_session: AsyncSession, test_portfolio: Portfolio
    ):
        """Test storing snapshot history over multiple days."""
        from datetime import timedelta

        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        snapshots = [
            PortfolioSnapshot(
                portfolio_id=test_portfolio.id,
                date=today - timedelta(days=i),
                total_value=Decimal(str(10000 + i * 100)),
                total_invested=Decimal("9500.00"),
            )
            for i in range(30)
        ]

        for snapshot in snapshots:
            db_session.add(snapshot)
        await db_session.commit()

        from sqlalchemy import select

        result = await db_session.execute(
            select(PortfolioSnapshot).where(
                PortfolioSnapshot.portfolio_id == test_portfolio.id
            )
        )
        stored_snapshots = result.scalars().all()

        assert len(stored_snapshots) == 30


class TestMarketCalculations:
    """Test market-related calculations."""

    def test_daily_return_calculation(self):
        """Test calculating daily return percentage."""
        previous_value = Decimal("10000.00")
        current_value = Decimal("10250.00")

        daily_return = (current_value - previous_value) / previous_value

        assert daily_return == Decimal("0.025")  # 2.5%

    def test_total_return_calculation(self):
        """Test calculating total return percentage."""
        invested = Decimal("9000.00")
        current_value = Decimal("10500.00")

        total_return = (current_value - invested) / invested

        assert round(total_return, 4) == Decimal("0.1667")  # ~16.67%

    def test_weighted_average_price(self):
        """Test calculating weighted average price."""
        transactions = [
            {"quantity": 100, "price": Decimal("35.00")},
            {"quantity": 50, "price": Decimal("37.00")},
            {"quantity": 50, "price": Decimal("33.00")},
        ]

        total_quantity = sum(t["quantity"] for t in transactions)
        total_cost = sum(t["quantity"] * t["price"] for t in transactions)
        weighted_avg = total_cost / total_quantity

        assert total_quantity == 200
        assert weighted_avg == Decimal("35.00")


class TestAssetClassification:
    """Test asset classification and allocation."""

    def test_allocation_calculation(self):
        """Test calculating portfolio allocation by asset class."""
        assets = [
            {"class": "stocks", "value": Decimal("6000")},
            {"class": "fiis", "value": Decimal("3000")},
            {"class": "fixed_income", "value": Decimal("1000")},
        ]

        total_value = sum(a["value"] for a in assets)
        allocations = {
            a["class"]: float(a["value"] / total_value * 100)
            for a in assets
        }

        assert allocations["stocks"] == 60.0
        assert allocations["fiis"] == 30.0
        assert allocations["fixed_income"] == 10.0

    def test_rebalancing_suggestion(self):
        """Test calculating rebalancing suggestions."""
        current_allocation = {"stocks": 70, "fiis": 20, "fixed_income": 10}
        target_allocation = {"stocks": 60, "fiis": 30, "fixed_income": 10}

        suggestions = {}
        for asset_class in current_allocation:
            diff = target_allocation[asset_class] - current_allocation[asset_class]
            if abs(diff) > 2:  # Only suggest if diff > 2%
                suggestions[asset_class] = "buy" if diff > 0 else "sell"

        assert suggestions["stocks"] == "sell"
        assert suggestions["fiis"] == "buy"
        assert "fixed_income" not in suggestions
