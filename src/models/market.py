"""Market data models."""
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Index, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from src.core.database import Base


class PriceHistory(Base):
    """Historical price data for assets (TimescaleDB hypertable)."""

    __tablename__ = "price_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=True)
    high_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=True)
    low_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=True)
    close_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    volume: Mapped[int | None] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_price_history_ticker_date", "ticker", "date", unique=True),
        Index("ix_price_history_date", "date"),
    )

    def __repr__(self) -> str:
        return f"<PriceHistory(ticker={self.ticker}, date={self.date}, close={self.close_price})>"


class PortfolioSnapshot(Base):
    """Daily portfolio value snapshots for performance tracking."""

    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    total_value: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    total_invested: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    daily_return: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_portfolio_snapshots_portfolio_date", "portfolio_id", "date", unique=True),
    )

    def __repr__(self) -> str:
        return f"<PortfolioSnapshot(portfolio_id={self.portfolio_id}, date={self.date}, value={self.total_value})>"
