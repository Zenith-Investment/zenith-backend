"""Analytics models for storing backtest and forecast history."""
from datetime import datetime
from decimal import Decimal
from enum import Enum

from sqlalchemy import DateTime, Enum as SQLEnum, ForeignKey, Integer, JSON, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.database import Base


class BacktestStatus(str, Enum):
    """Backtest execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Backtest(Base):
    """Stored backtest results."""
    __tablename__ = "backtests"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Strategy info
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False)
    strategy_params: Mapped[dict] = mapped_column(JSON, nullable=True)

    # Backtest parameters
    tickers: Mapped[list] = mapped_column(JSON, nullable=False)
    start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    initial_capital: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)

    # Status
    status: Mapped[BacktestStatus] = mapped_column(
        SQLEnum(BacktestStatus, values_callable=lambda x: [e.value for e in x]),
        default=BacktestStatus.PENDING,
        nullable=False,
    )

    # Results
    final_value: Mapped[Decimal | None] = mapped_column(Numeric(15, 2), nullable=True)
    total_return: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    annualized_return: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    volatility: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    sharpe_ratio: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    max_drawdown: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    win_rate: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    total_trades: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Full results JSON
    full_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    daily_values: Mapped[list | None] = mapped_column(JSON, nullable=True)
    trades: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Error info
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="backtests")

    def __repr__(self) -> str:
        return f"<Backtest(id={self.id}, strategy={self.strategy_name}, status={self.status})>"


class PriceForecastHistory(Base):
    """Stored price forecasts."""
    __tablename__ = "price_forecasts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Asset info
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    current_price: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)

    # Forecast info
    forecast_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    forecast_days: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_price: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    predicted_change_pct: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)

    # Range
    prediction_low: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    prediction_high: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)

    # Methodology and factors
    methodology: Mapped[str] = mapped_column(String(200), nullable=False)
    factors: Mapped[list] = mapped_column(JSON, nullable=True)
    strategy_backtests: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Actual result (for tracking accuracy)
    actual_price: Mapped[Decimal | None] = mapped_column(Numeric(15, 2), nullable=True)
    accuracy_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="forecasts")

    def __repr__(self) -> str:
        return f"<PriceForecastHistory(id={self.id}, ticker={self.ticker}, forecast_date={self.forecast_date})>"


class StrategyRecommendationHistory(Base):
    """Stored strategy recommendations."""
    __tablename__ = "strategy_recommendations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # User profile at time of recommendation
    risk_profile: Mapped[str] = mapped_column(String(50), nullable=False)
    investment_horizon_years: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    goals: Mapped[list] = mapped_column(JSON, nullable=True)

    # Primary recommendation
    primary_strategy: Mapped[str] = mapped_column(String(100), nullable=False)
    suitability_score: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)

    # Full recommendations
    all_recommendations: Mapped[list] = mapped_column(JSON, nullable=False)
    portfolio_allocation: Mapped[dict] = mapped_column(JSON, nullable=False)
    rebalance_frequency: Mapped[str] = mapped_column(String(50), nullable=False)

    # User action
    user_accepted: Mapped[bool | None] = mapped_column(nullable=True)
    user_feedback: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="strategy_recommendations")

    def __repr__(self) -> str:
        return f"<StrategyRecommendation(id={self.id}, strategy={self.primary_strategy})>"
