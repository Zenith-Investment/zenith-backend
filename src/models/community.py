"""Community models for sharing successful strategies."""
from datetime import datetime
from decimal import Decimal
from enum import Enum

from sqlalchemy import Boolean, DateTime, Enum as SQLEnum, ForeignKey, Integer, JSON, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.database import Base


class StrategyStatus(str, Enum):
    """Strategy sharing status."""
    PENDING = "pending"  # Awaiting verification
    VERIFIED = "verified"  # Verified successful strategy
    REJECTED = "rejected"  # Did not meet criteria


class CommunityStrategy(Base):
    """
    Shared successful strategies from the community.

    When a user's backtest shows good results and the user agrees to share,
    the strategy is stored here for other users with similar profiles.
    """
    __tablename__ = "community_strategies"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # Original creator (anonymized)
    creator_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Strategy details
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False)
    strategy_type: Mapped[str] = mapped_column(String(50), nullable=False)  # buy_and_hold, dca, etc
    strategy_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Applicable tickers/asset classes
    applicable_tickers: Mapped[list | None] = mapped_column(JSON, nullable=True)
    applicable_asset_classes: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Target profile
    target_risk_profile: Mapped[str] = mapped_column(String(50), nullable=False)  # conservative, moderate, aggressive
    min_investment_horizon_years: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    max_investment_horizon_years: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=True)
    recommended_goals: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Performance metrics (from original backtest)
    backtest_period_days: Mapped[int] = mapped_column(Integer, nullable=False)
    total_return: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    annualized_return: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    sharpe_ratio: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    max_drawdown: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    volatility: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    win_rate: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)

    # Community metrics
    times_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    success_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failure_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    avg_user_return: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    community_rating: Mapped[Decimal | None] = mapped_column(Numeric(3, 2), nullable=True)  # 1-5 stars

    # Verification status
    status: Mapped[StrategyStatus] = mapped_column(
        SQLEnum(StrategyStatus, values_callable=lambda x: [e.value for e in x]),
        default=StrategyStatus.PENDING,
        nullable=False,
    )
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Visibility
    is_public: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_featured: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        onupdate=func.now(),
        nullable=True,
    )

    # Relationships
    creator: Mapped["User"] = relationship("User", foreign_keys=[creator_id])
    uses: Mapped[list["StrategyUse"]] = relationship(
        "StrategyUse",
        back_populates="strategy",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<CommunityStrategy(id={self.id}, name={self.strategy_name}, rating={self.community_rating})>"


class StrategyUse(Base):
    """
    Tracks when users use community strategies.

    Records outcomes to improve recommendations and calculate success rates.
    """
    __tablename__ = "strategy_uses"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    strategy_id: Mapped[int] = mapped_column(
        ForeignKey("community_strategies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Use details
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    initial_value: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)

    # Outcome (updated periodically or when user stops using)
    current_value: Mapped[Decimal | None] = mapped_column(Numeric(18, 2), nullable=True)
    return_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # User feedback
    user_rating: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1-5
    user_feedback: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    strategy: Mapped["CommunityStrategy"] = relationship(
        "CommunityStrategy",
        back_populates="uses",
    )
    user: Mapped["User"] = relationship("User")

    def __repr__(self) -> str:
        return f"<StrategyUse(id={self.id}, strategy_id={self.strategy_id}, user_id={self.user_id})>"


class StrategyMLFeatures(Base):
    """
    Machine learning features extracted from successful strategies.

    Used to train the recommendation model to suggest better strategies.
    """
    __tablename__ = "strategy_ml_features"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    strategy_id: Mapped[int] = mapped_column(
        ForeignKey("community_strategies.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    # Feature vectors (stored as JSON for flexibility)
    market_condition_features: Mapped[dict] = mapped_column(JSON, nullable=False)
    risk_features: Mapped[dict] = mapped_column(JSON, nullable=False)
    return_features: Mapped[dict] = mapped_column(JSON, nullable=False)
    user_profile_features: Mapped[dict] = mapped_column(JSON, nullable=False)

    # Embedding vector (if using neural models)
    embedding_vector: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Feature extraction metadata
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Relationships
    strategy: Mapped["CommunityStrategy"] = relationship("CommunityStrategy")

    def __repr__(self) -> str:
        return f"<StrategyMLFeatures(id={self.id}, strategy_id={self.strategy_id})>"
