from datetime import datetime
from decimal import Decimal
from enum import Enum as PyEnum

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.database import Base
from src.schemas.portfolio import AssetClass, TransactionType


class PortfolioType(str, PyEnum):
    """Portfolio type classification."""
    REAL = "real"  # Real portfolio with actual investments
    SIMULATED = "simulated"  # Paper trading / simulation portfolio
    WATCHLIST = "watchlist"  # Watchlist for tracking


class Portfolio(Base):
    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,  # Removed unique=True to allow multiple portfolios
    )
    name: Mapped[str] = mapped_column(String(100), default="Minha Carteira", nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Portfolio configuration
    portfolio_type: Mapped[PortfolioType] = mapped_column(
        Enum(PortfolioType, values_callable=lambda x: [e.value for e in x]),
        default=PortfolioType.REAL,
        nullable=False,
    )
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    color: Mapped[str | None] = mapped_column(String(7), nullable=True)  # Hex color for UI
    icon: Mapped[str | None] = mapped_column(String(50), nullable=True)  # Icon name for UI

    # Investment goals
    target_value: Mapped[Decimal | None] = mapped_column(Numeric(18, 2), nullable=True)
    risk_profile: Mapped[str | None] = mapped_column(String(50), nullable=True)

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
    user: Mapped["User"] = relationship("User", back_populates="portfolios")
    assets: Mapped[list["PortfolioAsset"]] = relationship(
        "PortfolioAsset",
        back_populates="portfolio",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Portfolio(id={self.id}, user_id={self.user_id})>"


class PortfolioAsset(Base):
    __tablename__ = "portfolio_assets"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=False,
    )

    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    asset_class: Mapped[AssetClass] = mapped_column(
        Enum(AssetClass, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    average_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    broker: Mapped[str | None] = mapped_column(String(100), nullable=True)

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
    portfolio: Mapped["Portfolio"] = relationship("Portfolio", back_populates="assets")
    transactions: Mapped[list["Transaction"]] = relationship(
        "Transaction",
        back_populates="asset",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<PortfolioAsset(id={self.id}, ticker={self.ticker}, qty={self.quantity})>"


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("portfolio_assets.id", ondelete="CASCADE"),
        nullable=False,
    )

    transaction_type: Mapped[TransactionType] = mapped_column(
        Enum(TransactionType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    total_value: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    fees: Mapped[Decimal] = mapped_column(Numeric(18, 2), default=Decimal("0"), nullable=False)

    transaction_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    notes: Mapped[str | None] = mapped_column(String(500), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    asset: Mapped["PortfolioAsset"] = relationship("PortfolioAsset", back_populates="transactions")

    def __repr__(self) -> str:
        return f"<Transaction(id={self.id}, type={self.transaction_type}, qty={self.quantity})>"
