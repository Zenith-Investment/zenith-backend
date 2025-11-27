"""Broker connection models."""
from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Enum as SQLEnum, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.database import Base


class BrokerType(str, Enum):
    """Supported broker types."""
    XP = "xp"
    RICO = "rico"
    CLEAR = "clear"
    BTG = "btg"
    NUINVEST = "nuinvest"
    INTER = "inter"


class ConnectionStatus(str, Enum):
    """Broker connection status."""
    PENDING = "pending"  # Awaiting OAuth authorization
    ACTIVE = "active"  # Connected and working
    EXPIRED = "expired"  # Token expired, needs refresh
    ERROR = "error"  # Connection error
    REVOKED = "revoked"  # User revoked access


class SyncStatus(str, Enum):
    """Synchronization status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class BrokerConnection(Base):
    """
    Stores user's connection to a broker.

    This includes OAuth tokens (encrypted) and connection status.
    """
    __tablename__ = "broker_connections"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    broker_type: Mapped[BrokerType] = mapped_column(
        SQLEnum(BrokerType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )

    # OAuth tokens (encrypted in production)
    access_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    refresh_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Broker-specific account info
    broker_account_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    broker_account_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Connection status
    status: Mapped[ConnectionStatus] = mapped_column(
        SQLEnum(ConnectionStatus, values_callable=lambda x: [e.value for e in x]),
        default=ConnectionStatus.PENDING,
        nullable=False,
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Sync tracking
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # OAuth state for CSRF protection
    oauth_state: Mapped[str | None] = mapped_column(String(255), nullable=True)

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
    user: Mapped["User"] = relationship("User", back_populates="broker_connections")
    sync_history: Mapped[list["BrokerSyncHistory"]] = relationship(
        "BrokerSyncHistory",
        back_populates="connection",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<BrokerConnection(id={self.id}, user_id={self.user_id}, broker={self.broker_type.value})>"


class BrokerSyncHistory(Base):
    """
    Tracks synchronization history for broker connections.

    Records each sync attempt with status and statistics.
    """
    __tablename__ = "broker_sync_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    connection_id: Mapped[int] = mapped_column(
        ForeignKey("broker_connections.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Sync details
    sync_type: Mapped[str] = mapped_column(String(50), nullable=False)  # positions, transactions, accounts
    status: Mapped[SyncStatus] = mapped_column(
        SQLEnum(SyncStatus, values_callable=lambda x: [e.value for e in x]),
        default=SyncStatus.PENDING,
        nullable=False,
    )

    # Statistics
    records_synced: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    records_created: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    records_updated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timing
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    connection: Mapped["BrokerConnection"] = relationship(
        "BrokerConnection",
        back_populates="sync_history",
    )

    def __repr__(self) -> str:
        return f"<BrokerSyncHistory(id={self.id}, type={self.sync_type}, status={self.status.value})>"
