"""Add market history tables

Revision ID: 003_add_market_tables
Revises: 002_fix_enum_values
Create Date: 2025-01-22

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "003_add_market_tables"
down_revision: Union[str, None] = "002_fix_enum_values"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create price_history table
    op.create_table(
        "price_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(length=20), nullable=False),
        sa.Column("date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open_price", sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column("high_price", sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column("low_price", sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column("close_price", sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_price_history_ticker_date",
        "price_history",
        ["ticker", "date"],
        unique=True,
    )
    op.create_index("ix_price_history_date", "price_history", ["date"])

    # Convert to TimescaleDB hypertable (if TimescaleDB extension is available)
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
                PERFORM create_hypertable('price_history', 'date', if_not_exists => TRUE);
            END IF;
        END $$;
    """)

    # Create portfolio_snapshots table
    op.create_table(
        "portfolio_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("portfolio_id", sa.Integer(), nullable=False),
        sa.Column("date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("total_value", sa.Numeric(precision=18, scale=2), nullable=False),
        sa.Column("total_invested", sa.Numeric(precision=18, scale=2), nullable=False),
        sa.Column("daily_return", sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["portfolio_id"], ["portfolios.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_portfolio_snapshots_portfolio_date",
        "portfolio_snapshots",
        ["portfolio_id", "date"],
        unique=True,
    )

    # Convert to TimescaleDB hypertable
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
                PERFORM create_hypertable('portfolio_snapshots', 'date', if_not_exists => TRUE);
            END IF;
        END $$;
    """)


def downgrade() -> None:
    op.drop_index("ix_portfolio_snapshots_portfolio_date", table_name="portfolio_snapshots")
    op.drop_table("portfolio_snapshots")

    op.drop_index("ix_price_history_date", table_name="price_history")
    op.drop_index("ix_price_history_ticker_date", table_name="price_history")
    op.drop_table("price_history")
