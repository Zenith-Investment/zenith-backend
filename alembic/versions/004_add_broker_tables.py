"""Add broker connection tables.

Revision ID: 004_add_broker_tables
Revises: 003_add_market_tables
Create Date: 2025-01-22

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "004_add_broker_tables"
down_revision = "003_add_market_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create broker_type ENUM
    broker_type_enum = sa.Enum(
        "xp", "rico", "clear", "btg", "nuinvest", "inter",
        name="brokertype"
    )
    broker_type_enum.create(op.get_bind(), checkfirst=True)

    # Create connection_status ENUM
    connection_status_enum = sa.Enum(
        "pending", "active", "expired", "error", "revoked",
        name="connectionstatus"
    )
    connection_status_enum.create(op.get_bind(), checkfirst=True)

    # Create sync_status ENUM
    sync_status_enum = sa.Enum(
        "pending", "running", "success", "failed",
        name="syncstatus"
    )
    sync_status_enum.create(op.get_bind(), checkfirst=True)

    # Create broker_connections table
    op.create_table(
        "broker_connections",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("broker_type", broker_type_enum, nullable=False),
        sa.Column("access_token_encrypted", sa.Text(), nullable=True),
        sa.Column("refresh_token_encrypted", sa.Text(), nullable=True),
        sa.Column("token_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("broker_account_id", sa.String(100), nullable=True),
        sa.Column("broker_account_name", sa.String(255), nullable=True),
        sa.Column("status", connection_status_enum, nullable=False, server_default="pending"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("last_sync_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("oauth_state", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create index for user_id lookup
    op.create_index("ix_broker_connections_user_id", "broker_connections", ["user_id"])

    # Create unique constraint for user + broker combination
    op.create_index(
        "ix_broker_connections_user_broker",
        "broker_connections",
        ["user_id", "broker_type"],
        unique=True,
    )

    # Create broker_sync_history table
    op.create_table(
        "broker_sync_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("connection_id", sa.Integer(), nullable=False),
        sa.Column("sync_type", sa.String(50), nullable=False),
        sa.Column("status", sync_status_enum, nullable=False, server_default="pending"),
        sa.Column("records_synced", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("records_created", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("records_updated", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["connection_id"], ["broker_connections.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create index for connection_id lookup
    op.create_index("ix_broker_sync_history_connection_id", "broker_sync_history", ["connection_id"])


def downgrade() -> None:
    # Drop tables
    op.drop_table("broker_sync_history")
    op.drop_table("broker_connections")

    # Drop enums
    sa.Enum(name="syncstatus").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="connectionstatus").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="brokertype").drop(op.get_bind(), checkfirst=True)
