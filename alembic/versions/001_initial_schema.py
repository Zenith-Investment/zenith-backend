"""Initial schema - create all tables

Revision ID: 001_initial_schema
Revises:
Create Date: 2025-01-22

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create ENUM types
    subscription_plan_enum = postgresql.ENUM(
        "starter", "smart", "pro", "premium",
        name="subscriptionplan",
        create_type=False,
    )
    subscription_plan_enum.create(op.get_bind(), checkfirst=True)

    asset_class_enum = postgresql.ENUM(
        "stocks", "fiis", "fixed_income", "crypto", "etf", "bdr", "funds", "cash", "other",
        name="assetclass",
        create_type=False,
    )
    asset_class_enum.create(op.get_bind(), checkfirst=True)

    transaction_type_enum = postgresql.ENUM(
        "buy", "sell", "dividend", "split",
        name="transactiontype",
        create_type=False,
    )
    transaction_type_enum.create(op.get_bind(), checkfirst=True)

    risk_profile_enum = postgresql.ENUM(
        "conservative", "moderate", "balanced", "growth", "aggressive",
        name="riskprofile",
        create_type=False,
    )
    risk_profile_enum.create(op.get_bind(), checkfirst=True)

    investment_horizon_enum = postgresql.ENUM(
        "short_term", "medium_term", "long_term", "very_long_term",
        name="investmenthorizon",
        create_type=False,
    )
    investment_horizon_enum.create(op.get_bind(), checkfirst=True)

    message_role_enum = postgresql.ENUM(
        "user", "assistant", "system",
        name="messagerole",
        create_type=False,
    )
    message_role_enum.create(op.get_bind(), checkfirst=True)

    feedback_type_enum = postgresql.ENUM(
        "like", "dislike",
        name="feedbacktype",
        create_type=False,
    )
    feedback_type_enum.create(op.get_bind(), checkfirst=True)

    alert_condition_enum = postgresql.ENUM(
        "above", "below",
        name="alertcondition",
        create_type=False,
    )
    alert_condition_enum.create(op.get_bind(), checkfirst=True)

    # === 1. USERS TABLE ===
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=100), nullable=False),
        sa.Column("phone", sa.String(length=20), nullable=True),
        sa.Column("cpf_encrypted", sa.String(length=255), nullable=True),
        sa.Column("avatar_url", sa.String(length=500), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, default=True),
        sa.Column("is_verified", sa.Boolean(), nullable=False, default=False),
        sa.Column("is_superuser", sa.Boolean(), nullable=False, default=False),
        sa.Column(
            "subscription_plan",
            postgresql.ENUM(
                "starter", "smart", "pro", "premium",
                name="subscriptionplan",
                create_type=False,
            ),
            nullable=False,
            server_default="starter",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # === 2. INVESTOR PROFILES TABLE ===
    op.create_table(
        "investor_profiles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column(
            "risk_profile",
            postgresql.ENUM(
                "conservative", "moderate", "balanced", "growth", "aggressive",
                name="riskprofile",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("risk_score", sa.Integer(), nullable=False),
        sa.Column(
            "investment_horizon",
            postgresql.ENUM(
                "short_term", "medium_term", "long_term", "very_long_term",
                name="investmenthorizon",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("primary_goals", postgresql.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("monthly_income", sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column("monthly_investment", sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column("total_patrimony", sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column("experience_level", sa.String(length=50), nullable=True),
        sa.Column("assessment_data", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id"),
    )

    # === 3. ASSESSMENT SESSIONS TABLE ===
    op.create_table(
        "assessment_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="in_progress"),
        sa.Column("answers", sa.Text(), nullable=True),
        sa.Column("result", sa.Text(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id"),
    )

    # === 4. PORTFOLIOS TABLE ===
    op.create_table(
        "portfolios",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False, server_default="Minha Carteira"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id"),
    )

    # === 5. PORTFOLIO ASSETS TABLE ===
    op.create_table(
        "portfolio_assets",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("portfolio_id", sa.Integer(), nullable=False),
        sa.Column("ticker", sa.String(length=20), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=True),
        sa.Column(
            "asset_class",
            postgresql.ENUM(
                "stocks", "fiis", "fixed_income", "crypto", "etf", "bdr", "funds", "cash", "other",
                name="assetclass",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("quantity", sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column("average_price", sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column("broker", sa.String(length=100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["portfolio_id"], ["portfolios.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_portfolio_assets_ticker", "portfolio_assets", ["ticker"])

    # === 6. TRANSACTIONS TABLE ===
    op.create_table(
        "transactions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("asset_id", sa.Integer(), nullable=False),
        sa.Column(
            "transaction_type",
            postgresql.ENUM(
                "buy", "sell", "dividend", "split",
                name="transactiontype",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("quantity", sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column("price", sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column("total_value", sa.Numeric(precision=18, scale=2), nullable=False),
        sa.Column("fees", sa.Numeric(precision=18, scale=2), nullable=False, server_default="0"),
        sa.Column("transaction_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("notes", sa.String(length=500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["asset_id"], ["portfolio_assets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # === 7. CHAT SESSIONS TABLE ===
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chat_sessions_session_id", "chat_sessions", ["session_id"], unique=True)

    # === 8. CHAT MESSAGES TABLE ===
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("message_id", sa.String(length=36), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column(
            "role",
            postgresql.ENUM(
                "user", "assistant", "system",
                name="messagerole",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tokens_used", sa.Integer(), nullable=True),
        sa.Column("model_used", sa.String(length=50), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chat_messages_message_id", "chat_messages", ["message_id"], unique=True)

    # === 9. CHAT FEEDBACK TABLE ===
    op.create_table(
        "chat_feedback",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column(
            "feedback_type",
            postgresql.ENUM(
                "like", "dislike",
                name="feedbacktype",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("comment", sa.String(length=500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["message_id"], ["chat_messages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("message_id"),
    )

    # === 10. PRICE ALERTS TABLE ===
    op.create_table(
        "price_alerts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("ticker", sa.String(length=20), nullable=False),
        sa.Column("target_price", sa.Numeric(precision=18, scale=2), nullable=False),
        sa.Column(
            "condition",
            postgresql.ENUM(
                "above", "below",
                name="alertcondition",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, default=True),
        sa.Column("is_triggered", sa.Boolean(), nullable=False, default=False),
        sa.Column("triggered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("triggered_price", sa.Numeric(precision=18, scale=2), nullable=True),
        sa.Column("notes", sa.String(length=500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_price_alerts_user_id", "price_alerts", ["user_id"])
    op.create_index("ix_price_alerts_ticker", "price_alerts", ["ticker"])


def downgrade() -> None:
    # Drop tables in reverse order (respecting foreign key constraints)
    op.drop_index("ix_price_alerts_ticker", table_name="price_alerts")
    op.drop_index("ix_price_alerts_user_id", table_name="price_alerts")
    op.drop_table("price_alerts")

    op.drop_table("chat_feedback")

    op.drop_index("ix_chat_messages_message_id", table_name="chat_messages")
    op.drop_table("chat_messages")

    op.drop_index("ix_chat_sessions_session_id", table_name="chat_sessions")
    op.drop_table("chat_sessions")

    op.drop_table("transactions")

    op.drop_index("ix_portfolio_assets_ticker", table_name="portfolio_assets")
    op.drop_table("portfolio_assets")

    op.drop_table("portfolios")

    op.drop_table("assessment_sessions")

    op.drop_table("investor_profiles")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")

    # Drop ENUM types
    postgresql.ENUM(name="alertcondition").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="feedbacktype").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="messagerole").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="investmenthorizon").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="riskprofile").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="transactiontype").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="assetclass").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="subscriptionplan").drop(op.get_bind(), checkfirst=True)
