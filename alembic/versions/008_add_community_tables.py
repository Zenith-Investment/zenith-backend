"""Add community strategies tables

Revision ID: 008_community
Revises: 007_multiple_portfolios
Create Date: 2024-01-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '008_community'
down_revision: Union[str, None] = '007_multiple_portfolios'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create strategy status enum
    strategy_status = sa.Enum('pending', 'verified', 'rejected', name='strategystatus')
    strategy_status.create(op.get_bind(), checkfirst=True)

    # Create community_strategies table
    op.create_table(
        'community_strategies',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('creator_id', sa.Integer(), nullable=True),
        sa.Column('strategy_name', sa.String(length=100), nullable=False),
        sa.Column('strategy_type', sa.String(length=50), nullable=False),
        sa.Column('strategy_params', sa.JSON(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('applicable_tickers', sa.JSON(), nullable=True),
        sa.Column('applicable_asset_classes', sa.JSON(), nullable=True),
        sa.Column('target_risk_profile', sa.String(length=50), nullable=False),
        sa.Column('min_investment_horizon_years', sa.Numeric(precision=5, scale=2), nullable=False),
        sa.Column('max_investment_horizon_years', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('recommended_goals', sa.JSON(), nullable=True),
        sa.Column('backtest_period_days', sa.Integer(), nullable=False),
        sa.Column('total_return', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('annualized_return', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('sharpe_ratio', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('max_drawdown', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('volatility', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('win_rate', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('times_used', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('success_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('failure_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('avg_user_return', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('community_rating', sa.Numeric(precision=3, scale=2), nullable=True),
        sa.Column('status', sa.Enum('pending', 'verified', 'rejected', name='strategystatus'), nullable=False),
        sa.Column('verified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_public', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('is_featured', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['creator_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_community_strategies_creator_id'), 'community_strategies', ['creator_id'], unique=False)
    op.create_index(op.f('ix_community_strategies_target_risk_profile'), 'community_strategies', ['target_risk_profile'], unique=False)

    # Create strategy_uses table
    op.create_table(
        'strategy_uses',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('strategy_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('applied_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('initial_value', sa.Numeric(precision=18, scale=2), nullable=False),
        sa.Column('current_value', sa.Numeric(precision=18, scale=2), nullable=True),
        sa.Column('return_pct', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('stopped_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('user_rating', sa.Integer(), nullable=True),
        sa.Column('user_feedback', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['strategy_id'], ['community_strategies.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_strategy_uses_strategy_id'), 'strategy_uses', ['strategy_id'], unique=False)
    op.create_index(op.f('ix_strategy_uses_user_id'), 'strategy_uses', ['user_id'], unique=False)

    # Create strategy_ml_features table
    op.create_table(
        'strategy_ml_features',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('strategy_id', sa.Integer(), nullable=False),
        sa.Column('market_condition_features', sa.JSON(), nullable=False),
        sa.Column('risk_features', sa.JSON(), nullable=False),
        sa.Column('return_features', sa.JSON(), nullable=False),
        sa.Column('user_profile_features', sa.JSON(), nullable=False),
        sa.Column('embedding_vector', sa.JSON(), nullable=True),
        sa.Column('extracted_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('model_version', sa.String(length=50), nullable=True),
        sa.ForeignKeyConstraint(['strategy_id'], ['community_strategies.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('strategy_id')
    )


def downgrade() -> None:
    # Drop tables
    op.drop_table('strategy_ml_features')

    op.drop_index(op.f('ix_strategy_uses_user_id'), table_name='strategy_uses')
    op.drop_index(op.f('ix_strategy_uses_strategy_id'), table_name='strategy_uses')
    op.drop_table('strategy_uses')

    op.drop_index(op.f('ix_community_strategies_target_risk_profile'), table_name='community_strategies')
    op.drop_index(op.f('ix_community_strategies_creator_id'), table_name='community_strategies')
    op.drop_table('community_strategies')

    # Drop enum
    sa.Enum(name='strategystatus').drop(op.get_bind(), checkfirst=True)
