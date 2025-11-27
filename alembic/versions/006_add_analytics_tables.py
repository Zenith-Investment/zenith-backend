"""Add analytics tables for backtests, forecasts, and recommendations

Revision ID: 006_analytics
Revises: 005_add_subscription_tables
Create Date: 2024-01-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '006_analytics'
down_revision: Union[str, None] = '005_add_subscription_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create backtest status enum
    backtest_status = sa.Enum(
        'pending', 'running', 'completed', 'failed',
        name='backteststatus'
    )
    backtest_status.create(op.get_bind(), checkfirst=True)

    # Create backtests table
    op.create_table(
        'backtests',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('strategy_name', sa.String(length=100), nullable=False),
        sa.Column('strategy_params', sa.JSON(), nullable=True),
        sa.Column('tickers', sa.JSON(), nullable=False),
        sa.Column('start_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('initial_capital', sa.Numeric(precision=15, scale=2), nullable=False),
        sa.Column('status', sa.Enum('pending', 'running', 'completed', 'failed', name='backteststatus'), nullable=False),
        sa.Column('final_value', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('total_return', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('annualized_return', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('volatility', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('sharpe_ratio', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('max_drawdown', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('win_rate', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('total_trades', sa.Integer(), nullable=True),
        sa.Column('full_results', sa.JSON(), nullable=True),
        sa.Column('daily_values', sa.JSON(), nullable=True),
        sa.Column('trades', sa.JSON(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_backtests_user_id'), 'backtests', ['user_id'], unique=False)

    # Create price_forecasts table
    op.create_table(
        'price_forecasts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('ticker', sa.String(length=20), nullable=False),
        sa.Column('current_price', sa.Numeric(precision=15, scale=2), nullable=False),
        sa.Column('forecast_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('forecast_days', sa.Integer(), nullable=False),
        sa.Column('predicted_price', sa.Numeric(precision=15, scale=2), nullable=False),
        sa.Column('predicted_change_pct', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('confidence', sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column('prediction_low', sa.Numeric(precision=15, scale=2), nullable=False),
        sa.Column('prediction_high', sa.Numeric(precision=15, scale=2), nullable=False),
        sa.Column('methodology', sa.String(length=200), nullable=False),
        sa.Column('factors', sa.JSON(), nullable=True),
        sa.Column('strategy_backtests', sa.JSON(), nullable=True),
        sa.Column('actual_price', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('accuracy_pct', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_price_forecasts_user_id'), 'price_forecasts', ['user_id'], unique=False)
    op.create_index(op.f('ix_price_forecasts_ticker'), 'price_forecasts', ['ticker'], unique=False)

    # Create strategy_recommendations table
    op.create_table(
        'strategy_recommendations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('risk_profile', sa.String(length=50), nullable=False),
        sa.Column('investment_horizon_years', sa.Numeric(precision=5, scale=2), nullable=False),
        sa.Column('goals', sa.JSON(), nullable=True),
        sa.Column('primary_strategy', sa.String(length=100), nullable=False),
        sa.Column('suitability_score', sa.Numeric(precision=5, scale=2), nullable=False),
        sa.Column('all_recommendations', sa.JSON(), nullable=False),
        sa.Column('portfolio_allocation', sa.JSON(), nullable=False),
        sa.Column('rebalance_frequency', sa.String(length=50), nullable=False),
        sa.Column('user_accepted', sa.Boolean(), nullable=True),
        sa.Column('user_feedback', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_strategy_recommendations_user_id'), 'strategy_recommendations', ['user_id'], unique=False)


def downgrade() -> None:
    # Drop tables
    op.drop_index(op.f('ix_strategy_recommendations_user_id'), table_name='strategy_recommendations')
    op.drop_table('strategy_recommendations')

    op.drop_index(op.f('ix_price_forecasts_ticker'), table_name='price_forecasts')
    op.drop_index(op.f('ix_price_forecasts_user_id'), table_name='price_forecasts')
    op.drop_table('price_forecasts')

    op.drop_index(op.f('ix_backtests_user_id'), table_name='backtests')
    op.drop_table('backtests')

    # Drop enum
    sa.Enum(name='backteststatus').drop(op.get_bind(), checkfirst=True)
