"""Add support for multiple portfolios per user

Revision ID: 007_multiple_portfolios
Revises: 006_analytics
Create Date: 2024-01-15 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '007_multiple_portfolios'
down_revision: Union[str, None] = '006_analytics'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create portfolio type enum
    portfolio_type = sa.Enum('real', 'simulated', 'watchlist', name='portfoliotype')
    portfolio_type.create(op.get_bind(), checkfirst=True)

    # Remove unique constraint on user_id to allow multiple portfolios
    op.drop_constraint('portfolios_user_id_key', 'portfolios', type_='unique')

    # Add new columns to portfolios table
    op.add_column('portfolios', sa.Column('description', sa.Text(), nullable=True))
    op.add_column('portfolios', sa.Column(
        'portfolio_type',
        sa.Enum('real', 'simulated', 'watchlist', name='portfoliotype'),
        nullable=False,
        server_default='real'
    ))
    op.add_column('portfolios', sa.Column('is_primary', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('portfolios', sa.Column('color', sa.String(length=7), nullable=True))
    op.add_column('portfolios', sa.Column('icon', sa.String(length=50), nullable=True))
    op.add_column('portfolios', sa.Column('target_value', sa.Numeric(precision=18, scale=2), nullable=True))
    op.add_column('portfolios', sa.Column('risk_profile', sa.String(length=50), nullable=True))

    # Create index on user_id (no longer unique)
    op.create_index(op.f('ix_portfolios_user_id'), 'portfolios', ['user_id'], unique=False)

    # Set existing portfolios as primary
    op.execute("UPDATE portfolios SET is_primary = true WHERE id IN (SELECT MIN(id) FROM portfolios GROUP BY user_id)")


def downgrade() -> None:
    # Drop new columns
    op.drop_index(op.f('ix_portfolios_user_id'), table_name='portfolios')
    op.drop_column('portfolios', 'risk_profile')
    op.drop_column('portfolios', 'target_value')
    op.drop_column('portfolios', 'icon')
    op.drop_column('portfolios', 'color')
    op.drop_column('portfolios', 'is_primary')
    op.drop_column('portfolios', 'portfolio_type')
    op.drop_column('portfolios', 'description')

    # Re-add unique constraint
    op.create_unique_constraint('portfolios_user_id_key', 'portfolios', ['user_id'])

    # Drop enum
    sa.Enum(name='portfoliotype').drop(op.get_bind(), checkfirst=True)
