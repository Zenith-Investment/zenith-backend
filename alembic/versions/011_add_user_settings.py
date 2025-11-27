"""Add user_settings table for LLM and UI preferences

Revision ID: 011_user_settings
Revises: 010_notifications
Create Date: 2025-01-22 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '011_user_settings'
down_revision: Union[str, None] = '010_notifications'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create LLM provider enum
    llm_provider_enum = sa.Enum(
        'ollama', 'openai', 'anthropic', 'deepseek', 'groq', 'together', 'auto',
        name='llmprovider'
    )
    llm_provider_enum.create(op.get_bind(), checkfirst=True)

    # Create user_settings table
    op.create_table(
        'user_settings',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        # LLM Preferences
        sa.Column('llm_provider', llm_provider_enum, nullable=False, server_default='auto'),
        sa.Column('llm_model', sa.String(length=100), nullable=True),
        # User's API keys (encrypted storage recommended in production)
        sa.Column('openai_api_key', sa.String(length=255), nullable=True),
        sa.Column('anthropic_api_key', sa.String(length=255), nullable=True),
        sa.Column('deepseek_api_key', sa.String(length=255), nullable=True),
        sa.Column('groq_api_key', sa.String(length=255), nullable=True),
        sa.Column('together_api_key', sa.String(length=255), nullable=True),
        # UI Preferences
        sa.Column('theme', sa.String(length=20), nullable=False, server_default='system'),
        sa.Column('language', sa.String(length=10), nullable=False, server_default='pt-BR'),
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id')
    )
    op.create_index(op.f('ix_user_settings_user_id'), 'user_settings', ['user_id'], unique=True)


def downgrade() -> None:
    # Drop table
    op.drop_index(op.f('ix_user_settings_user_id'), table_name='user_settings')
    op.drop_table('user_settings')

    # Drop enum
    sa.Enum(name='llmprovider').drop(op.get_bind(), checkfirst=True)
