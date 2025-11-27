"""Add subscription tables

Revision ID: 005_add_subscription_tables
Revises: 004_add_broker_tables
Create Date: 2024-11-22 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '005_add_subscription_tables'
down_revision: Union[str, None] = '004_add_broker_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create subscription status enum
    op.execute("CREATE TYPE subscriptionstatus AS ENUM ('active', 'past_due', 'cancelled', 'expired', 'trial')")

    # Create payment status enum
    op.execute("CREATE TYPE paymentstatus AS ENUM ('pending', 'processing', 'paid', 'failed', 'refunded', 'cancelled')")

    # Create payment method enum
    op.execute("CREATE TYPE paymentmethod AS ENUM ('credit_card', 'debit_card', 'pix', 'boleto')")

    # Create subscriptions table
    op.create_table(
        'subscriptions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('plan', sa.Enum('starter', 'smart', 'pro', 'premium', name='subscriptionplan', create_type=False), nullable=False),
        sa.Column('status', sa.Enum('active', 'past_due', 'cancelled', 'expired', 'trial', name='subscriptionstatus', create_type=False), nullable=False),
        sa.Column('billing_cycle', sa.String(length=20), nullable=False, server_default='monthly'),
        sa.Column('stripe_subscription_id', sa.String(length=255), nullable=True),
        sa.Column('stripe_customer_id', sa.String(length=255), nullable=True),
        sa.Column('current_period_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('current_period_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('trial_end', sa.DateTime(timezone=True), nullable=True),
        sa.Column('cancelled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_subscriptions_user_id', 'subscriptions', ['user_id'], unique=False)

    # Create payments table
    op.create_table(
        'payments',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('subscription_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('amount', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('currency', sa.String(length=3), nullable=False, server_default='BRL'),
        sa.Column('status', sa.Enum('pending', 'processing', 'paid', 'failed', 'refunded', 'cancelled', name='paymentstatus', create_type=False), nullable=False),
        sa.Column('payment_method', sa.Enum('credit_card', 'debit_card', 'pix', 'boleto', name='paymentmethod', create_type=False), nullable=True),
        sa.Column('stripe_payment_intent_id', sa.String(length=255), nullable=True),
        sa.Column('stripe_invoice_id', sa.String(length=255), nullable=True),
        sa.Column('invoice_number', sa.String(length=50), nullable=True),
        sa.Column('invoice_url', sa.String(length=500), nullable=True),
        sa.Column('card_last_four', sa.String(length=4), nullable=True),
        sa.Column('card_brand', sa.String(length=20), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('paid_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('refunded_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['subscription_id'], ['subscriptions.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_payments_subscription_id', 'payments', ['subscription_id'], unique=False)
    op.create_index('ix_payments_user_id', 'payments', ['user_id'], unique=False)

    # Create coupons table
    op.create_table(
        'coupons',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('code', sa.String(length=50), nullable=False),
        sa.Column('discount_percent', sa.Integer(), nullable=True),
        sa.Column('discount_amount', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('max_uses', sa.Integer(), nullable=True),
        sa.Column('times_used', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('valid_from', sa.DateTime(timezone=True), nullable=False),
        sa.Column('valid_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('applicable_plans', sa.String(length=100), nullable=True),
        sa.Column('first_subscription_only', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('code')
    )
    op.create_index('ix_coupons_code', 'coupons', ['code'], unique=True)


def downgrade() -> None:
    op.drop_index('ix_coupons_code', table_name='coupons')
    op.drop_table('coupons')

    op.drop_index('ix_payments_user_id', table_name='payments')
    op.drop_index('ix_payments_subscription_id', table_name='payments')
    op.drop_table('payments')

    op.drop_index('ix_subscriptions_user_id', table_name='subscriptions')
    op.drop_table('subscriptions')

    op.execute("DROP TYPE paymentmethod")
    op.execute("DROP TYPE paymentstatus")
    op.execute("DROP TYPE subscriptionstatus")
