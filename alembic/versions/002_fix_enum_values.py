"""Fix ENUM values to use lowercase strings

Revision ID: 002_fix_enum_values
Revises: 001_initial_schema
Create Date: 2025-01-22

This migration fixes the ENUM values that were created with Python member names
(STOCKS, FIIS, etc.) instead of the actual string values (stocks, fiis, etc.).
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002_fix_enum_values"
down_revision: Union[str, None] = "001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Rename ENUM values from UPPERCASE to lowercase

    # 1. subscriptionplan
    op.execute("ALTER TYPE subscriptionplan RENAME VALUE 'STARTER' TO 'starter'")
    op.execute("ALTER TYPE subscriptionplan RENAME VALUE 'SMART' TO 'smart'")
    op.execute("ALTER TYPE subscriptionplan RENAME VALUE 'PRO' TO 'pro'")
    op.execute("ALTER TYPE subscriptionplan RENAME VALUE 'PREMIUM' TO 'premium'")

    # 2. assetclass
    op.execute("ALTER TYPE assetclass RENAME VALUE 'STOCKS' TO 'stocks'")
    op.execute("ALTER TYPE assetclass RENAME VALUE 'FIIS' TO 'fiis'")
    op.execute("ALTER TYPE assetclass RENAME VALUE 'FIXED_INCOME' TO 'fixed_income'")
    op.execute("ALTER TYPE assetclass RENAME VALUE 'CRYPTO' TO 'crypto'")
    op.execute("ALTER TYPE assetclass RENAME VALUE 'ETF' TO 'etf'")
    op.execute("ALTER TYPE assetclass RENAME VALUE 'BDR' TO 'bdr'")
    op.execute("ALTER TYPE assetclass RENAME VALUE 'FUNDS' TO 'funds'")
    op.execute("ALTER TYPE assetclass RENAME VALUE 'CASH' TO 'cash'")
    op.execute("ALTER TYPE assetclass RENAME VALUE 'OTHER' TO 'other'")

    # 3. transactiontype
    op.execute("ALTER TYPE transactiontype RENAME VALUE 'BUY' TO 'buy'")
    op.execute("ALTER TYPE transactiontype RENAME VALUE 'SELL' TO 'sell'")
    op.execute("ALTER TYPE transactiontype RENAME VALUE 'DIVIDEND' TO 'dividend'")
    op.execute("ALTER TYPE transactiontype RENAME VALUE 'SPLIT' TO 'split'")

    # 4. riskprofile
    op.execute("ALTER TYPE riskprofile RENAME VALUE 'CONSERVATIVE' TO 'conservative'")
    op.execute("ALTER TYPE riskprofile RENAME VALUE 'MODERATE' TO 'moderate'")
    op.execute("ALTER TYPE riskprofile RENAME VALUE 'BALANCED' TO 'balanced'")
    op.execute("ALTER TYPE riskprofile RENAME VALUE 'GROWTH' TO 'growth'")
    op.execute("ALTER TYPE riskprofile RENAME VALUE 'AGGRESSIVE' TO 'aggressive'")

    # 5. investmenthorizon
    op.execute("ALTER TYPE investmenthorizon RENAME VALUE 'SHORT_TERM' TO 'short_term'")
    op.execute("ALTER TYPE investmenthorizon RENAME VALUE 'MEDIUM_TERM' TO 'medium_term'")
    op.execute("ALTER TYPE investmenthorizon RENAME VALUE 'LONG_TERM' TO 'long_term'")
    op.execute("ALTER TYPE investmenthorizon RENAME VALUE 'VERY_LONG_TERM' TO 'very_long_term'")

    # 6. messagerole
    op.execute("ALTER TYPE messagerole RENAME VALUE 'USER' TO 'user'")
    op.execute("ALTER TYPE messagerole RENAME VALUE 'ASSISTANT' TO 'assistant'")
    op.execute("ALTER TYPE messagerole RENAME VALUE 'SYSTEM' TO 'system'")

    # 7. feedbacktype
    op.execute("ALTER TYPE feedbacktype RENAME VALUE 'LIKE' TO 'like'")
    op.execute("ALTER TYPE feedbacktype RENAME VALUE 'DISLIKE' TO 'dislike'")

    # 8. alertcondition
    op.execute("ALTER TYPE alertcondition RENAME VALUE 'ABOVE' TO 'above'")
    op.execute("ALTER TYPE alertcondition RENAME VALUE 'BELOW' TO 'below'")


def downgrade() -> None:
    # Revert ENUM values from lowercase to UPPERCASE

    # 1. subscriptionplan
    op.execute("ALTER TYPE subscriptionplan RENAME VALUE 'starter' TO 'STARTER'")
    op.execute("ALTER TYPE subscriptionplan RENAME VALUE 'smart' TO 'SMART'")
    op.execute("ALTER TYPE subscriptionplan RENAME VALUE 'pro' TO 'PRO'")
    op.execute("ALTER TYPE subscriptionplan RENAME VALUE 'premium' TO 'PREMIUM'")

    # 2. assetclass
    op.execute("ALTER TYPE assetclass RENAME VALUE 'stocks' TO 'STOCKS'")
    op.execute("ALTER TYPE assetclass RENAME VALUE 'fiis' TO 'FIIS'")
    op.execute("ALTER TYPE assetclass RENAME VALUE 'fixed_income' TO 'FIXED_INCOME'")
    op.execute("ALTER TYPE assetclass RENAME VALUE 'crypto' TO 'CRYPTO'")
    op.execute("ALTER TYPE assetclass RENAME VALUE 'etf' TO 'ETF'")
    op.execute("ALTER TYPE assetclass RENAME VALUE 'bdr' TO 'BDR'")
    op.execute("ALTER TYPE assetclass RENAME VALUE 'funds' TO 'FUNDS'")
    op.execute("ALTER TYPE assetclass RENAME VALUE 'cash' TO 'CASH'")
    op.execute("ALTER TYPE assetclass RENAME VALUE 'other' TO 'OTHER'")

    # 3. transactiontype
    op.execute("ALTER TYPE transactiontype RENAME VALUE 'buy' TO 'BUY'")
    op.execute("ALTER TYPE transactiontype RENAME VALUE 'sell' TO 'SELL'")
    op.execute("ALTER TYPE transactiontype RENAME VALUE 'dividend' TO 'DIVIDEND'")
    op.execute("ALTER TYPE transactiontype RENAME VALUE 'split' TO 'SPLIT'")

    # 4. riskprofile
    op.execute("ALTER TYPE riskprofile RENAME VALUE 'conservative' TO 'CONSERVATIVE'")
    op.execute("ALTER TYPE riskprofile RENAME VALUE 'moderate' TO 'MODERATE'")
    op.execute("ALTER TYPE riskprofile RENAME VALUE 'balanced' TO 'BALANCED'")
    op.execute("ALTER TYPE riskprofile RENAME VALUE 'growth' TO 'GROWTH'")
    op.execute("ALTER TYPE riskprofile RENAME VALUE 'aggressive' TO 'AGGRESSIVE'")

    # 5. investmenthorizon
    op.execute("ALTER TYPE investmenthorizon RENAME VALUE 'short_term' TO 'SHORT_TERM'")
    op.execute("ALTER TYPE investmenthorizon RENAME VALUE 'medium_term' TO 'MEDIUM_TERM'")
    op.execute("ALTER TYPE investmenthorizon RENAME VALUE 'long_term' TO 'LONG_TERM'")
    op.execute("ALTER TYPE investmenthorizon RENAME VALUE 'very_long_term' TO 'VERY_LONG_TERM'")

    # 6. messagerole
    op.execute("ALTER TYPE messagerole RENAME VALUE 'user' TO 'USER'")
    op.execute("ALTER TYPE messagerole RENAME VALUE 'assistant' TO 'ASSISTANT'")
    op.execute("ALTER TYPE messagerole RENAME VALUE 'system' TO 'SYSTEM'")

    # 7. feedbacktype
    op.execute("ALTER TYPE feedbacktype RENAME VALUE 'like' TO 'LIKE'")
    op.execute("ALTER TYPE feedbacktype RENAME VALUE 'dislike' TO 'DISLIKE'")

    # 8. alertcondition
    op.execute("ALTER TYPE alertcondition RENAME VALUE 'above' TO 'ABOVE'")
    op.execute("ALTER TYPE alertcondition RENAME VALUE 'below' TO 'BELOW'")
