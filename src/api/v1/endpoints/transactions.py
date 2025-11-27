"""Transaction endpoints for tracking buy/sell history."""
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
import structlog

from src.core.deps import CurrentUser, DbSession
from src.models.portfolio import Portfolio, PortfolioAsset, Transaction
from src.schemas.portfolio import (
    TransactionCreate,
    TransactionResponse,
    TransactionListResponse,
    TransactionType,
)

router = APIRouter()
logger = structlog.get_logger()


@router.get("/", response_model=TransactionListResponse)
async def list_transactions(
    current_user: CurrentUser,
    db: DbSession,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    asset_id: int | None = None,
    transaction_type: TransactionType | None = None,
) -> TransactionListResponse:
    """List all transactions for the current user's portfolio."""
    # Get user's portfolio
    portfolio_result = await db.execute(
        select(Portfolio).where(Portfolio.user_id == current_user.id)
    )
    portfolio = portfolio_result.scalar_one_or_none()

    if not portfolio:
        return TransactionListResponse(
            transactions=[], total=0, page=page, page_size=page_size
        )

    # Build query
    query = (
        select(Transaction)
        .join(PortfolioAsset)
        .where(PortfolioAsset.portfolio_id == portfolio.id)
        .options(selectinload(Transaction.asset))
    )

    if asset_id:
        query = query.where(Transaction.asset_id == asset_id)
    if transaction_type:
        query = query.where(Transaction.transaction_type == transaction_type)

    # Count total
    count_query = (
        select(func.count(Transaction.id))
        .join(PortfolioAsset)
        .where(PortfolioAsset.portfolio_id == portfolio.id)
    )
    if asset_id:
        count_query = count_query.where(Transaction.asset_id == asset_id)
    if transaction_type:
        count_query = count_query.where(Transaction.transaction_type == transaction_type)

    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Paginate and order
    query = (
        query.order_by(Transaction.transaction_date.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )

    result = await db.execute(query)
    transactions = result.scalars().all()

    return TransactionListResponse(
        transactions=[
            TransactionResponse(
                id=t.id,
                asset_id=t.asset_id,
                ticker=t.asset.ticker,
                asset_class=t.asset.asset_class,
                transaction_type=t.transaction_type,
                quantity=t.quantity,
                price=t.price,
                total_value=t.total_value,
                fees=t.fees,
                transaction_date=t.transaction_date,
                notes=t.notes,
                created_at=t.created_at,
            )
            for t in transactions
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/", response_model=TransactionResponse, status_code=status.HTTP_201_CREATED)
async def create_transaction(
    request: TransactionCreate,
    current_user: CurrentUser,
    db: DbSession,
) -> TransactionResponse:
    """Create a new transaction and update asset quantity/average price."""
    # Verify the asset belongs to user
    asset_result = await db.execute(
        select(PortfolioAsset)
        .join(Portfolio)
        .where(
            PortfolioAsset.id == request.asset_id,
            Portfolio.user_id == current_user.id,
        )
    )
    asset = asset_result.scalar_one_or_none()

    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ativo nao encontrado na sua carteira.",
        )

    # Calculate total value
    total_value = request.quantity * request.price + request.fees

    # Create transaction
    transaction = Transaction(
        asset_id=request.asset_id,
        transaction_type=request.transaction_type,
        quantity=request.quantity,
        price=request.price,
        total_value=total_value,
        fees=request.fees,
        transaction_date=request.transaction_date,
        notes=request.notes,
    )

    # Update asset based on transaction type
    if request.transaction_type == TransactionType.BUY:
        # Update average price
        old_total = asset.quantity * asset.average_price
        new_total = old_total + (request.quantity * request.price)
        new_quantity = asset.quantity + request.quantity
        asset.average_price = new_total / new_quantity if new_quantity > 0 else Decimal("0")
        asset.quantity = new_quantity

    elif request.transaction_type == TransactionType.SELL:
        if request.quantity > asset.quantity:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Quantidade a vender maior que a quantidade disponivel.",
            )
        asset.quantity = asset.quantity - request.quantity

    elif request.transaction_type == TransactionType.SPLIT:
        # For stock splits, multiply quantity by factor (stored in quantity field)
        asset.quantity = asset.quantity * request.quantity
        asset.average_price = asset.average_price / request.quantity

    db.add(transaction)
    await db.commit()
    await db.refresh(transaction)

    logger.info(
        "Transaction created",
        user_id=current_user.id,
        transaction_id=transaction.id,
        type=request.transaction_type.value,
    )

    return TransactionResponse(
        id=transaction.id,
        asset_id=transaction.asset_id,
        ticker=asset.ticker,
        asset_class=asset.asset_class,
        transaction_type=transaction.transaction_type,
        quantity=transaction.quantity,
        price=transaction.price,
        total_value=transaction.total_value,
        fees=transaction.fees,
        transaction_date=transaction.transaction_date,
        notes=transaction.notes,
        created_at=transaction.created_at,
    )


@router.delete("/{transaction_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_transaction(
    transaction_id: int,
    current_user: CurrentUser,
    db: DbSession,
) -> None:
    """Delete a transaction. Note: This does NOT revert asset changes."""
    # Verify ownership
    result = await db.execute(
        select(Transaction)
        .join(PortfolioAsset)
        .join(Portfolio)
        .where(
            Transaction.id == transaction_id,
            Portfolio.user_id == current_user.id,
        )
    )
    transaction = result.scalar_one_or_none()

    if not transaction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transacao nao encontrada.",
        )

    await db.delete(transaction)
    await db.commit()

    logger.info(
        "Transaction deleted",
        user_id=current_user.id,
        transaction_id=transaction_id,
    )
