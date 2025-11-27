"""Export endpoints for portfolio data."""
import io
from datetime import datetime
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import selectinload
import pandas as pd
import structlog

from src.core.deps import CurrentUser, DbSession
from src.models.portfolio import Portfolio, PortfolioAsset, Transaction
from src.models.alert import PriceAlert

router = APIRouter()
logger = structlog.get_logger()


def decimal_to_float(obj):
    """Convert Decimal objects to float for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    return obj


@router.get("/portfolio/excel")
async def export_portfolio_excel(
    current_user: CurrentUser,
    db: DbSession,
) -> StreamingResponse:
    """Export portfolio data to Excel format."""
    # Get user's portfolio with assets
    result = await db.execute(
        select(Portfolio)
        .where(Portfolio.user_id == current_user.id)
        .options(selectinload(Portfolio.assets))
    )
    portfolio = result.scalar_one_or_none()

    if not portfolio or not portfolio.assets:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nenhum ativo encontrado na carteira.",
        )

    # Prepare data for DataFrame
    assets_data = []
    for asset in portfolio.assets:
        total_invested = float(asset.quantity) * float(asset.average_price)
        assets_data.append({
            "Ticker": asset.ticker,
            "Classe": asset.asset_class.value,
            "Quantidade": float(asset.quantity),
            "Preco Medio (R$)": float(asset.average_price),
            "Total Investido (R$)": total_invested,
            "Corretora": asset.broker or "-",
            "Data de Inclusao": asset.created_at.strftime("%d/%m/%Y"),
        })

    df = pd.DataFrame(assets_data)

    # Create Excel file
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Carteira", index=False)

        # Auto-adjust column widths
        worksheet = writer.sheets["Carteira"]
        for idx, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = max_len

    output.seek(0)

    filename = f"carteira_{current_user.id}_{datetime.now().strftime('%Y%m%d')}.xlsx"

    logger.info(
        "Portfolio exported to Excel",
        user_id=current_user.id,
        assets_count=len(assets_data),
    )

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/portfolio/csv")
async def export_portfolio_csv(
    current_user: CurrentUser,
    db: DbSession,
) -> StreamingResponse:
    """Export portfolio data to CSV format."""
    # Get user's portfolio with assets
    result = await db.execute(
        select(Portfolio)
        .where(Portfolio.user_id == current_user.id)
        .options(selectinload(Portfolio.assets))
    )
    portfolio = result.scalar_one_or_none()

    if not portfolio or not portfolio.assets:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nenhum ativo encontrado na carteira.",
        )

    # Prepare data for DataFrame
    assets_data = []
    for asset in portfolio.assets:
        total_invested = float(asset.quantity) * float(asset.average_price)
        assets_data.append({
            "Ticker": asset.ticker,
            "Classe": asset.asset_class.value,
            "Quantidade": float(asset.quantity),
            "Preco Medio (R$)": float(asset.average_price),
            "Total Investido (R$)": total_invested,
            "Corretora": asset.broker or "",
            "Data de Inclusao": asset.created_at.strftime("%d/%m/%Y"),
        })

    df = pd.DataFrame(assets_data)

    # Create CSV
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    filename = f"carteira_{current_user.id}_{datetime.now().strftime('%Y%m%d')}.csv"

    logger.info(
        "Portfolio exported to CSV",
        user_id=current_user.id,
        assets_count=len(assets_data),
    )

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/transactions/excel")
async def export_transactions_excel(
    current_user: CurrentUser,
    db: DbSession,
    start_date: datetime | None = Query(None, description="Filter transactions from this date"),
    end_date: datetime | None = Query(None, description="Filter transactions until this date"),
) -> StreamingResponse:
    """Export transaction history to Excel format."""
    # Get user's portfolio
    portfolio_result = await db.execute(
        select(Portfolio).where(Portfolio.user_id == current_user.id)
    )
    portfolio = portfolio_result.scalar_one_or_none()

    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nenhuma transacao encontrada.",
        )

    # Get transactions
    query = (
        select(Transaction)
        .join(PortfolioAsset)
        .where(PortfolioAsset.portfolio_id == portfolio.id)
        .options(selectinload(Transaction.asset))
        .order_by(Transaction.transaction_date.desc())
    )

    if start_date:
        query = query.where(Transaction.transaction_date >= start_date)
    if end_date:
        query = query.where(Transaction.transaction_date <= end_date)

    result = await db.execute(query)
    transactions = result.scalars().all()

    if not transactions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nenhuma transacao encontrada no periodo.",
        )

    # Prepare data for DataFrame
    transaction_type_labels = {
        "buy": "Compra",
        "sell": "Venda",
        "dividend": "Dividendo",
        "split": "Desdobramento",
    }

    transactions_data = []
    for t in transactions:
        transactions_data.append({
            "Data": t.transaction_date.strftime("%d/%m/%Y"),
            "Ticker": t.asset.ticker,
            "Classe": t.asset.asset_class.value,
            "Tipo": transaction_type_labels.get(t.transaction_type.value, t.transaction_type.value),
            "Quantidade": float(t.quantity),
            "Preco (R$)": float(t.price),
            "Taxas (R$)": float(t.fees),
            "Total (R$)": float(t.total_value),
            "Notas": t.notes or "",
        })

    df = pd.DataFrame(transactions_data)

    # Create Excel file
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Transacoes", index=False)

        # Auto-adjust column widths
        worksheet = writer.sheets["Transacoes"]
        for idx, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = max_len

    output.seek(0)

    filename = f"transacoes_{current_user.id}_{datetime.now().strftime('%Y%m%d')}.xlsx"

    logger.info(
        "Transactions exported to Excel",
        user_id=current_user.id,
        transactions_count=len(transactions_data),
    )

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/transactions/csv")
async def export_transactions_csv(
    current_user: CurrentUser,
    db: DbSession,
    start_date: datetime | None = Query(None, description="Filter transactions from this date"),
    end_date: datetime | None = Query(None, description="Filter transactions until this date"),
) -> StreamingResponse:
    """Export transaction history to CSV format."""
    # Get user's portfolio
    portfolio_result = await db.execute(
        select(Portfolio).where(Portfolio.user_id == current_user.id)
    )
    portfolio = portfolio_result.scalar_one_or_none()

    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nenhuma transacao encontrada.",
        )

    # Get transactions
    query = (
        select(Transaction)
        .join(PortfolioAsset)
        .where(PortfolioAsset.portfolio_id == portfolio.id)
        .options(selectinload(Transaction.asset))
        .order_by(Transaction.transaction_date.desc())
    )

    if start_date:
        query = query.where(Transaction.transaction_date >= start_date)
    if end_date:
        query = query.where(Transaction.transaction_date <= end_date)

    result = await db.execute(query)
    transactions = result.scalars().all()

    if not transactions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nenhuma transacao encontrada no periodo.",
        )

    # Prepare data for DataFrame
    transaction_type_labels = {
        "buy": "Compra",
        "sell": "Venda",
        "dividend": "Dividendo",
        "split": "Desdobramento",
    }

    transactions_data = []
    for t in transactions:
        transactions_data.append({
            "Data": t.transaction_date.strftime("%d/%m/%Y"),
            "Ticker": t.asset.ticker,
            "Classe": t.asset.asset_class.value,
            "Tipo": transaction_type_labels.get(t.transaction_type.value, t.transaction_type.value),
            "Quantidade": float(t.quantity),
            "Preco (R$)": float(t.price),
            "Taxas (R$)": float(t.fees),
            "Total (R$)": float(t.total_value),
            "Notas": t.notes or "",
        })

    df = pd.DataFrame(transactions_data)

    # Create CSV
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    filename = f"transacoes_{current_user.id}_{datetime.now().strftime('%Y%m%d')}.csv"

    logger.info(
        "Transactions exported to CSV",
        user_id=current_user.id,
        transactions_count=len(transactions_data),
    )

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
