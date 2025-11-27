"""
Custom Reports Service.

Generates comprehensive reports for portfolios, backtests, and analytics.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from io import BytesIO
from typing import Optional
import json

import pandas as pd
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.portfolio import Portfolio, PortfolioAsset, Transaction
from src.models.analytics import Backtest, PriceForecastHistory, StrategyRecommendationHistory
from src.models.user import User
from src.services.market import market_service

logger = structlog.get_logger()


@dataclass
class ReportConfig:
    """Report configuration."""
    report_type: str  # portfolio, performance, tax, backtest, forecast
    format: str  # pdf, excel, csv, json
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    portfolio_ids: Optional[list[int]] = None
    include_charts: bool = True
    include_transactions: bool = True
    language: str = "pt_BR"


@dataclass
class ReportResult:
    """Report generation result."""
    filename: str
    content: bytes
    content_type: str
    generated_at: datetime


class ReportsService:
    """Service for generating custom reports."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def generate_report(
        self,
        user: User,
        config: ReportConfig,
    ) -> ReportResult:
        """
        Generate a custom report based on configuration.

        Args:
            user: User requesting the report
            config: Report configuration

        Returns:
            ReportResult with file content
        """
        if config.report_type == "portfolio":
            return await self._generate_portfolio_report(user, config)
        elif config.report_type == "performance":
            return await self._generate_performance_report(user, config)
        elif config.report_type == "tax":
            return await self._generate_tax_report(user, config)
        elif config.report_type == "backtest":
            return await self._generate_backtest_report(user, config)
        elif config.report_type == "forecast":
            return await self._generate_forecast_report(user, config)
        else:
            raise ValueError(f"Unknown report type: {config.report_type}")

    async def _generate_portfolio_report(
        self,
        user: User,
        config: ReportConfig,
    ) -> ReportResult:
        """Generate portfolio report."""
        # Get portfolios
        query = select(Portfolio).where(Portfolio.user_id == user.id)
        if config.portfolio_ids:
            query = query.where(Portfolio.id.in_(config.portfolio_ids))
        result = await self.db.execute(query)
        portfolios = result.scalars().all()

        # Prepare data
        data = []
        for portfolio in portfolios:
            for asset in portfolio.assets:
                current_price = await market_service.get_current_price(asset.ticker)
                current_value = asset.quantity * (current_price or asset.average_price)
                profit_loss = current_value - (asset.quantity * asset.average_price)

                data.append({
                    "Portfolio": portfolio.name,
                    "Ticker": asset.ticker,
                    "Classe": asset.asset_class.value,
                    "Quantidade": float(asset.quantity),
                    "Preço Médio": float(asset.average_price),
                    "Preço Atual": float(current_price) if current_price else None,
                    "Valor Investido": float(asset.quantity * asset.average_price),
                    "Valor Atual": float(current_value),
                    "Lucro/Prejuízo": float(profit_loss),
                    "Lucro/Prejuízo %": float(profit_loss / (asset.quantity * asset.average_price) * 100) if asset.average_price else 0,
                    "Corretora": asset.broker,
                })

        df = pd.DataFrame(data)

        # Generate output based on format
        return await self._export_dataframe(df, "portfolio_report", config.format)

    async def _generate_performance_report(
        self,
        user: User,
        config: ReportConfig,
    ) -> ReportResult:
        """Generate performance report."""
        # Get portfolios
        query = select(Portfolio).where(Portfolio.user_id == user.id)
        if config.portfolio_ids:
            query = query.where(Portfolio.id.in_(config.portfolio_ids))
        result = await self.db.execute(query)
        portfolios = result.scalars().all()

        # Calculate performance metrics
        performance_data = []
        for portfolio in portfolios:
            total_invested = sum(a.quantity * a.average_price for a in portfolio.assets)
            total_current = Decimal("0")

            for asset in portfolio.assets:
                current_price = await market_service.get_current_price(asset.ticker)
                total_current += asset.quantity * (current_price or asset.average_price)

            profit_loss = total_current - total_invested
            profit_loss_pct = (profit_loss / total_invested * 100) if total_invested > 0 else 0

            performance_data.append({
                "Portfolio": portfolio.name,
                "Tipo": portfolio.portfolio_type.value,
                "Total Investido": float(total_invested),
                "Valor Atual": float(total_current),
                "Lucro/Prejuízo": float(profit_loss),
                "Lucro/Prejuízo %": float(profit_loss_pct),
                "Número de Ativos": len(portfolio.assets),
                "Meta de Valor": float(portfolio.target_value) if portfolio.target_value else None,
                "Progresso para Meta %": float(total_current / portfolio.target_value * 100) if portfolio.target_value else None,
            })

        df = pd.DataFrame(performance_data)
        return await self._export_dataframe(df, "performance_report", config.format)

    async def _generate_tax_report(
        self,
        user: User,
        config: ReportConfig,
    ) -> ReportResult:
        """Generate tax report for Brazilian investors (IR)."""
        # Get portfolios and transactions
        query = select(Portfolio).where(Portfolio.user_id == user.id)
        result = await self.db.execute(query)
        portfolios = result.scalars().all()

        # Get transactions within period
        transactions_data = []
        for portfolio in portfolios:
            for asset in portfolio.assets:
                trans_query = select(Transaction).where(Transaction.asset_id == asset.id)
                if config.period_start:
                    trans_query = trans_query.where(Transaction.transaction_date >= config.period_start)
                if config.period_end:
                    trans_query = trans_query.where(Transaction.transaction_date <= config.period_end)

                trans_result = await self.db.execute(trans_query)
                transactions = trans_result.scalars().all()

                for t in transactions:
                    transactions_data.append({
                        "Portfolio": portfolio.name,
                        "Ticker": asset.ticker,
                        "Classe": asset.asset_class.value,
                        "Tipo": t.transaction_type.value,
                        "Data": t.transaction_date.strftime("%d/%m/%Y"),
                        "Quantidade": float(t.quantity),
                        "Preço": float(t.price),
                        "Valor Total": float(t.total_value),
                        "Taxas": float(t.fees),
                        "Valor Líquido": float(t.total_value - t.fees) if t.transaction_type.value == "sell" else float(t.total_value + t.fees),
                    })

        df = pd.DataFrame(transactions_data) if transactions_data else pd.DataFrame()

        # Add summary section
        if not df.empty:
            # Calculate tax summary
            sells = df[df["Tipo"] == "sell"]
            total_vendas = sells["Valor Total"].sum() if not sells.empty else 0
            total_taxas = df["Taxas"].sum()

            summary = pd.DataFrame([{
                "Descrição": "RESUMO PARA IMPOSTO DE RENDA",
                "Valor": "",
            }, {
                "Descrição": "Total de Vendas no Período",
                "Valor": f"R$ {total_vendas:,.2f}",
            }, {
                "Descrição": "Total de Taxas/Custos",
                "Valor": f"R$ {total_taxas:,.2f}",
            }, {
                "Descrição": "",
                "Valor": "",
            }, {
                "Descrição": "NOTA: Este relatório é apenas informativo.",
                "Valor": "Consulte um contador.",
            }])

            # Combine with transactions
            df = pd.concat([summary, df], ignore_index=True)

        return await self._export_dataframe(df, "tax_report", config.format)

    async def _generate_backtest_report(
        self,
        user: User,
        config: ReportConfig,
    ) -> ReportResult:
        """Generate backtest history report."""
        query = (
            select(Backtest)
            .where(Backtest.user_id == user.id)
            .order_by(Backtest.created_at.desc())
        )
        if config.period_start:
            query = query.where(Backtest.created_at >= config.period_start)
        if config.period_end:
            query = query.where(Backtest.created_at <= config.period_end)

        result = await self.db.execute(query)
        backtests = result.scalars().all()

        data = []
        for b in backtests:
            data.append({
                "ID": b.id,
                "Estratégia": b.strategy_name,
                "Tickers": ", ".join(b.tickers) if b.tickers else "",
                "Período Início": b.start_date.strftime("%d/%m/%Y"),
                "Período Fim": b.end_date.strftime("%d/%m/%Y"),
                "Capital Inicial": float(b.initial_capital),
                "Valor Final": float(b.final_value) if b.final_value else None,
                "Retorno Total %": float(b.total_return) if b.total_return else None,
                "Retorno Anualizado %": float(b.annualized_return) if b.annualized_return else None,
                "Volatilidade %": float(b.volatility) if b.volatility else None,
                "Sharpe Ratio": float(b.sharpe_ratio) if b.sharpe_ratio else None,
                "Max Drawdown %": float(b.max_drawdown) if b.max_drawdown else None,
                "Win Rate %": float(b.win_rate) if b.win_rate else None,
                "Total de Trades": b.total_trades,
                "Status": b.status.value,
                "Data Execução": b.created_at.strftime("%d/%m/%Y %H:%M"),
            })

        df = pd.DataFrame(data)
        return await self._export_dataframe(df, "backtest_report", config.format)

    async def _generate_forecast_report(
        self,
        user: User,
        config: ReportConfig,
    ) -> ReportResult:
        """Generate forecast history report."""
        query = (
            select(PriceForecastHistory)
            .where(PriceForecastHistory.user_id == user.id)
            .order_by(PriceForecastHistory.created_at.desc())
        )
        if config.period_start:
            query = query.where(PriceForecastHistory.created_at >= config.period_start)
        if config.period_end:
            query = query.where(PriceForecastHistory.created_at <= config.period_end)

        result = await self.db.execute(query)
        forecasts = result.scalars().all()

        data = []
        for f in forecasts:
            data.append({
                "ID": f.id,
                "Ticker": f.ticker,
                "Data Previsão": f.forecast_date.strftime("%d/%m/%Y"),
                "Dias à Frente": f.forecast_days,
                "Preço na Data": float(f.current_price),
                "Preço Previsto": float(f.predicted_price),
                "Variação Prevista %": float(f.predicted_change_pct),
                "Confiança %": float(f.confidence * 100),
                "Preço Mínimo": float(f.prediction_low),
                "Preço Máximo": float(f.prediction_high),
                "Metodologia": f.methodology,
                "Preço Real": float(f.actual_price) if f.actual_price else "Aguardando",
                "Acurácia %": float(f.accuracy_pct) if f.accuracy_pct else "N/A",
                "Data Criação": f.created_at.strftime("%d/%m/%Y %H:%M"),
            })

        df = pd.DataFrame(data)
        return await self._export_dataframe(df, "forecast_report", config.format)

    async def _export_dataframe(
        self,
        df: pd.DataFrame,
        filename_base: str,
        format: str,
    ) -> ReportResult:
        """Export DataFrame to specified format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_base}_{timestamp}"

        if format == "csv":
            content = df.to_csv(index=False).encode("utf-8-sig")
            return ReportResult(
                filename=f"{filename}.csv",
                content=content,
                content_type="text/csv",
                generated_at=datetime.now(),
            )

        elif format == "excel":
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Dados")

                # Auto-adjust column widths
                worksheet = writer.sheets["Dados"]
                for idx, col in enumerate(df.columns):
                    max_length = max(
                        df[col].astype(str).map(len).max(),
                        len(col)
                    ) + 2
                    worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 50)

            content = output.getvalue()
            return ReportResult(
                filename=f"{filename}.xlsx",
                content=content,
                content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                generated_at=datetime.now(),
            )

        elif format == "json":
            content = df.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8")
            return ReportResult(
                filename=f"{filename}.json",
                content=content,
                content_type="application/json",
                generated_at=datetime.now(),
            )

        else:
            raise ValueError(f"Unsupported format: {format}")

    async def get_available_reports(self) -> list[dict]:
        """Get list of available report types."""
        return [
            {
                "type": "portfolio",
                "name": "Relatório de Carteira",
                "description": "Lista todos os ativos das carteiras com valores atuais",
                "formats": ["csv", "excel", "json"],
            },
            {
                "type": "performance",
                "name": "Relatório de Performance",
                "description": "Métricas de desempenho das carteiras",
                "formats": ["csv", "excel", "json"],
            },
            {
                "type": "tax",
                "name": "Relatório para Imposto de Renda",
                "description": "Transações e resumo para declaração de IR",
                "formats": ["csv", "excel"],
            },
            {
                "type": "backtest",
                "name": "Histórico de Backtests",
                "description": "Todos os backtests realizados com resultados",
                "formats": ["csv", "excel", "json"],
            },
            {
                "type": "forecast",
                "name": "Histórico de Previsões",
                "description": "Todas as previsões de preço realizadas",
                "formats": ["csv", "excel", "json"],
            },
        ]


# Global instance creator
def get_reports_service(db: AsyncSession) -> ReportsService:
    return ReportsService(db)
