"""Custom Reports API endpoints."""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import structlog
from io import BytesIO

from src.core.deps import CurrentUser, DbSession
from src.services.reports import ReportConfig, ReportsService, get_reports_service

router = APIRouter()
logger = structlog.get_logger()


class ReportRequest(BaseModel):
    """Report generation request."""
    report_type: str = Field(..., pattern="^(portfolio|performance|tax|backtest|forecast)$")
    format: str = Field(default="excel", pattern="^(csv|excel|json)$")
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    portfolio_ids: Optional[list[int]] = None
    include_charts: bool = True
    include_transactions: bool = True


@router.get("/types")
async def get_report_types(
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Get available report types."""
    service = get_reports_service(db)
    reports = await service.get_available_reports()

    return {
        "reports": reports,
        "note": "Use POST /reports/generate para gerar um relatório.",
    }


@router.post("/generate")
async def generate_report(
    request: ReportRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> StreamingResponse:
    """
    Generate a custom report.

    Available report types:
    - portfolio: Portfolio composition with current values
    - performance: Performance metrics
    - tax: Tax report for Brazilian investors (IR)
    - backtest: Backtest history
    - forecast: Forecast history

    Available formats:
    - csv: Comma-separated values
    - excel: Microsoft Excel (.xlsx)
    - json: JSON format
    """
    service = get_reports_service(db)

    config = ReportConfig(
        report_type=request.report_type,
        format=request.format,
        period_start=request.period_start,
        period_end=request.period_end,
        portfolio_ids=request.portfolio_ids,
        include_charts=request.include_charts,
        include_transactions=request.include_transactions,
    )

    try:
        result = await service.generate_report(current_user, config)

        logger.info(
            "Report generated",
            user_id=current_user.id,
            report_type=request.report_type,
            format=request.format,
        )

        return StreamingResponse(
            BytesIO(result.content),
            media_type=result.content_type,
            headers={
                "Content-Disposition": f"attachment; filename={result.filename}",
            },
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Failed to generate report", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Falha ao gerar relatório.",
        )
