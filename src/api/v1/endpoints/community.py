"""Community Strategies API endpoints."""
from datetime import datetime
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
import structlog

from src.core.deps import CurrentUser, DbSession
from src.services.community_strategies import (
    CommunityStrategiesService,
    get_community_service,
)

router = APIRouter()
logger = structlog.get_logger()


# ===========================================
# Schemas
# ===========================================

class StrategyResponse(BaseModel):
    """Community strategy response."""
    id: int
    strategy_name: str
    strategy_type: str
    description: Optional[str]
    target_risk_profile: str
    min_investment_horizon_years: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    times_used: int
    success_count: int
    avg_user_return: Optional[float]
    community_rating: Optional[float]
    is_featured: bool
    applicable_tickers: Optional[list[str]]


class StrategyMatchResponse(BaseModel):
    """Strategy match for user."""
    strategy: StrategyResponse
    match_score: float
    reasons: list[str]


class UseStrategyRequest(BaseModel):
    """Request to use a strategy."""
    initial_value: Decimal = Field(..., gt=0)


class UpdateOutcomeRequest(BaseModel):
    """Update strategy use outcome."""
    current_value: Decimal = Field(..., gt=0)
    is_active: bool = True
    rating: Optional[int] = Field(None, ge=1, le=5)
    feedback: Optional[str] = None


# ===========================================
# Endpoints
# ===========================================

@router.get("/recommended")
async def get_recommended_strategies(
    current_user: CurrentUser,
    db: DbSession,
    limit: int = Query(default=5, ge=1, le=20),
) -> dict:
    """
    Get recommended community strategies for the current user.

    Recommendations are based on the user's investor profile,
    risk tolerance, and investment horizon.
    """
    service = get_community_service(db)
    matches = await service.get_recommended_strategies(current_user, limit)

    return {
        "recommendations": [
            {
                "strategy": StrategyResponse(
                    id=m.strategy.id,
                    strategy_name=m.strategy.strategy_name,
                    strategy_type=m.strategy.strategy_type,
                    description=m.strategy.description,
                    target_risk_profile=m.strategy.target_risk_profile,
                    min_investment_horizon_years=float(m.strategy.min_investment_horizon_years),
                    total_return=float(m.strategy.total_return),
                    annualized_return=float(m.strategy.annualized_return),
                    sharpe_ratio=float(m.strategy.sharpe_ratio),
                    max_drawdown=float(m.strategy.max_drawdown),
                    volatility=float(m.strategy.volatility),
                    times_used=m.strategy.times_used,
                    success_count=m.strategy.success_count,
                    avg_user_return=float(m.strategy.avg_user_return) if m.strategy.avg_user_return else None,
                    community_rating=float(m.strategy.community_rating) if m.strategy.community_rating else None,
                    is_featured=m.strategy.is_featured,
                    applicable_tickers=m.strategy.applicable_tickers,
                ),
                "match_score": m.match_score,
                "reasons": m.reasons,
            }
            for m in matches
        ],
        "total": len(matches),
        "disclaimer": (
            "⚠️ Estratégias da comunidade são baseadas em backtests históricos. "
            "Resultados passados não garantem resultados futuros. "
            "A decisão de investir é de responsabilidade do usuário."
        ),
    }


@router.get("/featured")
async def get_featured_strategies(
    current_user: CurrentUser,
    db: DbSession,
    limit: int = Query(default=10, ge=1, le=50),
) -> dict:
    """Get featured community strategies."""
    service = get_community_service(db)
    strategies = await service.get_featured_strategies(limit)

    return {
        "strategies": [
            StrategyResponse(
                id=s.id,
                strategy_name=s.strategy_name,
                strategy_type=s.strategy_type,
                description=s.description,
                target_risk_profile=s.target_risk_profile,
                min_investment_horizon_years=float(s.min_investment_horizon_years),
                total_return=float(s.total_return),
                annualized_return=float(s.annualized_return),
                sharpe_ratio=float(s.sharpe_ratio),
                max_drawdown=float(s.max_drawdown),
                volatility=float(s.volatility),
                times_used=s.times_used,
                success_count=s.success_count,
                avg_user_return=float(s.avg_user_return) if s.avg_user_return else None,
                community_rating=float(s.community_rating) if s.community_rating else None,
                is_featured=s.is_featured,
                applicable_tickers=s.applicable_tickers,
            )
            for s in strategies
        ],
        "total": len(strategies),
    }


@router.get("/top")
async def get_top_strategies(
    current_user: CurrentUser,
    db: DbSession,
    sort_by: str = Query(default="rating", pattern="^(rating|return|success_rate|popular|sharpe_ratio)$"),
    limit: int = Query(default=10, ge=1, le=50),
) -> dict:
    """
    Get top performing community strategies.

    Sort options:
    - rating: By community rating (default)
    - return: By annualized return
    - success_rate: By success rate
    - popular: By times used
    - sharpe_ratio: By Sharpe ratio
    """
    service = get_community_service(db)
    strategies = await service.get_top_strategies(sort_by, limit)

    return {
        "strategies": [
            StrategyResponse(
                id=s.id,
                strategy_name=s.strategy_name,
                strategy_type=s.strategy_type,
                description=s.description,
                target_risk_profile=s.target_risk_profile,
                min_investment_horizon_years=float(s.min_investment_horizon_years),
                total_return=float(s.total_return),
                annualized_return=float(s.annualized_return),
                sharpe_ratio=float(s.sharpe_ratio),
                max_drawdown=float(s.max_drawdown),
                volatility=float(s.volatility),
                times_used=s.times_used,
                success_count=s.success_count,
                avg_user_return=float(s.avg_user_return) if s.avg_user_return else None,
                community_rating=float(s.community_rating) if s.community_rating else None,
                is_featured=s.is_featured,
                applicable_tickers=s.applicable_tickers,
            )
            for s in strategies
        ],
        "sort_by": sort_by,
        "total": len(strategies),
    }


@router.post("/{strategy_id}/use")
async def use_strategy(
    strategy_id: int,
    request: UseStrategyRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """
    Start using a community strategy.

    Records that the user is applying this strategy to track outcomes.
    """
    service = get_community_service(db)

    try:
        use = await service.use_strategy(
            strategy_id=strategy_id,
            user=current_user,
            initial_value=request.initial_value,
        )

        logger.info(
            "User started using strategy",
            user_id=current_user.id,
            strategy_id=strategy_id,
        )

        return {
            "use_id": use.id,
            "strategy_id": strategy_id,
            "initial_value": float(use.initial_value),
            "applied_at": use.applied_at.isoformat(),
            "message": "Estratégia aplicada com sucesso. Acompanhe seus resultados!",
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.put("/use/{use_id}/outcome")
async def update_outcome(
    use_id: int,
    request: UpdateOutcomeRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """
    Update the outcome of a strategy use.

    Call this periodically or when stopping the strategy to track performance.
    """
    service = get_community_service(db)

    try:
        use = await service.update_strategy_outcome(
            use_id=use_id,
            current_value=request.current_value,
            is_active=request.is_active,
            rating=request.rating,
            feedback=request.feedback,
        )

        return {
            "use_id": use.id,
            "current_value": float(use.current_value) if use.current_value else None,
            "return_pct": float(use.return_pct) if use.return_pct else None,
            "is_active": use.is_active,
            "message": "Resultado atualizado com sucesso!",
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


class StrategyUseResponse(BaseModel):
    """User's strategy use record."""
    id: int
    strategy_id: int
    strategy_name: str
    strategy_type: str
    initial_value: float
    current_value: Optional[float]
    return_pct: Optional[float]
    is_active: bool
    started_at: datetime
    ended_at: Optional[datetime]
    rating: Optional[int]
    feedback: Optional[str]


@router.get("/my-strategies")
async def get_my_strategies(
    current_user: CurrentUser,
    db: DbSession,
    active_only: bool = Query(default=False),
) -> dict:
    """
    Get all strategies the current user is using or has used.

    Returns the user's strategy history with performance metrics.
    """
    from sqlalchemy import select
    from src.models.community import CommunityStrategy, StrategyUse

    query = (
        select(StrategyUse, CommunityStrategy)
        .join(CommunityStrategy, StrategyUse.strategy_id == CommunityStrategy.id)
        .where(StrategyUse.user_id == current_user.id)
        .order_by(StrategyUse.applied_at.desc())
    )

    if active_only:
        query = query.where(StrategyUse.is_active == True)

    result = await db.execute(query)
    rows = result.all()

    strategies = [
        StrategyUseResponse(
            id=use.id,
            strategy_id=strategy.id,
            strategy_name=strategy.strategy_name,
            strategy_type=strategy.strategy_type,
            initial_value=float(use.initial_value),
            current_value=float(use.current_value) if use.current_value else None,
            return_pct=float(use.return_pct) if use.return_pct else None,
            is_active=use.is_active,
            started_at=use.applied_at,
            ended_at=use.stopped_at,
            rating=use.rating,
            feedback=use.feedback,
        )
        for use, strategy in rows
    ]

    active_count = sum(1 for s in strategies if s.is_active)
    total_invested = sum(s.initial_value for s in strategies if s.is_active)

    return {
        "strategies": strategies,
        "total": len(strategies),
        "active_count": active_count,
        "total_invested": total_invested,
    }


@router.get("/{strategy_id}")
async def get_strategy_details(
    strategy_id: int,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Get detailed information about a community strategy."""
    from sqlalchemy import select
    from src.models.community import CommunityStrategy

    query = select(CommunityStrategy).where(CommunityStrategy.id == strategy_id)
    result = await db.execute(query)
    strategy = result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Estratégia não encontrada.",
        )

    success_rate = (
        strategy.success_count / strategy.times_used * 100
        if strategy.times_used > 0
        else None
    )

    return {
        "strategy": StrategyResponse(
            id=strategy.id,
            strategy_name=strategy.strategy_name,
            strategy_type=strategy.strategy_type,
            description=strategy.description,
            target_risk_profile=strategy.target_risk_profile,
            min_investment_horizon_years=float(strategy.min_investment_horizon_years),
            total_return=float(strategy.total_return),
            annualized_return=float(strategy.annualized_return),
            sharpe_ratio=float(strategy.sharpe_ratio),
            max_drawdown=float(strategy.max_drawdown),
            volatility=float(strategy.volatility),
            times_used=strategy.times_used,
            success_count=strategy.success_count,
            avg_user_return=float(strategy.avg_user_return) if strategy.avg_user_return else None,
            community_rating=float(strategy.community_rating) if strategy.community_rating else None,
            is_featured=strategy.is_featured,
            applicable_tickers=strategy.applicable_tickers,
        ),
        "success_rate": success_rate,
        "strategy_params": strategy.strategy_params,
        "recommended_goals": strategy.recommended_goals,
        "created_at": strategy.created_at.isoformat(),
        "disclaimer": (
            "⚠️ Esta estratégia é baseada em backtest histórico. "
            "Resultados passados não garantem resultados futuros."
        ),
    }
