"""Advanced analytics and AI endpoints."""
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
import structlog
import pandas as pd

from src.core.deps import CurrentUser, DbSession
from src.services.market import market_service
from src.models.analytics import (
    Backtest,
    BacktestStatus,
    PriceForecastHistory,
    StrategyRecommendationHistory,
)

router = APIRouter()
logger = structlog.get_logger()


# ===========================================
# Request/Response Schemas
# ===========================================

class BacktestRequest(BaseModel):
    """Backtest request."""
    strategy: str = Field(..., description="Strategy name: buy_and_hold, dca, rebalancing, momentum")
    tickers: list[str] = Field(..., min_length=1, max_length=10)
    start_date: date
    end_date: date
    initial_capital: float = Field(default=10000, ge=1000)
    allocation: Optional[dict[str, float]] = None


class BacktestResponse(BaseModel):
    """Backtest result response."""
    strategy_name: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    benchmark_return: Optional[float]
    alpha: Optional[float]
    daily_values: list[dict]
    disclaimer: str


class ForecastRequest(BaseModel):
    """Price forecast request."""
    ticker: str
    forecast_days: int = Field(default=30, ge=7, le=90)


class ForecastResponse(BaseModel):
    """Price forecast response."""
    ticker: str
    current_price: float
    forecast_date: str
    predicted_price: float
    predicted_change_pct: float
    confidence: float
    prediction_range: dict
    methodology: str
    factors: list[str]
    strategy_backtests: list[dict]
    disclaimer: str
    user_decision_required: bool = True


class TechnicalAnalysisResponse(BaseModel):
    """Technical analysis response."""
    ticker: str
    signals: list[dict]
    overall_signal: str
    overall_strength: float
    support_levels: list[float]
    resistance_levels: list[float]
    trend: str
    disclaimer: str


class RiskAnalysisRequest(BaseModel):
    """Risk analysis request."""
    tickers: list[str]
    allocation: Optional[dict[str, float]] = None


class RiskAnalysisResponse(BaseModel):
    """Risk analysis response."""
    volatility: float
    var_95: float
    var_99: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    risk_score: int
    risk_category: str
    diversification: dict
    disclaimer: str


class StrategyRecommendationRequest(BaseModel):
    """Strategy recommendation request."""
    risk_profile: str = Field(..., pattern="^(conservative|moderate|aggressive)$")
    investment_horizon_years: float = Field(..., ge=0.5, le=30)
    goals: list[str] = Field(default=[])
    current_portfolio_value: float = Field(default=10000, ge=0)
    tickers_for_backtest: Optional[list[str]] = None


class StrategyRecommendationResponse(BaseModel):
    """Strategy recommendation response."""
    investor_profile: str
    risk_tolerance: str
    investment_horizon: str
    recommendations: list[dict]
    primary_recommendation: dict
    portfolio_allocation: dict
    rebalance_frequency: str
    disclaimer: str


# ===========================================
# Backtesting Endpoints
# ===========================================

@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> BacktestResponse:
    """
    Run a backtest for a strategy.

    Strategies available:
    - buy_and_hold: Buy and hold all tickers equally
    - dca: Dollar cost averaging with monthly investments
    - rebalancing: Periodic rebalancing to target allocation
    - momentum: Buy winners, sell losers based on momentum
    """
    from src.ai.backtesting import (
        BacktestEngine,
        BuyAndHoldStrategy,
        DCAStrategy,
        RebalancingStrategy,
        MomentumStrategy,
    )

    # Fetch historical data
    price_data = {}
    for ticker in request.tickers:
        history = await market_service.get_history(
            ticker,
            period="2y",
            interval="1d",
        )
        if history:
            df = pd.DataFrame([{
                "date": h.date,
                "open": float(h.open),
                "high": float(h.high),
                "low": float(h.low),
                "close": float(h.close),
                "volume": h.volume,
            } for h in history])
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            price_data[ticker] = df

    if not price_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Não foi possível obter dados históricos para os tickers informados.",
        )

    # Create strategy
    if request.strategy == "buy_and_hold":
        strategy = BuyAndHoldStrategy(
            tickers=request.tickers,
            allocation=request.allocation,
        )
    elif request.strategy == "dca":
        strategy = DCAStrategy(
            tickers=request.tickers,
            investment_amount=Decimal(str(request.initial_capital / 12)),
            allocation=request.allocation,
        )
    elif request.strategy == "rebalancing":
        allocation = request.allocation or {t: 1.0/len(request.tickers) for t in request.tickers}
        strategy = RebalancingStrategy(target_allocation=allocation)
    elif request.strategy == "momentum":
        strategy = MomentumStrategy(tickers=request.tickers)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Estratégia não suportada: {request.strategy}",
        )

    # Run backtest
    engine = BacktestEngine(initial_capital=Decimal(str(request.initial_capital)))

    try:
        result = await engine.run(
            strategy=strategy,
            price_data=price_data,
            start_date=request.start_date,
            end_date=request.end_date,
        )
    except Exception as e:
        logger.error("Backtest failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao executar backtest: {str(e)}",
        )

    # Save backtest to database
    backtest_record = Backtest(
        user_id=current_user.id,
        strategy_name=result.strategy_name,
        strategy_params={"allocation": request.allocation} if request.allocation else None,
        tickers=request.tickers,
        start_date=datetime.combine(request.start_date, datetime.min.time()),
        end_date=datetime.combine(request.end_date, datetime.min.time()),
        initial_capital=Decimal(str(request.initial_capital)),
        status=BacktestStatus.COMPLETED,
        final_value=Decimal(str(request.initial_capital * (1 + result.total_return / 100))),
        total_return=Decimal(str(result.total_return)),
        annualized_return=Decimal(str(result.annualized_return)),
        volatility=Decimal(str(result.volatility)),
        sharpe_ratio=Decimal(str(result.sharpe_ratio)),
        max_drawdown=Decimal(str(result.max_drawdown)),
        win_rate=Decimal(str(result.win_rate)),
        total_trades=result.total_trades,
        daily_values=result.daily_values,
        completed_at=datetime.utcnow(),
    )
    db.add(backtest_record)
    await db.commit()

    return BacktestResponse(
        strategy_name=result.strategy_name,
        total_return=result.total_return,
        annualized_return=result.annualized_return,
        volatility=result.volatility,
        sharpe_ratio=result.sharpe_ratio,
        max_drawdown=result.max_drawdown,
        win_rate=result.win_rate,
        total_trades=result.total_trades,
        benchmark_return=result.benchmark_return,
        alpha=result.alpha,
        daily_values=result.daily_values,
        disclaimer=(
            "⚠️ Resultados de backtest são baseados em dados históricos e NÃO garantem "
            "resultados futuros. O mercado é imprevisível. Use apenas como referência educacional."
        ),
    )


@router.get("/backtest/strategies")
async def list_backtest_strategies() -> dict:
    """List available backtest strategies with descriptions."""
    return {
        "strategies": [
            {
                "id": "buy_and_hold",
                "name": "Buy and Hold",
                "description": "Comprar e manter ativos a longo prazo",
                "risk_level": "moderate",
                "recommended_for": ["moderate", "aggressive"],
            },
            {
                "id": "dca",
                "name": "DCA (Custo Médio)",
                "description": "Investir valores fixos periodicamente",
                "risk_level": "low",
                "recommended_for": ["conservative", "moderate"],
            },
            {
                "id": "rebalancing",
                "name": "Rebalanceamento",
                "description": "Manter alocação-alvo com ajustes periódicos",
                "risk_level": "moderate",
                "recommended_for": ["moderate", "aggressive"],
            },
            {
                "id": "momentum",
                "name": "Momentum",
                "description": "Seguir tendências de alta e baixa",
                "risk_level": "high",
                "recommended_for": ["aggressive"],
            },
        ],
    }


# ===========================================
# Price Forecast Endpoints
# ===========================================

@router.post("/forecast", response_model=ForecastResponse)
async def get_price_forecast(
    request: ForecastRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> ForecastResponse:
    """
    Get price forecast for an asset.

    WARNING: Forecasts are for educational purposes only.
    They are NOT investment recommendations.
    """
    from src.ai.predictive.price_forecast import forecast_engine

    # Fetch historical data
    history = await market_service.get_history(
        request.ticker,
        period="2y",
        interval="1d",
    )

    if not history or len(history) < 60:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dados históricos insuficientes para gerar previsão.",
        )

    # Convert to DataFrame
    df = pd.DataFrame([{
        "date": h.date,
        "open": float(h.open),
        "high": float(h.high),
        "low": float(h.low),
        "close": float(h.close),
        "volume": h.volume,
    } for h in history])
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Generate forecast
    forecast = await forecast_engine.generate_forecast(
        ticker=request.ticker,
        price_data=df,
        forecast_days=request.forecast_days,
        include_backtests=True,
    )

    # Save forecast to database
    forecast_record = PriceForecastHistory(
        user_id=current_user.id,
        ticker=request.ticker,
        current_price=Decimal(str(forecast.current_price)),
        forecast_date=datetime.combine(forecast.forecast_date, datetime.min.time()),
        forecast_days=request.forecast_days,
        predicted_price=Decimal(str(forecast.predicted_price)),
        predicted_change_pct=Decimal(str(forecast.predicted_change_pct)),
        confidence=Decimal(str(forecast.confidence)),
        prediction_low=Decimal(str(forecast.prediction_range[0])),
        prediction_high=Decimal(str(forecast.prediction_range[1])),
        methodology=forecast.methodology,
        factors=forecast.factors,
        strategy_backtests=forecast.strategy_backtests,
    )
    db.add(forecast_record)
    await db.commit()

    result = forecast.to_dict()
    return ForecastResponse(**result)


# ===========================================
# Technical Analysis Endpoints
# ===========================================

@router.get("/technical/{ticker}", response_model=TechnicalAnalysisResponse)
async def get_technical_analysis(
    ticker: str,
    current_user: CurrentUser,
    db: DbSession,
) -> TechnicalAnalysisResponse:
    """Get technical analysis for an asset."""
    from src.ai.predictive.technical_analysis import TechnicalAnalyzer

    # Fetch historical data
    history = await market_service.get_history(ticker, period="1y", interval="1d")

    if not history or len(history) < 60:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dados históricos insuficientes para análise técnica.",
        )

    # Convert to DataFrame
    df = pd.DataFrame([{
        "date": h.date,
        "open": float(h.open),
        "high": float(h.high),
        "low": float(h.low),
        "close": float(h.close),
        "volume": h.volume,
    } for h in history])
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Run analysis
    analyzer = TechnicalAnalyzer()
    result = analyzer.analyze(ticker, df)

    return TechnicalAnalysisResponse(**result.to_dict())


# ===========================================
# Risk Analysis Endpoints
# ===========================================

@router.post("/risk", response_model=RiskAnalysisResponse)
async def analyze_portfolio_risk(
    request: RiskAnalysisRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> RiskAnalysisResponse:
    """Analyze risk metrics for a portfolio."""
    from src.ai.predictive.risk_metrics import RiskAnalyzer

    # Fetch historical data and calculate portfolio values
    price_data = {}
    for ticker in request.tickers:
        history = await market_service.get_history(ticker, period="1y", interval="1d")
        if history:
            price_data[ticker] = [float(h.close) for h in history]

    if not price_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Não foi possível obter dados para análise de risco.",
        )

    # Calculate portfolio values
    allocation = request.allocation or {t: 1.0/len(request.tickers) for t in request.tickers}
    min_length = min(len(v) for v in price_data.values())

    portfolio_values = []
    for i in range(min_length):
        value = sum(
            price_data[t][i] * allocation.get(t, 0)
            for t in request.tickers
        )
        portfolio_values.append(value * 10000)  # Normalize to 10k base

    # Analyze risk
    analyzer = RiskAnalyzer()
    risk_metrics = analyzer.analyze_portfolio_risk(portfolio_values)

    # Analyze diversification
    positions = [
        {"ticker": t, "value": allocation.get(t, 0) * 10000, "sector": "N/A", "asset_class": "acao"}
        for t in request.tickers
    ]
    diversification = analyzer.analyze_diversification(positions)

    return RiskAnalysisResponse(
        volatility=risk_metrics.volatility,
        var_95=risk_metrics.var_95,
        var_99=risk_metrics.var_99,
        max_drawdown=risk_metrics.max_drawdown,
        sharpe_ratio=risk_metrics.sharpe_ratio,
        sortino_ratio=risk_metrics.sortino_ratio,
        risk_score=risk_metrics.risk_score,
        risk_category=risk_metrics.risk_category,
        diversification=diversification.to_dict(),
        disclaimer=(
            "Métricas de risco são baseadas em dados históricos. "
            "Volatilidade e perdas futuras podem ser maiores."
        ),
    )


# ===========================================
# Strategy Recommendation Endpoints
# ===========================================

@router.post("/recommendations", response_model=StrategyRecommendationResponse)
async def get_strategy_recommendations(
    request: StrategyRecommendationRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> StrategyRecommendationResponse:
    """
    Get personalized strategy recommendations based on investor profile.

    The AI analyzes your profile and backtest results to suggest
    the most suitable strategies for your goals.
    """
    from src.ai.strategy_recommendation import strategy_engine

    # Fetch historical data if tickers provided
    historical_data = None
    if request.tickers_for_backtest:
        historical_data = {}
        for ticker in request.tickers_for_backtest:
            history = await market_service.get_history(ticker, period="2y", interval="1d")
            if history:
                df = pd.DataFrame([{
                    "date": h.date,
                    "open": float(h.open),
                    "high": float(h.high),
                    "low": float(h.low),
                    "close": float(h.close),
                    "volume": h.volume,
                } for h in history])
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                historical_data[ticker] = df

    # Get recommendations
    recommendations = await strategy_engine.get_recommendations(
        risk_profile=request.risk_profile,
        investment_horizon_years=request.investment_horizon_years,
        goals=request.goals,
        current_portfolio_value=Decimal(str(request.current_portfolio_value)),
        historical_data=historical_data,
    )

    return StrategyRecommendationResponse(**recommendations.to_dict())


# ===========================================
# RAG / Knowledge Endpoints
# ===========================================

@router.get("/knowledge/search")
async def search_financial_knowledge(
    query: str = Query(..., min_length=3),
    category: Optional[str] = None,
    current_user: CurrentUser = None,
) -> dict:
    """Search financial knowledge base."""
    from src.ai.rag.financial_rag import rag_service

    results = await rag_service.search_market_knowledge(
        query=query,
        category=category,
        limit=5,
    )

    education = await rag_service.search_education(query=query, limit=3)

    return {
        "market_knowledge": results,
        "education": education,
        "query": query,
    }


@router.post("/knowledge/context")
async def get_enriched_context(
    query: str,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Get enriched context for AI responses."""
    from src.ai.rag.financial_rag import rag_service

    # Get user profile if available
    user_profile = None
    if current_user.investor_profile:
        user_profile = {
            "risk_profile": current_user.investor_profile.risk_profile,
            "investment_horizon": current_user.investor_profile.investment_horizon,
            "experience_level": current_user.investor_profile.experience_level,
        }

    context = await rag_service.get_enriched_context(
        query=query,
        user_id=current_user.id,
        user_profile=user_profile,
    )

    return {
        "context": context,
        "query": query,
    }


# ===========================================
# History Endpoints
# ===========================================

@router.get("/backtest/history")
async def get_backtest_history(
    current_user: CurrentUser,
    db: DbSession,
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """Get user's backtest history."""
    query = (
        select(Backtest)
        .where(Backtest.user_id == current_user.id)
        .order_by(Backtest.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(query)
    backtests = result.scalars().all()

    return {
        "backtests": [
            {
                "id": b.id,
                "strategy_name": b.strategy_name,
                "tickers": b.tickers,
                "start_date": b.start_date.isoformat(),
                "end_date": b.end_date.isoformat(),
                "initial_capital": float(b.initial_capital),
                "status": b.status.value,
                "total_return": float(b.total_return) if b.total_return else None,
                "sharpe_ratio": float(b.sharpe_ratio) if b.sharpe_ratio else None,
                "max_drawdown": float(b.max_drawdown) if b.max_drawdown else None,
                "created_at": b.created_at.isoformat(),
            }
            for b in backtests
        ],
        "total": len(backtests),
        "limit": limit,
        "offset": offset,
    }


@router.get("/backtest/{backtest_id}")
async def get_backtest_detail(
    backtest_id: int,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Get detailed backtest result."""
    query = select(Backtest).where(
        Backtest.id == backtest_id,
        Backtest.user_id == current_user.id,
    )
    result = await db.execute(query)
    backtest = result.scalar_one_or_none()

    if not backtest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Backtest não encontrado.",
        )

    return {
        "id": backtest.id,
        "strategy_name": backtest.strategy_name,
        "strategy_params": backtest.strategy_params,
        "tickers": backtest.tickers,
        "start_date": backtest.start_date.isoformat(),
        "end_date": backtest.end_date.isoformat(),
        "initial_capital": float(backtest.initial_capital),
        "status": backtest.status.value,
        "final_value": float(backtest.final_value) if backtest.final_value else None,
        "total_return": float(backtest.total_return) if backtest.total_return else None,
        "annualized_return": float(backtest.annualized_return) if backtest.annualized_return else None,
        "volatility": float(backtest.volatility) if backtest.volatility else None,
        "sharpe_ratio": float(backtest.sharpe_ratio) if backtest.sharpe_ratio else None,
        "max_drawdown": float(backtest.max_drawdown) if backtest.max_drawdown else None,
        "win_rate": float(backtest.win_rate) if backtest.win_rate else None,
        "total_trades": backtest.total_trades,
        "daily_values": backtest.daily_values,
        "trades": backtest.trades,
        "created_at": backtest.created_at.isoformat(),
        "completed_at": backtest.completed_at.isoformat() if backtest.completed_at else None,
    }


@router.get("/forecast/history")
async def get_forecast_history(
    current_user: CurrentUser,
    db: DbSession,
    ticker: Optional[str] = None,
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """Get user's forecast history."""
    query = select(PriceForecastHistory).where(
        PriceForecastHistory.user_id == current_user.id
    )

    if ticker:
        query = query.where(PriceForecastHistory.ticker == ticker.upper())

    query = query.order_by(PriceForecastHistory.created_at.desc()).offset(offset).limit(limit)

    result = await db.execute(query)
    forecasts = result.scalars().all()

    return {
        "forecasts": [
            {
                "id": f.id,
                "ticker": f.ticker,
                "current_price": float(f.current_price),
                "predicted_price": float(f.predicted_price),
                "predicted_change_pct": float(f.predicted_change_pct),
                "confidence": float(f.confidence),
                "forecast_date": f.forecast_date.isoformat(),
                "methodology": f.methodology,
                "actual_price": float(f.actual_price) if f.actual_price else None,
                "accuracy_pct": float(f.accuracy_pct) if f.accuracy_pct else None,
                "created_at": f.created_at.isoformat(),
            }
            for f in forecasts
        ],
        "total": len(forecasts),
        "limit": limit,
        "offset": offset,
    }


@router.get("/forecast/{forecast_id}")
async def get_forecast_detail(
    forecast_id: int,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Get detailed forecast result."""
    query = select(PriceForecastHistory).where(
        PriceForecastHistory.id == forecast_id,
        PriceForecastHistory.user_id == current_user.id,
    )
    result = await db.execute(query)
    forecast = result.scalar_one_or_none()

    if not forecast:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Previsão não encontrada.",
        )

    return {
        "id": forecast.id,
        "ticker": forecast.ticker,
        "current_price": float(forecast.current_price),
        "forecast_date": forecast.forecast_date.isoformat(),
        "forecast_days": forecast.forecast_days,
        "predicted_price": float(forecast.predicted_price),
        "predicted_change_pct": float(forecast.predicted_change_pct),
        "confidence": float(forecast.confidence),
        "prediction_range": {
            "low": float(forecast.prediction_low),
            "high": float(forecast.prediction_high),
        },
        "methodology": forecast.methodology,
        "factors": forecast.factors,
        "strategy_backtests": forecast.strategy_backtests,
        "actual_price": float(forecast.actual_price) if forecast.actual_price else None,
        "accuracy_pct": float(forecast.accuracy_pct) if forecast.accuracy_pct else None,
        "created_at": forecast.created_at.isoformat(),
    }


@router.get("/recommendations/history")
async def get_recommendations_history(
    current_user: CurrentUser,
    db: DbSession,
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """Get user's strategy recommendation history."""
    query = (
        select(StrategyRecommendationHistory)
        .where(StrategyRecommendationHistory.user_id == current_user.id)
        .order_by(StrategyRecommendationHistory.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(query)
    recommendations = result.scalars().all()

    return {
        "recommendations": [
            {
                "id": r.id,
                "risk_profile": r.risk_profile,
                "investment_horizon_years": float(r.investment_horizon_years),
                "primary_strategy": r.primary_strategy,
                "suitability_score": float(r.suitability_score),
                "user_accepted": r.user_accepted,
                "created_at": r.created_at.isoformat(),
            }
            for r in recommendations
        ],
        "total": len(recommendations),
        "limit": limit,
        "offset": offset,
    }


# ===========================================
# ML Model Training Endpoints
# ===========================================

@router.post("/ml/train/{ticker}")
async def train_ml_model(
    ticker: str,
    current_user: CurrentUser,
    db: DbSession,
    force_retrain: bool = Query(default=False),
) -> dict:
    """
    Train ML models for a specific ticker.

    This endpoint trains LSTM, XGBoost, and Random Forest models
    for price prediction.
    """
    from src.ai.ml.model_trainer import model_trainer

    # Fetch historical data
    history = await market_service.get_history(
        ticker.upper(),
        period="2y",
        interval="1d",
    )

    if not history or len(history) < 120:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dados históricos insuficientes para treinar modelo (mínimo 120 dias).",
        )

    # Convert to DataFrame
    df = pd.DataFrame([{
        "date": h.date,
        "open": float(h.open),
        "high": float(h.high),
        "low": float(h.low),
        "close": float(h.close),
        "volume": h.volume,
    } for h in history])
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Train model
    result = await model_trainer.train_for_ticker(
        ticker=ticker.upper(),
        price_data=df,
        target_days=5,
        force_retrain=force_retrain,
    )

    return {
        "ticker": ticker.upper(),
        "model_name": result.model_name,
        "status": result.status,
        "samples_used": result.samples_used,
        "training_date": result.training_date.isoformat(),
        "metrics": result.metrics,
        "error": result.error,
    }


@router.get("/ml/models")
async def list_trained_models(
    current_user: CurrentUser,
) -> dict:
    """List all trained ML models."""
    from src.ai.ml.model_trainer import model_trainer

    models = model_trainer.list_trained_models()

    return {
        "models": models,
        "total": len(models),
    }
