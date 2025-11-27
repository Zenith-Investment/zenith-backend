"""AI/ML background processing tasks."""
import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal

import structlog

from src.ai.llm.client import LLMClient
from src.core.database import get_celery_async_session
from src.models.portfolio import Portfolio, PortfolioAsset
from src.models.profile import InvestorProfile
from src.models.user import User
from src.workers.celery_app import celery_app
from sqlalchemy import select
from sqlalchemy.orm import selectinload

logger = structlog.get_logger()


def run_async(coro):
    """Helper to run async code in sync Celery tasks."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Risk score calculation based on assessment answers
RISK_SCORE_WEIGHTS = {
    "time_horizon": {
        "short_term": 10,
        "medium_term": 25,
        "long_term": 40,
        "very_long_term": 50,
    },
    "loss_reaction": {
        "sell_all": 5,
        "sell_some": 20,
        "hold": 35,
        "buy_more": 50,
    },
    "experience": {
        "none": 10,
        "beginner": 20,
        "intermediate": 35,
        "advanced": 50,
    },
    "income_stability": {
        "unstable": 10,
        "variable": 25,
        "stable": 40,
        "very_stable": 50,
    },
}

RISK_PROFILES = [
    (0, 25, "conservative"),
    (26, 40, "moderate"),
    (41, 55, "balanced"),
    (56, 75, "growth"),
    (76, 100, "aggressive"),
]

ALLOCATION_BY_PROFILE = {
    "conservative": {
        "fixed_income": 70,
        "stocks": 10,
        "fiis": 10,
        "cash": 10,
    },
    "moderate": {
        "fixed_income": 50,
        "stocks": 20,
        "fiis": 20,
        "cash": 10,
    },
    "balanced": {
        "fixed_income": 35,
        "stocks": 30,
        "fiis": 25,
        "cash": 10,
    },
    "growth": {
        "fixed_income": 20,
        "stocks": 45,
        "fiis": 25,
        "cash": 10,
    },
    "aggressive": {
        "fixed_income": 10,
        "stocks": 55,
        "fiis": 25,
        "cash": 10,
    },
}


def calculate_risk_score(assessment_data: dict) -> tuple[int, str]:
    """Calculate risk score from assessment answers."""
    score = 0
    weights_applied = 0

    for category, answers in RISK_SCORE_WEIGHTS.items():
        user_answer = assessment_data.get(category)
        if user_answer and user_answer in answers:
            score += answers[user_answer]
            weights_applied += 1

    # Normalize to 0-100
    if weights_applied > 0:
        score = int((score / (weights_applied * 50)) * 100)

    # Determine profile
    risk_profile = "balanced"  # Default
    for min_score, max_score, profile in RISK_PROFILES:
        if min_score <= score <= max_score:
            risk_profile = profile
            break

    return score, risk_profile


async def _generate_recommendations_async(user_id: int, portfolio_id: int) -> dict:
    """Generate AI-powered portfolio recommendations."""
    from src.services.market import market_service

    async with get_celery_async_session()() as session:
        # Get user with profile and portfolio
        result = await session.execute(
            select(User)
            .where(User.id == user_id)
            .options(
                selectinload(User.investor_profile),
                selectinload(User.portfolio).selectinload(Portfolio.assets),
            )
        )
        user = result.scalar_one_or_none()

        if not user:
            return {"error": "User not found"}

        if not user.portfolio or not user.portfolio.assets:
            return {"error": "No portfolio or assets found"}

        # Get current prices
        tickers = [asset.ticker for asset in user.portfolio.assets]
        quotes = await market_service.get_quotes(tickers)

        # Build portfolio summary
        total_value = Decimal("0")
        allocation = {}

        for asset in user.portfolio.assets:
            quote = quotes.get(asset.ticker.upper())
            current_value = (
                asset.quantity * quote.current_price
                if quote
                else asset.quantity * asset.average_price
            )
            total_value += current_value

            asset_class = asset.asset_class.value
            if asset_class not in allocation:
                allocation[asset_class] = Decimal("0")
            allocation[asset_class] += current_value

        # Calculate allocation percentages
        allocation_pct = {}
        for asset_class, value in allocation.items():
            allocation_pct[asset_class] = round(float(value / total_value * 100), 1) if total_value > 0 else 0

        # Get target allocation based on profile
        risk_profile = "balanced"
        if user.investor_profile:
            risk_profile = user.investor_profile.risk_profile.value

        target_allocation = ALLOCATION_BY_PROFILE.get(risk_profile, ALLOCATION_BY_PROFILE["balanced"])

        # Generate recommendations using LLM
        llm = LLMClient()

        prompt = f"""Analise a carteira de investimentos e gere recomendações personalizadas.

PERFIL DO INVESTIDOR:
- Perfil de risco: {risk_profile}
- Horizonte: {user.investor_profile.investment_horizon.value if user.investor_profile else 'medium_term'}

CARTEIRA ATUAL:
- Valor total: R$ {float(total_value):,.2f}
- Alocação atual: {json.dumps(allocation_pct, indent=2)}

ALOCAÇÃO ALVO PARA O PERFIL:
{json.dumps(target_allocation, indent=2)}

ATIVOS NA CARTEIRA:
{chr(10).join([f"- {a.ticker}: {float(a.quantity)} unidades @ R$ {float(a.average_price):.2f}" for a in user.portfolio.assets])}

Gere 3-5 recomendações específicas e acionáveis para melhorar a carteira. Considere:
1. Desvios da alocação alvo
2. Diversificação
3. Oportunidades de rebalanceamento
4. Sugestões de novos ativos se apropriado

Responda em formato JSON:
{{
    "recommendations": [
        {{
            "type": "rebalance|buy|sell|hold",
            "asset_class": "stocks|fiis|fixed_income|crypto|etf",
            "ticker": "XXXX3" (opcional),
            "action": "descrição da ação",
            "reason": "motivo da recomendação",
            "priority": "high|medium|low"
        }}
    ],
    "summary": "resumo geral da análise",
    "risk_assessment": "avaliação do risco atual da carteira"
}}"""

        try:
            response, tokens = await llm.chat(prompt)

            # Try to parse JSON from response
            try:
                # Find JSON in response
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    recommendations = json.loads(json_str)
                else:
                    recommendations = {"raw_response": response}
            except json.JSONDecodeError:
                recommendations = {"raw_response": response}

            return {
                "user_id": user_id,
                "portfolio_id": portfolio_id,
                "profile": risk_profile,
                "current_allocation": allocation_pct,
                "target_allocation": target_allocation,
                "total_value": float(total_value),
                "recommendations": recommendations,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "tokens_used": tokens,
            }

        except Exception as e:
            logger.error("Failed to generate LLM recommendations", error=str(e))
            return {
                "user_id": user_id,
                "portfolio_id": portfolio_id,
                "error": str(e),
            }


async def _analyze_asset_async(ticker: str) -> dict:
    """Perform comprehensive AI analysis on an asset."""
    from src.services.market import market_service

    try:
        # Get quote and fundamentals
        quote = await market_service.get_quote(ticker)
        fundamentals = await market_service.get_fundamentals(ticker)

        if not quote:
            return {"ticker": ticker, "error": "Asset not found"}

        # Build analysis prompt
        llm = LLMClient()

        fundamentals_str = ""
        if fundamentals:
            fundamentals_str = f"""
DADOS FUNDAMENTALISTAS:
- P/L: {fundamentals.get('pe_ratio', 'N/A')}
- P/VP: {fundamentals.get('pb_ratio', 'N/A')}
- Dividend Yield: {fundamentals.get('dividend_yield', 'N/A')}%
- ROE: {fundamentals.get('roe', 'N/A')}%
- Margem Líquida: {fundamentals.get('profit_margin', 'N/A')}%
- Setor: {fundamentals.get('sector', 'N/A')}
"""

        prompt = f"""Faça uma análise completa do ativo {ticker}.

DADOS DE MERCADO:
- Preço atual: R$ {float(quote.current_price):.2f}
- Variação diária: {float(quote.change_percent) if hasattr(quote, 'change_percent') else 'N/A'}%
- Nome: {quote.name}
{fundamentals_str}

Forneça uma análise abrangente incluindo:
1. Análise fundamentalista (se aplicável)
2. Pontos fortes e riscos
3. Comparação com setor
4. Perspectivas futuras
5. Recomendação (compra/venda/manter)

Responda em formato JSON:
{{
    "ticker": "{ticker}",
    "name": "nome completo",
    "sector": "setor",
    "analysis": {{
        "fundamentalist": "análise fundamentalista",
        "strengths": ["ponto forte 1", "ponto forte 2"],
        "risks": ["risco 1", "risco 2"],
        "outlook": "perspectiva"
    }},
    "recommendation": {{
        "action": "buy|sell|hold",
        "confidence": "high|medium|low",
        "target_price": numero ou null,
        "reasoning": "motivo"
    }},
    "summary": "resumo executivo em 2-3 frases"
}}"""

        response, tokens = await llm.chat(prompt)

        # Parse JSON
        try:
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                analysis = json.loads(json_str)
            else:
                analysis = {"raw_response": response}
        except json.JSONDecodeError:
            analysis = {"raw_response": response}

        return {
            "ticker": ticker,
            "current_price": float(quote.current_price),
            "analysis": analysis,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "tokens_used": tokens,
        }

    except Exception as e:
        logger.error("Failed to analyze asset", ticker=ticker, error=str(e))
        return {"ticker": ticker, "error": str(e)}


@celery_app.task
def process_user_profile(user_id: int, assessment_data: dict):
    """Process user assessment and generate investor profile."""
    logger.info("Processing user profile", user_id=user_id)

    try:
        # Calculate risk score
        risk_score, risk_profile = calculate_risk_score(assessment_data)

        # Get recommended allocation
        allocation = ALLOCATION_BY_PROFILE.get(risk_profile, ALLOCATION_BY_PROFILE["balanced"])

        result = {
            "user_id": user_id,
            "risk_score": risk_score,
            "risk_profile": risk_profile,
            "recommended_allocation": allocation,
            "status": "completed",
        }

        logger.info("User profile processed", **result)
        return result

    except Exception as e:
        logger.error("Failed to process user profile", user_id=user_id, error=str(e))
        return {"user_id": user_id, "status": "failed", "error": str(e)}


@celery_app.task
def generate_portfolio_recommendations(user_id: int, portfolio_id: int):
    """Generate AI-powered portfolio recommendations."""
    logger.info(
        "Generating portfolio recommendations",
        user_id=user_id,
        portfolio_id=portfolio_id,
    )

    result = run_async(_generate_recommendations_async(user_id, portfolio_id))
    logger.info("Portfolio recommendations generated", user_id=user_id)
    return result


@celery_app.task
def analyze_asset(ticker: str):
    """Perform comprehensive AI analysis on an asset."""
    logger.info("Analyzing asset", ticker=ticker)

    result = run_async(_analyze_asset_async(ticker))
    logger.info("Asset analysis completed", ticker=ticker)
    return result


@celery_app.task
def train_prediction_model(model_type: str, training_data_path: str):
    """Train or retrain ML prediction models."""
    logger.info("Training prediction model", model_type=model_type)
    # For MVP, this is a placeholder for future ML model training
    return {
        "model_type": model_type,
        "status": "not_implemented",
        "message": "ML model training will be implemented in a future release",
    }


@celery_app.task
def generate_chat_embeddings(message_id: str, content: str):
    """Generate embeddings for chat messages for semantic search."""
    logger.info("Generating chat embeddings", message_id=message_id)
    # For MVP, this is a placeholder for Qdrant integration
    return {
        "message_id": message_id,
        "status": "not_implemented",
        "message": "Embeddings will be implemented with Qdrant integration",
    }
