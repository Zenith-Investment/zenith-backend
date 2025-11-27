"""
Strategy recommendation engine that matches strategies to investor profiles.

This module analyzes investor profiles and historical backtest results to
suggest the most suitable strategies for each user.
"""
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Optional
import structlog

from src.ai.backtesting import (
    BacktestEngine,
    BuyAndHoldStrategy,
    RebalancingStrategy,
    MomentumStrategy,
    DCAStrategy,
)
from src.ai.predictive.risk_metrics import RiskAnalyzer

logger = structlog.get_logger()


@dataclass
class StrategyRecommendation:
    """A strategy recommendation with explanation."""
    strategy_name: str
    strategy_type: str
    suitability_score: float  # 0-100
    expected_return_range: tuple[float, float]
    risk_level: str  # "low", "moderate", "high"
    description: str
    pros: list[str]
    cons: list[str]
    backtest_results: Optional[dict] = None
    allocation_suggestion: Optional[dict[str, float]] = None

    def to_dict(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "strategy_type": self.strategy_type,
            "suitability_score": round(self.suitability_score, 1),
            "expected_return_range": {
                "min": round(self.expected_return_range[0], 2),
                "max": round(self.expected_return_range[1], 2),
            },
            "risk_level": self.risk_level,
            "description": self.description,
            "pros": self.pros,
            "cons": self.cons,
            "backtest_results": self.backtest_results,
            "allocation_suggestion": self.allocation_suggestion,
        }


@dataclass
class ProfileBasedRecommendations:
    """Complete recommendation set for an investor."""
    investor_profile: str
    risk_tolerance: str
    investment_horizon: str
    recommendations: list[StrategyRecommendation]
    primary_recommendation: StrategyRecommendation
    portfolio_allocation: dict[str, float]
    rebalance_frequency: str
    disclaimer: str = (
        "Estas recomendações são baseadas em análise de dados históricos e seu perfil de investidor. "
        "NÃO constituem recomendação de investimento. A decisão final é de sua responsabilidade. "
        "Consulte um profissional certificado (CEA/CNPI) antes de investir."
    )

    def to_dict(self) -> dict:
        return {
            "investor_profile": self.investor_profile,
            "risk_tolerance": self.risk_tolerance,
            "investment_horizon": self.investment_horizon,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "primary_recommendation": self.primary_recommendation.to_dict(),
            "portfolio_allocation": self.portfolio_allocation,
            "rebalance_frequency": self.rebalance_frequency,
            "disclaimer": self.disclaimer,
        }


# Strategy definitions with profile compatibility
STRATEGY_PROFILES = {
    "buy_and_hold": {
        "name": "Buy and Hold",
        "type": "passive",
        "risk_profiles": ["conservative", "moderate", "aggressive"],
        "min_horizon_years": 3,
        "risk_level": "moderate",
        "description": "Comprar e manter ativos a longo prazo, ignorando flutuações de curto prazo.",
        "pros": [
            "Baixo custo de transação",
            "Menos tempo dedicado ao mercado",
            "Benefício fiscal (menos imposto sobre ganhos)",
            "Historicamente eficaz no longo prazo",
        ],
        "cons": [
            "Requer paciência e disciplina",
            "Pode sofrer em mercados em queda prolongada",
            "Não aproveita oscilações de curto prazo",
        ],
    },
    "dca": {
        "name": "DCA (Custo Médio)",
        "type": "passive",
        "risk_profiles": ["conservative", "moderate"],
        "min_horizon_years": 2,
        "risk_level": "low",
        "description": "Investir valores fixos regularmente, independente do preço do mercado.",
        "pros": [
            "Reduz impacto da volatilidade",
            "Disciplina de investimento regular",
            "Não requer timing de mercado",
            "Ideal para quem recebe salário mensal",
        ],
        "cons": [
            "Pode ter retorno menor em mercados em alta constante",
            "Requer compromisso de longo prazo",
            "Custos de transação frequentes",
        ],
    },
    "rebalancing": {
        "name": "Rebalanceamento Periódico",
        "type": "passive",
        "risk_profiles": ["moderate", "aggressive"],
        "min_horizon_years": 1,
        "risk_level": "moderate",
        "description": "Manter alocação-alvo ajustando posições periodicamente.",
        "pros": [
            "Mantém nível de risco desejado",
            "Compra na baixa e vende na alta automaticamente",
            "Disciplina de investimento",
        ],
        "cons": [
            "Custos de transação",
            "Impostos sobre ganhos realizados",
            "Requer monitoramento regular",
        ],
    },
    "momentum": {
        "name": "Momentum",
        "type": "active",
        "risk_profiles": ["aggressive"],
        "min_horizon_years": 1,
        "risk_level": "high",
        "description": "Investir em ativos com tendência de alta recente.",
        "pros": [
            "Potencial de altos retornos em mercados de alta",
            "Aproveita tendências de mercado",
            "Pode superar o mercado em períodos específicos",
        ],
        "cons": [
            "Alto risco de perdas em reversões",
            "Requer acompanhamento frequente",
            "Custos de transação elevados",
            "Pode sofrer em mercados laterais",
        ],
    },
    "value": {
        "name": "Value Investing",
        "type": "active",
        "risk_profiles": ["moderate", "aggressive"],
        "min_horizon_years": 3,
        "risk_level": "moderate",
        "description": "Buscar ativos subvalorizados com base em análise fundamentalista.",
        "pros": [
            "Foco em valor intrínseco",
            "Menor risco de overpay",
            "Funciona bem no longo prazo",
        ],
        "cons": [
            "Requer conhecimento de análise fundamentalista",
            "Ativos podem ficar 'baratos' por muito tempo",
            "Pode perder rallies de crescimento",
        ],
    },
    "dividend": {
        "name": "Dividendos",
        "type": "passive",
        "risk_profiles": ["conservative", "moderate"],
        "min_horizon_years": 3,
        "risk_level": "low",
        "description": "Foco em ativos que pagam dividendos consistentes.",
        "pros": [
            "Renda passiva regular",
            "Empresas geralmente mais maduras e estáveis",
            "Reinvestimento de dividendos acelera crescimento",
        ],
        "cons": [
            "Crescimento de capital pode ser menor",
            "Dividendos não são garantidos",
            "Setores específicos podem dominar carteira",
        ],
    },
}


class StrategyRecommendationEngine:
    """Engine for recommending strategies based on investor profile."""

    def __init__(self):
        self.risk_analyzer = RiskAnalyzer()
        self.backtest_engine = BacktestEngine()

    def get_profile_characteristics(self, risk_profile: str) -> dict:
        """Get characteristics for each risk profile."""
        profiles = {
            "conservative": {
                "max_volatility": 15,
                "max_drawdown": 15,
                "target_sharpe": 0.5,
                "preferred_risk_level": ["low", "moderate"],
                "stock_allocation": (0.2, 0.4),
                "fixed_income_allocation": (0.5, 0.7),
            },
            "moderate": {
                "max_volatility": 25,
                "max_drawdown": 25,
                "target_sharpe": 0.7,
                "preferred_risk_level": ["moderate"],
                "stock_allocation": (0.4, 0.6),
                "fixed_income_allocation": (0.3, 0.5),
            },
            "aggressive": {
                "max_volatility": 40,
                "max_drawdown": 35,
                "target_sharpe": 1.0,
                "preferred_risk_level": ["moderate", "high"],
                "stock_allocation": (0.6, 0.9),
                "fixed_income_allocation": (0.1, 0.3),
            },
        }
        return profiles.get(risk_profile, profiles["moderate"])

    def calculate_suitability_score(
        self,
        strategy_key: str,
        risk_profile: str,
        investment_horizon_years: float,
        backtest_result: Optional[dict] = None,
    ) -> float:
        """Calculate how suitable a strategy is for the investor."""
        strategy = STRATEGY_PROFILES.get(strategy_key)
        if not strategy:
            return 0

        score = 50  # Base score

        # Profile compatibility
        if risk_profile in strategy["risk_profiles"]:
            score += 20
        elif risk_profile == "moderate":
            score += 10  # Moderate can adapt
        else:
            score -= 20

        # Horizon compatibility
        if investment_horizon_years >= strategy["min_horizon_years"]:
            score += 15
        else:
            # Penalize based on how short the horizon is
            horizon_ratio = investment_horizon_years / strategy["min_horizon_years"]
            score -= (1 - horizon_ratio) * 30

        # Backtest results
        if backtest_result:
            profile_chars = self.get_profile_characteristics(risk_profile)

            # Volatility check
            if backtest_result.get("volatility", 100) <= profile_chars["max_volatility"]:
                score += 10

            # Drawdown check
            if backtest_result.get("max_drawdown", 100) <= profile_chars["max_drawdown"]:
                score += 10

            # Sharpe ratio
            if backtest_result.get("sharpe_ratio", 0) >= profile_chars["target_sharpe"]:
                score += 15

        return max(0, min(100, score))

    async def get_recommendations(
        self,
        risk_profile: str,
        investment_horizon_years: float,
        goals: list[str],
        current_portfolio_value: Decimal = Decimal("10000"),
        historical_data: Optional[dict] = None,
    ) -> ProfileBasedRecommendations:
        """
        Generate personalized strategy recommendations.

        Args:
            risk_profile: "conservative", "moderate", or "aggressive"
            investment_horizon_years: Investment time horizon
            goals: List of investment goals
            current_portfolio_value: Current portfolio value for backtesting
            historical_data: Dict of ticker -> DataFrame for backtesting
        """
        recommendations = []

        # Evaluate each strategy
        for strategy_key, strategy_info in STRATEGY_PROFILES.items():
            # Run backtest if data available
            backtest_result = None
            if historical_data:
                backtest_result = await self._run_strategy_backtest(
                    strategy_key,
                    historical_data,
                    current_portfolio_value,
                )

            # Calculate suitability
            suitability = self.calculate_suitability_score(
                strategy_key,
                risk_profile,
                investment_horizon_years,
                backtest_result,
            )

            # Estimate return range based on risk level
            return_range = self._estimate_return_range(
                strategy_info["risk_level"],
                backtest_result,
            )

            # Create recommendation
            recommendation = StrategyRecommendation(
                strategy_name=strategy_info["name"],
                strategy_type=strategy_info["type"],
                suitability_score=suitability,
                expected_return_range=return_range,
                risk_level=strategy_info["risk_level"],
                description=strategy_info["description"],
                pros=strategy_info["pros"],
                cons=strategy_info["cons"],
                backtest_results=backtest_result,
                allocation_suggestion=self._get_allocation_for_strategy(
                    strategy_key, risk_profile
                ),
            )

            recommendations.append(recommendation)

        # Sort by suitability
        recommendations.sort(key=lambda r: r.suitability_score, reverse=True)

        # Get primary recommendation
        primary = recommendations[0] if recommendations else None

        # Get portfolio allocation
        portfolio_allocation = self._calculate_optimal_allocation(
            risk_profile,
            investment_horizon_years,
            goals,
        )

        # Determine rebalance frequency
        rebalance_freq = self._recommend_rebalance_frequency(
            risk_profile,
            investment_horizon_years,
        )

        return ProfileBasedRecommendations(
            investor_profile=risk_profile,
            risk_tolerance=self._risk_tolerance_description(risk_profile),
            investment_horizon=f"{investment_horizon_years} anos",
            recommendations=recommendations[:5],  # Top 5
            primary_recommendation=primary,
            portfolio_allocation=portfolio_allocation,
            rebalance_frequency=rebalance_freq,
        )

    async def _run_strategy_backtest(
        self,
        strategy_key: str,
        historical_data: dict,
        capital: Decimal,
    ) -> Optional[dict]:
        """Run backtest for a strategy."""
        try:
            tickers = list(historical_data.keys())

            # Get strategy instance
            if strategy_key == "buy_and_hold":
                strategy = BuyAndHoldStrategy(tickers)
            elif strategy_key == "dca":
                strategy = DCAStrategy(tickers)
            elif strategy_key == "rebalancing":
                allocation = {t: 1.0/len(tickers) for t in tickers}
                strategy = RebalancingStrategy(allocation)
            elif strategy_key == "momentum":
                strategy = MomentumStrategy(tickers)
            else:
                return None

            # Determine date range
            first_df = list(historical_data.values())[0]
            start_date = first_df.index[0].date()
            end_date = first_df.index[-1].date()

            # Run backtest
            engine = BacktestEngine(initial_capital=capital)
            result = await engine.run(
                strategy=strategy,
                price_data=historical_data,
                start_date=start_date,
                end_date=end_date,
            )

            return {
                "total_return": result.total_return,
                "annualized_return": result.annualized_return,
                "volatility": result.volatility,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "total_trades": result.total_trades,
            }

        except Exception as e:
            logger.error(f"Backtest failed for {strategy_key}", error=str(e))
            return None

    def _estimate_return_range(
        self,
        risk_level: str,
        backtest_result: Optional[dict],
    ) -> tuple[float, float]:
        """Estimate expected return range."""
        # Base ranges by risk level (conservative estimates)
        base_ranges = {
            "low": (4, 10),
            "moderate": (6, 15),
            "high": (8, 25),
        }

        base = base_ranges.get(risk_level, (5, 12))

        if backtest_result:
            # Adjust based on backtest
            annual_return = backtest_result.get("annualized_return", 0)
            volatility = backtest_result.get("volatility", 20)

            # Use backtest as anchor with volatility range
            low = max(base[0], annual_return - volatility)
            high = min(base[1] * 1.5, annual_return + volatility)

            return (low, high)

        return base

    def _get_allocation_for_strategy(
        self,
        strategy_key: str,
        risk_profile: str,
    ) -> dict[str, float]:
        """Get suggested allocation for a strategy."""
        allocations = {
            "conservative": {
                "buy_and_hold": {"acoes": 30, "fiis": 15, "renda_fixa": 50, "reserva": 5},
                "dca": {"acoes": 25, "fiis": 15, "renda_fixa": 55, "reserva": 5},
                "dividend": {"acoes_dividendos": 30, "fiis": 25, "renda_fixa": 40, "reserva": 5},
            },
            "moderate": {
                "buy_and_hold": {"acoes": 45, "fiis": 20, "renda_fixa": 30, "reserva": 5},
                "rebalancing": {"acoes": 50, "fiis": 20, "renda_fixa": 25, "reserva": 5},
                "value": {"acoes_valor": 50, "fiis": 15, "renda_fixa": 30, "reserva": 5},
            },
            "aggressive": {
                "buy_and_hold": {"acoes": 70, "fiis": 15, "renda_fixa": 10, "reserva": 5},
                "momentum": {"acoes_growth": 75, "fiis": 10, "renda_fixa": 10, "reserva": 5},
                "rebalancing": {"acoes": 65, "internacional": 15, "fiis": 15, "reserva": 5},
            },
        }

        profile_alloc = allocations.get(risk_profile, allocations["moderate"])
        return profile_alloc.get(strategy_key, {"acoes": 50, "renda_fixa": 45, "reserva": 5})

    def _calculate_optimal_allocation(
        self,
        risk_profile: str,
        horizon_years: float,
        goals: list[str],
    ) -> dict[str, float]:
        """Calculate optimal portfolio allocation."""
        base_allocations = {
            "conservative": {
                "renda_fixa": 55,
                "acoes": 20,
                "fiis": 15,
                "internacional": 5,
                "reserva": 5,
            },
            "moderate": {
                "renda_fixa": 35,
                "acoes": 35,
                "fiis": 15,
                "internacional": 10,
                "reserva": 5,
            },
            "aggressive": {
                "renda_fixa": 15,
                "acoes": 50,
                "fiis": 15,
                "internacional": 15,
                "reserva": 5,
            },
        }

        allocation = base_allocations.get(risk_profile, base_allocations["moderate"]).copy()

        # Adjust for horizon
        if horizon_years > 10:
            # Can take more risk with longer horizon
            allocation["acoes"] += 5
            allocation["renda_fixa"] -= 5
        elif horizon_years < 3:
            # Need more safety with shorter horizon
            allocation["acoes"] -= 10
            allocation["renda_fixa"] += 10

        # Adjust for goals
        if "renda_passiva" in goals:
            allocation["fiis"] += 5
            allocation["acoes"] -= 5
        if "aposentadoria" in goals and horizon_years > 15:
            allocation["internacional"] += 5
            allocation["renda_fixa"] -= 5

        return allocation

    def _recommend_rebalance_frequency(
        self,
        risk_profile: str,
        horizon_years: float,
    ) -> str:
        """Recommend rebalancing frequency."""
        if risk_profile == "conservative":
            return "anual"
        elif risk_profile == "moderate":
            return "semestral"
        else:
            if horizon_years < 3:
                return "trimestral"
            return "semestral"

    def _risk_tolerance_description(self, risk_profile: str) -> str:
        """Get description of risk tolerance."""
        descriptions = {
            "conservative": "Baixa tolerância a perdas. Prefere segurança a rentabilidade.",
            "moderate": "Aceita oscilações moderadas em busca de melhores retornos.",
            "aggressive": "Alta tolerância a risco. Busca maximizar retornos de longo prazo.",
        }
        return descriptions.get(risk_profile, descriptions["moderate"])


# Global instance
strategy_engine = StrategyRecommendationEngine()
