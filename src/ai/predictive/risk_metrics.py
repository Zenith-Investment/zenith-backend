"""Risk analysis module for portfolio metrics."""
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
import numpy as np
import pandas as pd
import statistics


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""
    volatility: float  # Annualized volatility
    var_95: float  # Value at Risk (95% confidence)
    var_99: float  # Value at Risk (99% confidence)
    max_drawdown: float  # Maximum drawdown percentage
    sharpe_ratio: float  # Risk-adjusted return
    sortino_ratio: float  # Downside risk-adjusted return
    beta: Optional[float]  # Market beta
    correlation_to_market: Optional[float]
    risk_score: int  # 1-10 (1=low risk, 10=high risk)
    risk_category: str  # "low", "moderate", "high", "very_high"

    def to_dict(self) -> dict:
        return {
            "volatility": round(self.volatility, 2),
            "var_95": round(self.var_95, 2),
            "var_99": round(self.var_99, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "beta": round(self.beta, 2) if self.beta else None,
            "correlation_to_market": round(self.correlation_to_market, 2) if self.correlation_to_market else None,
            "risk_score": self.risk_score,
            "risk_category": self.risk_category,
            "disclaimer": (
                "Métricas de risco são baseadas em dados históricos e não garantem "
                "comportamento futuro. Volatilidade e perdas podem ser maiores que as indicadas."
            ),
        }


@dataclass
class DiversificationAnalysis:
    """Portfolio diversification analysis."""
    concentration_index: float  # HHI (0-1, lower = more diversified)
    sector_exposure: dict[str, float]
    asset_class_exposure: dict[str, float]
    correlation_matrix: Optional[dict]
    diversification_score: int  # 1-10 (10=well diversified)
    recommendations: list[str]

    def to_dict(self) -> dict:
        return {
            "concentration_index": round(self.concentration_index, 4),
            "sector_exposure": self.sector_exposure,
            "asset_class_exposure": self.asset_class_exposure,
            "diversification_score": self.diversification_score,
            "recommendations": self.recommendations,
        }


class RiskAnalyzer:
    """Risk analysis calculator."""

    RISK_FREE_RATE = 0.1075  # Selic annual rate

    def calculate_volatility(self, returns: list[float]) -> float:
        """Calculate annualized volatility."""
        if len(returns) < 2:
            return 0.0
        daily_vol = statistics.stdev(returns)
        return daily_vol * (252 ** 0.5) * 100  # Annualized percentage

    def calculate_var(
        self,
        returns: list[float],
        confidence: float = 0.95,
        portfolio_value: float = 100000,
    ) -> float:
        """Calculate Value at Risk using historical simulation."""
        if len(returns) < 30:
            return 0.0

        percentile = (1 - confidence) * 100
        var_return = np.percentile(returns, percentile)
        return abs(var_return * portfolio_value)

    def calculate_max_drawdown(self, values: list[float]) -> float:
        """Calculate maximum drawdown percentage."""
        if not values:
            return 0.0

        peak = values[0]
        max_dd = 0

        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def calculate_sharpe_ratio(
        self,
        returns: list[float],
        risk_free_rate: Optional[float] = None,
    ) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 30:
            return 0.0

        rf = risk_free_rate or self.RISK_FREE_RATE

        mean_return = statistics.mean(returns) * 252  # Annualized
        volatility = self.calculate_volatility(returns) / 100

        if volatility == 0:
            return 0.0

        return (mean_return - rf) / volatility

    def calculate_sortino_ratio(
        self,
        returns: list[float],
        risk_free_rate: Optional[float] = None,
    ) -> float:
        """Calculate Sortino ratio (using downside deviation)."""
        if len(returns) < 30:
            return 0.0

        rf = risk_free_rate or self.RISK_FREE_RATE

        mean_return = statistics.mean(returns) * 252
        negative_returns = [r for r in returns if r < 0]

        if len(negative_returns) < 2:
            return 0.0

        downside_deviation = statistics.stdev(negative_returns) * (252 ** 0.5)

        if downside_deviation == 0:
            return 0.0

        return (mean_return - rf) / downside_deviation

    def calculate_beta(
        self,
        asset_returns: list[float],
        market_returns: list[float],
    ) -> float:
        """Calculate beta relative to market."""
        if len(asset_returns) != len(market_returns) or len(asset_returns) < 30:
            return 1.0

        covariance = np.cov(asset_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)

        if market_variance == 0:
            return 1.0

        return covariance / market_variance

    def analyze_portfolio_risk(
        self,
        daily_values: list[float],
        market_values: Optional[list[float]] = None,
    ) -> RiskMetrics:
        """
        Analyze portfolio risk metrics.

        Args:
            daily_values: List of daily portfolio values
            market_values: List of daily market index values (for beta calculation)
        """
        if len(daily_values) < 2:
            return RiskMetrics(
                volatility=0,
                var_95=0,
                var_99=0,
                max_drawdown=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                beta=None,
                correlation_to_market=None,
                risk_score=5,
                risk_category="moderate",
            )

        # Calculate returns
        returns = [
            (daily_values[i] - daily_values[i-1]) / daily_values[i-1]
            for i in range(1, len(daily_values))
        ]

        # Calculate metrics
        volatility = self.calculate_volatility(returns)
        var_95 = self.calculate_var(returns, 0.95, daily_values[-1])
        var_99 = self.calculate_var(returns, 0.99, daily_values[-1])
        max_dd = self.calculate_max_drawdown(daily_values)
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)

        # Calculate beta if market data available
        beta = None
        correlation = None
        if market_values and len(market_values) == len(daily_values):
            market_returns = [
                (market_values[i] - market_values[i-1]) / market_values[i-1]
                for i in range(1, len(market_values))
            ]
            beta = self.calculate_beta(returns, market_returns)
            correlation = np.corrcoef(returns, market_returns)[0][1]

        # Calculate risk score (1-10)
        risk_score = self._calculate_risk_score(volatility, max_dd, beta)

        # Determine risk category
        if risk_score <= 3:
            risk_category = "low"
        elif risk_score <= 5:
            risk_category = "moderate"
        elif risk_score <= 7:
            risk_category = "high"
        else:
            risk_category = "very_high"

        return RiskMetrics(
            volatility=volatility,
            var_95=var_95,
            var_99=var_99,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            beta=beta,
            correlation_to_market=correlation,
            risk_score=risk_score,
            risk_category=risk_category,
        )

    def _calculate_risk_score(
        self,
        volatility: float,
        max_drawdown: float,
        beta: Optional[float],
    ) -> int:
        """Calculate risk score from 1-10."""
        score = 0

        # Volatility contribution (0-4 points)
        if volatility < 10:
            score += 1
        elif volatility < 20:
            score += 2
        elif volatility < 30:
            score += 3
        else:
            score += 4

        # Max drawdown contribution (0-4 points)
        if max_drawdown < 10:
            score += 1
        elif max_drawdown < 20:
            score += 2
        elif max_drawdown < 30:
            score += 3
        else:
            score += 4

        # Beta contribution (0-2 points)
        if beta:
            if beta < 0.8:
                score += 0
            elif beta < 1.2:
                score += 1
            else:
                score += 2

        return min(10, max(1, score))

    def analyze_diversification(
        self,
        positions: list[dict],
    ) -> DiversificationAnalysis:
        """
        Analyze portfolio diversification.

        Args:
            positions: List of positions with ticker, value, sector, asset_class
        """
        if not positions:
            return DiversificationAnalysis(
                concentration_index=1.0,
                sector_exposure={},
                asset_class_exposure={},
                correlation_matrix=None,
                diversification_score=1,
                recommendations=["Adicione ativos à sua carteira"],
            )

        total_value = sum(p.get("value", 0) for p in positions)
        if total_value == 0:
            return DiversificationAnalysis(
                concentration_index=1.0,
                sector_exposure={},
                asset_class_exposure={},
                correlation_matrix=None,
                diversification_score=1,
                recommendations=["Sua carteira não possui valor"],
            )

        # Calculate HHI (Herfindahl-Hirschman Index)
        weights = [p.get("value", 0) / total_value for p in positions]
        hhi = sum(w ** 2 for w in weights)

        # Sector exposure
        sector_exposure = {}
        for p in positions:
            sector = p.get("sector", "Outros")
            value = p.get("value", 0)
            sector_exposure[sector] = sector_exposure.get(sector, 0) + value / total_value * 100

        # Asset class exposure
        asset_class_exposure = {}
        for p in positions:
            asset_class = p.get("asset_class", "Outros")
            value = p.get("value", 0)
            asset_class_exposure[asset_class] = asset_class_exposure.get(asset_class, 0) + value / total_value * 100

        # Calculate diversification score
        div_score = self._calculate_diversification_score(
            hhi, len(positions), len(sector_exposure), len(asset_class_exposure)
        )

        # Generate recommendations
        recommendations = self._generate_diversification_recommendations(
            hhi, positions, sector_exposure, asset_class_exposure
        )

        return DiversificationAnalysis(
            concentration_index=hhi,
            sector_exposure=sector_exposure,
            asset_class_exposure=asset_class_exposure,
            correlation_matrix=None,
            diversification_score=div_score,
            recommendations=recommendations,
        )

    def _calculate_diversification_score(
        self,
        hhi: float,
        num_positions: int,
        num_sectors: int,
        num_asset_classes: int,
    ) -> int:
        """Calculate diversification score from 1-10."""
        score = 0

        # HHI contribution (0-4 points, lower HHI = better)
        if hhi < 0.1:
            score += 4
        elif hhi < 0.2:
            score += 3
        elif hhi < 0.3:
            score += 2
        elif hhi < 0.5:
            score += 1

        # Number of positions (0-3 points)
        if num_positions >= 15:
            score += 3
        elif num_positions >= 10:
            score += 2
        elif num_positions >= 5:
            score += 1

        # Sector diversity (0-2 points)
        if num_sectors >= 5:
            score += 2
        elif num_sectors >= 3:
            score += 1

        # Asset class diversity (0-1 point)
        if num_asset_classes >= 3:
            score += 1

        return min(10, max(1, score))

    def _generate_diversification_recommendations(
        self,
        hhi: float,
        positions: list[dict],
        sector_exposure: dict,
        asset_class_exposure: dict,
    ) -> list[str]:
        """Generate diversification recommendations."""
        recommendations = []

        # Concentration
        if hhi > 0.3:
            largest = max(positions, key=lambda p: p.get("value", 0))
            recommendations.append(
                f"Sua carteira está muito concentrada. Considere reduzir a posição em {largest.get('ticker', 'ativos dominantes')}."
            )

        # Number of positions
        if len(positions) < 5:
            recommendations.append(
                "Considere adicionar mais ativos para melhorar a diversificação."
            )

        # Sector concentration
        for sector, exposure in sector_exposure.items():
            if exposure > 40:
                recommendations.append(
                    f"Alta exposição ao setor {sector} ({exposure:.1f}%). Considere diversificar em outros setores."
                )

        # Asset class diversity
        if len(asset_class_exposure) < 2:
            recommendations.append(
                "Considere diversificar em diferentes classes de ativos (ações, FIIs, renda fixa, etc.)."
            )

        # If no issues found
        if not recommendations:
            recommendations.append("Sua carteira apresenta boa diversificação!")

        return recommendations
