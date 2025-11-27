"""Recommendation service for generating personalized investment suggestions."""
from decimal import Decimal
from enum import Enum
from typing import Literal

import structlog
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.portfolio import Portfolio, PortfolioAsset
from src.models.profile import InvestorProfile
from src.models.user import User
from src.schemas.portfolio import AssetClass
from src.schemas.profile import RiskProfile
from src.services.market import market_service

logger = structlog.get_logger()


class RecommendationAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class Recommendation(BaseModel):
    ticker: str
    name: str | None = None
    action: RecommendationAction
    reason: str
    confidence: float  # 0-1
    target_allocation: float | None = None  # Target percentage in portfolio
    current_price: Decimal | None = None
    asset_class: AssetClass


class AllocationTarget(BaseModel):
    asset_class: AssetClass
    target_percentage: float
    current_percentage: float
    difference: float
    action: Literal["increase", "decrease", "maintain"]


class RecommendationsResponse(BaseModel):
    recommendations: list[Recommendation]
    allocation_targets: list[AllocationTarget]
    summary: str


# Default allocation targets by risk profile
ALLOCATION_TARGETS = {
    RiskProfile.CONSERVATIVE: {
        AssetClass.FIXED_INCOME: 60.0,
        AssetClass.FIIS: 20.0,
        AssetClass.STOCKS: 15.0,
        AssetClass.CASH: 5.0,
    },
    RiskProfile.MODERATE: {
        AssetClass.STOCKS: 35.0,
        AssetClass.FIIS: 25.0,
        AssetClass.FIXED_INCOME: 30.0,
        AssetClass.ETF: 5.0,
        AssetClass.CASH: 5.0,
    },
    RiskProfile.BALANCED: {
        AssetClass.STOCKS: 45.0,
        AssetClass.FIIS: 20.0,
        AssetClass.FIXED_INCOME: 20.0,
        AssetClass.ETF: 10.0,
        AssetClass.CASH: 5.0,
    },
    RiskProfile.GROWTH: {
        AssetClass.STOCKS: 55.0,
        AssetClass.FIIS: 15.0,
        AssetClass.ETF: 15.0,
        AssetClass.FIXED_INCOME: 10.0,
        AssetClass.CASH: 5.0,
    },
    RiskProfile.AGGRESSIVE: {
        AssetClass.STOCKS: 70.0,
        AssetClass.ETF: 10.0,
        AssetClass.CRYPTO: 10.0,
        AssetClass.FIIS: 5.0,
        AssetClass.CASH: 5.0,
    },
}

# Popular assets by category for recommendations
RECOMMENDED_ASSETS = {
    AssetClass.STOCKS: [
        ("WEGE3", "Weg S.A.", "Empresa industrial sólida com forte crescimento"),
        ("ITUB4", "Itaú Unibanco", "Maior banco privado do Brasil, pagador de dividendos"),
        ("PETR4", "Petrobras", "Gigante do setor de petróleo, alto dividend yield"),
        ("VALE3", "Vale S.A.", "Líder global em mineração de ferro"),
        ("BBAS3", "Banco do Brasil", "Banco estatal com bons dividendos"),
        ("ABEV3", "Ambev", "Líder no setor de bebidas"),
        ("RENT3", "Localiza", "Líder em locação de veículos"),
        ("EGIE3", "Engie Brasil", "Empresa de energia elétrica estável"),
    ],
    AssetClass.FIIS: [
        ("HGLG11", "CSHG Logística", "FII de galpões logísticos com boa distribuição"),
        ("XPML11", "XP Malls", "FII de shopping centers diversificado"),
        ("BTLG11", "BTG Pactual Logística", "FII de logística com vacância baixa"),
        ("KNRI11", "Kinea Renda Imobiliária", "FII de lajes corporativas"),
        ("VISC11", "Vinci Shopping Centers", "FII de shoppings com gestão ativa"),
    ],
    AssetClass.ETF: [
        ("BOVA11", "iShares Ibovespa", "ETF que replica o Ibovespa"),
        ("IVVB11", "iShares S&P 500", "ETF que replica o S&P 500 em BRL"),
        ("SMAL11", "iShares Small Cap", "ETF de small caps brasileiras"),
    ],
    AssetClass.FIXED_INCOME: [
        ("TESOURO_SELIC", "Tesouro Selic", "Título público com liquidez diária"),
        ("TESOURO_IPCA", "Tesouro IPCA+", "Título público com proteção inflacionária"),
        ("CDB_DI", "CDB DI", "CDB com rendimento atrelado ao CDI"),
    ],
}


class RecommendationService:
    """Service for generating personalized investment recommendations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_recommendations(self, user: User) -> RecommendationsResponse:
        """Generate personalized recommendations based on user profile and portfolio."""
        # Load user profile
        profile = await self._get_user_profile(user)
        if not profile:
            return self._get_default_recommendations()

        # Load user portfolio
        portfolio = await self._get_user_portfolio(user)
        current_allocation = await self._calculate_current_allocation(portfolio)

        # Get target allocation based on profile
        target_allocation = ALLOCATION_TARGETS.get(
            profile.risk_profile,
            ALLOCATION_TARGETS[RiskProfile.MODERATE]
        )

        # Generate allocation targets
        allocation_targets = self._generate_allocation_targets(
            current_allocation, target_allocation
        )

        # Generate specific recommendations
        recommendations = await self._generate_recommendations(
            profile, portfolio, current_allocation, target_allocation
        )

        # Generate summary
        summary = self._generate_summary(profile, allocation_targets)

        return RecommendationsResponse(
            recommendations=recommendations,
            allocation_targets=allocation_targets,
            summary=summary,
        )

    async def _get_user_profile(self, user: User) -> InvestorProfile | None:
        """Get user's investor profile."""
        result = await self.db.execute(
            select(InvestorProfile).where(InvestorProfile.user_id == user.id)
        )
        return result.scalar_one_or_none()

    async def _get_user_portfolio(self, user: User) -> Portfolio | None:
        """Get user's portfolio with assets."""
        result = await self.db.execute(
            select(Portfolio)
            .where(Portfolio.user_id == user.id)
            .options(selectinload(Portfolio.assets))
        )
        return result.scalar_one_or_none()

    async def _calculate_current_allocation(
        self, portfolio: Portfolio | None
    ) -> dict[AssetClass, float]:
        """Calculate current portfolio allocation by asset class."""
        allocation = {}

        if not portfolio or not portfolio.assets:
            return allocation

        total_value = Decimal("0")
        class_values: dict[AssetClass, Decimal] = {}

        for asset in portfolio.assets:
            value = asset.quantity * asset.average_price
            total_value += value
            class_values[asset.asset_class] = class_values.get(
                asset.asset_class, Decimal("0")
            ) + value

        if total_value > 0:
            for asset_class, value in class_values.items():
                allocation[asset_class] = float((value / total_value) * 100)

        return allocation

    def _generate_allocation_targets(
        self,
        current: dict[AssetClass, float],
        target: dict[AssetClass, float],
    ) -> list[AllocationTarget]:
        """Generate allocation targets comparing current vs target."""
        targets = []

        all_classes = set(current.keys()) | set(target.keys())

        for asset_class in all_classes:
            current_pct = current.get(asset_class, 0.0)
            target_pct = target.get(asset_class, 0.0)
            diff = target_pct - current_pct

            if abs(diff) < 2:
                action = "maintain"
            elif diff > 0:
                action = "increase"
            else:
                action = "decrease"

            targets.append(
                AllocationTarget(
                    asset_class=asset_class,
                    target_percentage=target_pct,
                    current_percentage=current_pct,
                    difference=diff,
                    action=action,
                )
            )

        # Sort by absolute difference descending
        targets.sort(key=lambda x: abs(x.difference), reverse=True)

        return targets

    async def _generate_recommendations(
        self,
        profile: InvestorProfile,
        portfolio: Portfolio | None,
        current_allocation: dict[AssetClass, float],
        target_allocation: dict[AssetClass, float],
    ) -> list[Recommendation]:
        """Generate specific asset recommendations."""
        recommendations = []

        # Get current portfolio tickers
        current_tickers = set()
        if portfolio and portfolio.assets:
            current_tickers = {asset.ticker for asset in portfolio.assets}

        # Identify classes that need increase
        classes_to_increase = []
        for asset_class, target in target_allocation.items():
            current = current_allocation.get(asset_class, 0.0)
            if target - current >= 5:  # At least 5% below target
                classes_to_increase.append((asset_class, target - current))

        # Sort by biggest gap
        classes_to_increase.sort(key=lambda x: x[1], reverse=True)

        # Generate buy recommendations for underweight classes
        for asset_class, gap in classes_to_increase[:3]:  # Top 3 gaps
            assets = RECOMMENDED_ASSETS.get(asset_class, [])

            for ticker, name, reason in assets:
                if ticker not in current_tickers:
                    # Get current price if available
                    current_price = None
                    try:
                        quote = await market_service.get_quote(ticker)
                        if quote:
                            current_price = quote.current_price
                    except Exception:
                        pass

                    recommendations.append(
                        Recommendation(
                            ticker=ticker,
                            name=name,
                            action=RecommendationAction.BUY,
                            reason=f"{reason}. Alocação atual em {asset_class.value}: {current_allocation.get(asset_class, 0):.1f}%, meta: {target_allocation.get(asset_class, 0):.1f}%",
                            confidence=min(0.9, 0.5 + (gap / 100)),
                            target_allocation=target_allocation.get(asset_class),
                            current_price=current_price,
                            asset_class=asset_class,
                        )
                    )
                    break  # One recommendation per class

        # Generate hold recommendations for current holdings
        if portfolio and portfolio.assets:
            for asset in portfolio.assets[:3]:  # Top 3 holdings
                recommendations.append(
                    Recommendation(
                        ticker=asset.ticker,
                        name=asset.name,
                        action=RecommendationAction.HOLD,
                        reason="Ativo já presente na carteira. Continue acompanhando.",
                        confidence=0.7,
                        current_price=asset.average_price,
                        asset_class=asset.asset_class,
                    )
                )

        # Sort by confidence
        recommendations.sort(key=lambda x: x.confidence, reverse=True)

        return recommendations[:6]  # Return top 6 recommendations

    def _generate_summary(
        self,
        profile: InvestorProfile,
        allocation_targets: list[AllocationTarget],
    ) -> str:
        """Generate a text summary of recommendations."""
        profile_name = {
            RiskProfile.CONSERVATIVE: "Conservador",
            RiskProfile.MODERATE: "Moderado",
            RiskProfile.BALANCED: "Equilibrado",
            RiskProfile.GROWTH: "Crescimento",
            RiskProfile.AGGRESSIVE: "Agressivo",
        }.get(profile.risk_profile, "Moderado")

        # Find main adjustments needed
        increases = [t for t in allocation_targets if t.action == "increase"]
        decreases = [t for t in allocation_targets if t.action == "decrease"]

        summary_parts = [
            f"Com base no seu perfil {profile_name}, sua carteira ideal deveria ter maior exposição a ativos de "
        ]

        if increases:
            class_names = {
                AssetClass.STOCKS: "ações",
                AssetClass.FIIS: "fundos imobiliários",
                AssetClass.FIXED_INCOME: "renda fixa",
                AssetClass.ETF: "ETFs",
                AssetClass.CRYPTO: "criptomoedas",
                AssetClass.CASH: "reserva",
            }
            increase_names = [class_names.get(t.asset_class, t.asset_class.value) for t in increases[:2]]
            summary_parts.append(" e ".join(increase_names))
        else:
            summary_parts.append("diversificação geral")

        summary_parts.append(". ")

        if decreases:
            summary_parts.append(
                "Considere reduzir gradualmente posições que excedem a alocação ideal."
            )

        return "".join(summary_parts)

    def _get_default_recommendations(self) -> RecommendationsResponse:
        """Return default recommendations for users without profile."""
        return RecommendationsResponse(
            recommendations=[
                Recommendation(
                    ticker="BOVA11",
                    name="iShares Ibovespa",
                    action=RecommendationAction.BUY,
                    reason="ETF diversificado para começar a investir em renda variável",
                    confidence=0.8,
                    asset_class=AssetClass.ETF,
                ),
                Recommendation(
                    ticker="TESOURO_SELIC",
                    name="Tesouro Selic",
                    action=RecommendationAction.BUY,
                    reason="Investimento seguro para reserva de emergência",
                    confidence=0.9,
                    asset_class=AssetClass.FIXED_INCOME,
                ),
            ],
            allocation_targets=[],
            summary="Complete o questionário de perfil para receber recomendações personalizadas baseadas nos seus objetivos e tolerância a risco.",
        )
