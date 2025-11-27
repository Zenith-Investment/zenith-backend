"""
Community Strategies Service.

Manages sharing and discovering successful investment strategies.
Uses ML to match strategies with user profiles.
"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional
import numpy as np
import structlog
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.community import CommunityStrategy, StrategyUse, StrategyMLFeatures, StrategyStatus
from src.models.analytics import Backtest, BacktestStatus
from src.models.profile import InvestorProfile
from src.models.user import User

logger = structlog.get_logger()


# Criteria for a successful strategy
STRATEGY_SUCCESS_CRITERIA = {
    "min_total_return": 5.0,  # Minimum 5% total return
    "min_sharpe_ratio": 0.5,  # Minimum Sharpe ratio
    "max_drawdown": -30.0,  # Maximum 30% drawdown
    "min_backtest_days": 180,  # Minimum 6 months backtest
}


@dataclass
class StrategyMatch:
    """Strategy match result for user."""
    strategy: CommunityStrategy
    match_score: float  # 0-100
    reasons: list[str]


class CommunityStrategiesService:
    """Service for community strategies."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def check_and_share_strategy(
        self,
        backtest: Backtest,
        user: User,
    ) -> Optional[CommunityStrategy]:
        """
        Check if a backtest qualifies as a successful strategy and offer sharing.

        Args:
            backtest: Completed backtest
            user: User who ran the backtest

        Returns:
            CommunityStrategy if created, None otherwise
        """
        # Check if meets success criteria
        if not self._meets_success_criteria(backtest):
            return None

        # Check if similar strategy already exists
        similar = await self._find_similar_strategy(backtest)
        if similar:
            logger.info(
                "Similar strategy already exists",
                backtest_id=backtest.id,
                existing_strategy_id=similar.id,
            )
            return None

        # Get user profile for targeting
        profile = await self._get_user_profile(user.id)
        if not profile:
            return None

        # Create community strategy
        strategy = CommunityStrategy(
            creator_id=user.id,
            strategy_name=f"{backtest.strategy_name} - Comunidade",
            strategy_type=backtest.strategy_name.lower().replace(" ", "_"),
            strategy_params=backtest.strategy_params,
            description=self._generate_strategy_description(backtest),
            applicable_tickers=backtest.tickers,
            target_risk_profile=profile.risk_profile,
            min_investment_horizon_years=Decimal(str(profile.investment_horizon or 1)),
            backtest_period_days=(backtest.end_date - backtest.start_date).days,
            total_return=backtest.total_return,
            annualized_return=backtest.annualized_return,
            sharpe_ratio=backtest.sharpe_ratio,
            max_drawdown=backtest.max_drawdown,
            volatility=backtest.volatility,
            win_rate=backtest.win_rate,
            status=StrategyStatus.PENDING,
        )

        self.db.add(strategy)
        await self.db.commit()
        await self.db.refresh(strategy)

        # Extract ML features for future recommendations
        await self._extract_ml_features(strategy, backtest, profile)

        logger.info(
            "Community strategy created",
            strategy_id=strategy.id,
            backtest_id=backtest.id,
        )

        return strategy

    def _meets_success_criteria(self, backtest: Backtest) -> bool:
        """Check if backtest meets success criteria."""
        if backtest.status != BacktestStatus.COMPLETED:
            return False

        if backtest.total_return is None:
            return False

        # Check criteria
        if float(backtest.total_return) < STRATEGY_SUCCESS_CRITERIA["min_total_return"]:
            return False

        if backtest.sharpe_ratio and float(backtest.sharpe_ratio) < STRATEGY_SUCCESS_CRITERIA["min_sharpe_ratio"]:
            return False

        if backtest.max_drawdown and float(backtest.max_drawdown) < STRATEGY_SUCCESS_CRITERIA["max_drawdown"]:
            return False

        period_days = (backtest.end_date - backtest.start_date).days
        if period_days < STRATEGY_SUCCESS_CRITERIA["min_backtest_days"]:
            return False

        return True

    async def _find_similar_strategy(self, backtest: Backtest) -> Optional[CommunityStrategy]:
        """Find similar existing strategy."""
        query = select(CommunityStrategy).where(
            and_(
                CommunityStrategy.strategy_type == backtest.strategy_name.lower().replace(" ", "_"),
                CommunityStrategy.status == StrategyStatus.VERIFIED,
                # Similar return range
                CommunityStrategy.annualized_return.between(
                    backtest.annualized_return * Decimal("0.8"),
                    backtest.annualized_return * Decimal("1.2"),
                ),
            )
        )

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def _get_user_profile(self, user_id: int) -> Optional[InvestorProfile]:
        """Get user's investor profile."""
        query = select(InvestorProfile).where(InvestorProfile.user_id == user_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    def _generate_strategy_description(self, backtest: Backtest) -> str:
        """Generate description for shared strategy."""
        return (
            f"Estratégia {backtest.strategy_name} testada por {(backtest.end_date - backtest.start_date).days} dias. "
            f"Retorno de {float(backtest.total_return):.2f}% com Sharpe Ratio de {float(backtest.sharpe_ratio or 0):.2f}. "
            f"Recomendada para investidores que buscam crescimento sustentável."
        )

    async def _extract_ml_features(
        self,
        strategy: CommunityStrategy,
        backtest: Backtest,
        profile: InvestorProfile,
    ) -> None:
        """Extract ML features from strategy for recommendation model."""
        # Market condition features
        market_features = {
            "backtest_period_days": (backtest.end_date - backtest.start_date).days,
            "num_tickers": len(backtest.tickers) if backtest.tickers else 0,
        }

        # Risk features
        risk_features = {
            "volatility": float(backtest.volatility) if backtest.volatility else 0,
            "max_drawdown": float(backtest.max_drawdown) if backtest.max_drawdown else 0,
            "sharpe_ratio": float(backtest.sharpe_ratio) if backtest.sharpe_ratio else 0,
        }

        # Return features
        return_features = {
            "total_return": float(backtest.total_return) if backtest.total_return else 0,
            "annualized_return": float(backtest.annualized_return) if backtest.annualized_return else 0,
            "win_rate": float(backtest.win_rate) if backtest.win_rate else 0,
        }

        # User profile features
        profile_features = {
            "risk_profile": profile.risk_profile,
            "investment_horizon": float(profile.investment_horizon) if profile.investment_horizon else 1,
            "experience_level": profile.experience_level,
        }

        ml_features = StrategyMLFeatures(
            strategy_id=strategy.id,
            market_condition_features=market_features,
            risk_features=risk_features,
            return_features=return_features,
            user_profile_features=profile_features,
            model_version="v1.0",
        )

        self.db.add(ml_features)
        await self.db.commit()

    async def get_recommended_strategies(
        self,
        user: User,
        limit: int = 5,
    ) -> list[StrategyMatch]:
        """
        Get recommended strategies for a user based on their profile.

        Uses ML matching to find the best strategies.

        Args:
            user: User to get recommendations for
            limit: Maximum number of recommendations

        Returns:
            List of StrategyMatch with scores and reasons
        """
        profile = await self._get_user_profile(user.id)
        if not profile:
            return []

        # Get verified strategies matching user's risk profile
        query = select(CommunityStrategy).where(
            and_(
                CommunityStrategy.status == StrategyStatus.VERIFIED,
                CommunityStrategy.is_public == True,
                or_(
                    CommunityStrategy.target_risk_profile == profile.risk_profile,
                    # Conservative can also use moderate strategies
                    and_(
                        profile.risk_profile == "conservative",
                        CommunityStrategy.target_risk_profile.in_(["conservative", "moderate"]),
                    ),
                    # Aggressive can use all
                    and_(
                        profile.risk_profile == "aggressive",
                        CommunityStrategy.target_risk_profile.in_(["conservative", "moderate", "aggressive"]),
                    ),
                ),
            )
        ).order_by(
            CommunityStrategy.community_rating.desc().nulls_last(),
            CommunityStrategy.times_used.desc(),
        ).limit(limit * 2)  # Get more for scoring

        result = await self.db.execute(query)
        strategies = result.scalars().all()

        # Score and rank strategies
        matches = []
        for strategy in strategies:
            score, reasons = await self._calculate_match_score(strategy, profile)
            matches.append(StrategyMatch(
                strategy=strategy,
                match_score=score,
                reasons=reasons,
            ))

        # Sort by score and return top matches
        matches.sort(key=lambda x: x.match_score, reverse=True)
        return matches[:limit]

    async def _calculate_match_score(
        self,
        strategy: CommunityStrategy,
        profile: InvestorProfile,
    ) -> tuple[float, list[str]]:
        """Calculate match score between strategy and user profile."""
        score = 50.0  # Base score
        reasons = []

        # Risk profile match (30 points)
        if strategy.target_risk_profile == profile.risk_profile:
            score += 30
            reasons.append(f"Perfil de risco compatível ({profile.risk_profile})")
        elif profile.risk_profile == "aggressive":
            score += 15
            reasons.append("Estratégia conservadora pode diversificar seu portfólio agressivo")

        # Investment horizon match (20 points)
        user_horizon = float(profile.investment_horizon) if profile.investment_horizon else 1
        strategy_min_horizon = float(strategy.min_investment_horizon_years)

        if user_horizon >= strategy_min_horizon:
            score += 20
            reasons.append("Horizonte de investimento compatível")
        else:
            score -= 10

        # Performance metrics (20 points)
        if strategy.annualized_return and float(strategy.annualized_return) > 10:
            score += 10
            reasons.append(f"Retorno anualizado de {float(strategy.annualized_return):.1f}%")

        if strategy.sharpe_ratio and float(strategy.sharpe_ratio) > 1:
            score += 10
            reasons.append(f"Sharpe Ratio de {float(strategy.sharpe_ratio):.2f}")

        # Community success (10 points)
        if strategy.times_used > 0:
            success_rate = strategy.success_count / strategy.times_used if strategy.times_used else 0
            if success_rate > 0.7:
                score += 10
                reasons.append(f"Taxa de sucesso de {success_rate:.0%} na comunidade")
            elif success_rate > 0.5:
                score += 5

        # Rating (10 points)
        if strategy.community_rating:
            rating_bonus = float(strategy.community_rating) / 5 * 10
            score += rating_bonus
            reasons.append(f"Avaliação de {float(strategy.community_rating):.1f}/5 estrelas")

        # Cap score at 100
        score = min(100, max(0, score))

        return score, reasons

    async def use_strategy(
        self,
        strategy_id: int,
        user: User,
        initial_value: Decimal,
    ) -> StrategyUse:
        """
        Record that a user is using a community strategy.

        Args:
            strategy_id: ID of the strategy being used
            user: User applying the strategy
            initial_value: Initial portfolio/investment value

        Returns:
            StrategyUse record
        """
        # Get strategy
        query = select(CommunityStrategy).where(CommunityStrategy.id == strategy_id)
        result = await self.db.execute(query)
        strategy = result.scalar_one_or_none()

        if not strategy:
            raise ValueError("Estratégia não encontrada")

        # Create use record
        use = StrategyUse(
            strategy_id=strategy_id,
            user_id=user.id,
            applied_at=datetime.utcnow(),
            initial_value=initial_value,
        )

        self.db.add(use)

        # Update strategy metrics
        strategy.times_used += 1

        await self.db.commit()
        await self.db.refresh(use)

        logger.info(
            "User started using strategy",
            user_id=user.id,
            strategy_id=strategy_id,
        )

        return use

    async def update_strategy_outcome(
        self,
        use_id: int,
        current_value: Decimal,
        is_active: bool = True,
        rating: Optional[int] = None,
        feedback: Optional[str] = None,
    ) -> StrategyUse:
        """Update outcome of a strategy use."""
        query = select(StrategyUse).where(StrategyUse.id == use_id)
        result = await self.db.execute(query)
        use = result.scalar_one_or_none()

        if not use:
            raise ValueError("Registro de uso não encontrado")

        # Calculate return
        return_pct = (current_value - use.initial_value) / use.initial_value * 100

        use.current_value = current_value
        use.return_pct = return_pct
        use.is_active = is_active

        if rating:
            use.user_rating = rating

        if feedback:
            use.user_feedback = feedback

        if not is_active:
            use.stopped_at = datetime.utcnow()

            # Update strategy success metrics
            strategy = use.strategy
            if float(return_pct) > 0:
                strategy.success_count += 1
            else:
                strategy.failure_count += 1

            # Update average return
            uses_query = select(func.avg(StrategyUse.return_pct)).where(
                and_(
                    StrategyUse.strategy_id == strategy.id,
                    StrategyUse.return_pct.isnot(None),
                )
            )
            avg_result = await self.db.execute(uses_query)
            avg_return = avg_result.scalar()
            if avg_return:
                strategy.avg_user_return = Decimal(str(avg_return))

            # Update community rating
            rating_query = select(func.avg(StrategyUse.user_rating)).where(
                and_(
                    StrategyUse.strategy_id == strategy.id,
                    StrategyUse.user_rating.isnot(None),
                )
            )
            rating_result = await self.db.execute(rating_query)
            avg_rating = rating_result.scalar()
            if avg_rating:
                strategy.community_rating = Decimal(str(avg_rating))

        await self.db.commit()
        await self.db.refresh(use)

        return use

    async def get_featured_strategies(self, limit: int = 10) -> list[CommunityStrategy]:
        """Get featured community strategies."""
        query = (
            select(CommunityStrategy)
            .where(
                and_(
                    CommunityStrategy.status == StrategyStatus.VERIFIED,
                    CommunityStrategy.is_public == True,
                    CommunityStrategy.is_featured == True,
                )
            )
            .order_by(CommunityStrategy.community_rating.desc().nulls_last())
            .limit(limit)
        )

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_top_strategies(
        self,
        sort_by: str = "rating",  # rating, return, success_rate
        limit: int = 10,
    ) -> list[CommunityStrategy]:
        """Get top performing community strategies."""
        query = select(CommunityStrategy).where(
            and_(
                CommunityStrategy.status == StrategyStatus.VERIFIED,
                CommunityStrategy.is_public == True,
            )
        )

        if sort_by == "rating":
            query = query.order_by(CommunityStrategy.community_rating.desc().nulls_last())
        elif sort_by == "return":
            query = query.order_by(CommunityStrategy.annualized_return.desc())
        elif sort_by == "success_rate":
            # Calculate success rate
            query = query.order_by(
                (CommunityStrategy.success_count / (CommunityStrategy.times_used + 1)).desc()
            )
        elif sort_by == "sharpe_ratio":
            query = query.order_by(CommunityStrategy.sharpe_ratio.desc())
        else:
            query = query.order_by(CommunityStrategy.times_used.desc())

        query = query.limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def verify_strategy(self, strategy_id: int) -> CommunityStrategy:
        """Verify a community strategy (admin only)."""
        query = select(CommunityStrategy).where(CommunityStrategy.id == strategy_id)
        result = await self.db.execute(query)
        strategy = result.scalar_one_or_none()

        if not strategy:
            raise ValueError("Estratégia não encontrada")

        strategy.status = StrategyStatus.VERIFIED
        strategy.verified_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(strategy)

        return strategy


def get_community_service(db: AsyncSession) -> CommunityStrategiesService:
    return CommunityStrategiesService(db)
