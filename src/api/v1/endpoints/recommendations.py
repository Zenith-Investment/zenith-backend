"""Recommendation endpoints for personalized investment suggestions."""
from fastapi import APIRouter, HTTPException, status
import structlog

from src.core.deps import CurrentUser, DbSession
from src.services.recommendation import RecommendationService, RecommendationsResponse

router = APIRouter()
logger = structlog.get_logger()


@router.get("/", response_model=RecommendationsResponse)
async def get_recommendations(
    current_user: CurrentUser,
    db: DbSession,
) -> RecommendationsResponse:
    """Get personalized investment recommendations based on user profile and portfolio.

    Returns:
        - List of specific asset recommendations (buy/sell/hold)
        - Allocation targets comparing current vs ideal allocation
        - Summary explaining the recommendations
    """
    recommendation_service = RecommendationService(db)

    try:
        recommendations = await recommendation_service.get_recommendations(current_user)

        logger.info(
            "Recommendations generated",
            user_id=current_user.id,
            recommendation_count=len(recommendations.recommendations),
        )

        return recommendations

    except Exception as e:
        logger.error(
            "Failed to generate recommendations",
            user_id=current_user.id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Falha ao gerar recomendações.",
        )
