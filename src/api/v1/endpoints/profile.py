"""Investor profile endpoints."""
from fastapi import APIRouter, HTTPException, status
import structlog

from src.core.deps import CurrentUser, DbSession
from src.schemas.profile import (
    AssessmentStartResponse,
    AssessmentAnswerRequest,
    AssessmentResultResponse,
    InvestorProfileResponse,
    AssessmentQuestion,
)
from src.services.profile import ProfileService, ASSESSMENT_QUESTIONS

router = APIRouter()
logger = structlog.get_logger()


@router.post("/assessment/start", response_model=AssessmentStartResponse)
async def start_assessment(
    current_user: CurrentUser,
    db: DbSession,
) -> AssessmentStartResponse:
    """Start investor profile assessment questionnaire."""
    profile_service = ProfileService(db)

    result = await profile_service.start_assessment(current_user)
    logger.info("Assessment started", user_id=current_user.id, session_id=result["session_id"])

    return AssessmentStartResponse(
        session_id=result["session_id"],
        total_questions=result["total_questions"],
        questions=[
            AssessmentQuestion(
                id=q["id"],
                question=q["question"],
                options=q["options"],
                category=q["category"],
            )
            for q in result["questions"]
        ],
    )


@router.post("/assessment/submit", response_model=AssessmentResultResponse)
async def submit_assessment(
    request: AssessmentAnswerRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> AssessmentResultResponse:
    """Submit all assessment answers and get profile result."""
    profile_service = ProfileService(db)

    try:
        result = await profile_service.submit_answers(
            current_user,
            request.session_id,
            request.answers,
        )
        logger.info(
            "Assessment completed",
            user_id=current_user.id,
            risk_profile=result.risk_profile.value,
            risk_score=result.risk_score,
        )
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/", response_model=InvestorProfileResponse)
async def get_investor_profile(
    current_user: CurrentUser,
    db: DbSession,
) -> InvestorProfileResponse:
    """Get current user's investor profile."""
    profile_service = ProfileService(db)

    profile = await profile_service.get_investor_profile(current_user)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Investor profile not found. Please complete the assessment first.",
        )

    return InvestorProfileResponse(
        id=profile.id,
        user_id=profile.user_id,
        risk_profile=profile.risk_profile,
        risk_score=profile.risk_score,
        investment_horizon=profile.investment_horizon,
        primary_goals=profile.primary_goals,
        monthly_income=profile.monthly_income,
        monthly_investment=profile.monthly_investment,
        total_patrimony=profile.total_patrimony,
        experience_level=profile.experience_level,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


@router.get("/questions")
async def get_assessment_questions() -> dict:
    """Get all assessment questions (public endpoint for preview)."""
    return {
        "total_questions": len(ASSESSMENT_QUESTIONS),
        "questions": ASSESSMENT_QUESTIONS,
    }
