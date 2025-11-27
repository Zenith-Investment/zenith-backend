from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    BALANCED = "balanced"
    GROWTH = "growth"
    AGGRESSIVE = "aggressive"


class InvestmentGoal(str, Enum):
    EMERGENCY_FUND = "emergency_fund"
    RETIREMENT = "retirement"
    WEALTH_BUILDING = "wealth_building"
    INCOME = "income"
    CAPITAL_PRESERVATION = "capital_preservation"
    EDUCATION = "education"
    REAL_ESTATE = "real_estate"
    OTHER = "other"


class InvestmentHorizon(str, Enum):
    SHORT_TERM = "short_term"  # < 2 years
    MEDIUM_TERM = "medium_term"  # 2-5 years
    LONG_TERM = "long_term"  # 5-10 years
    VERY_LONG_TERM = "very_long_term"  # > 10 years


class AssessmentQuestion(BaseModel):
    id: int
    question: str
    options: list[dict]
    category: str


class AssessmentStartResponse(BaseModel):
    session_id: str
    total_questions: int
    questions: list[AssessmentQuestion]


class AssessmentAnswerRequest(BaseModel):
    session_id: str
    answers: dict[int, int]  # question_id -> answer_id


class AllocationRecommendation(BaseModel):
    asset_class: str
    percentage: float = Field(..., ge=0, le=100)
    description: str


class AssessmentResultResponse(BaseModel):
    risk_profile: RiskProfile
    risk_score: int = Field(..., ge=0, le=100)
    investment_horizon: InvestmentHorizon
    primary_goals: list[InvestmentGoal]
    recommended_allocation: list[AllocationRecommendation]
    explanation: str


class InvestorProfileResponse(BaseModel):
    id: int
    user_id: int
    risk_profile: RiskProfile
    risk_score: int
    investment_horizon: InvestmentHorizon
    primary_goals: list[InvestmentGoal]
    monthly_income: float | None = None
    monthly_investment: float | None = None
    total_patrimony: float | None = None
    experience_level: str | None = None
    created_at: datetime
    updated_at: datetime | None = None

    class Config:
        from_attributes = True
