from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.database import Base
from src.schemas.profile import InvestmentGoal, InvestmentHorizon, RiskProfile


class InvestorProfile(Base):
    __tablename__ = "investor_profiles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    risk_profile: Mapped[RiskProfile] = mapped_column(
        Enum(RiskProfile, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    risk_score: Mapped[int] = mapped_column(Integer, nullable=False)
    investment_horizon: Mapped[InvestmentHorizon] = mapped_column(
        Enum(InvestmentHorizon, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    primary_goals: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        nullable=False,
        default=[],
    )

    monthly_income: Mapped[Decimal | None] = mapped_column(Numeric(15, 2), nullable=True)
    monthly_investment: Mapped[Decimal | None] = mapped_column(Numeric(15, 2), nullable=True)
    total_patrimony: Mapped[Decimal | None] = mapped_column(Numeric(15, 2), nullable=True)
    experience_level: Mapped[str | None] = mapped_column(String(50), nullable=True)

    assessment_data: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        onupdate=func.now(),
        nullable=True,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="investor_profile")

    def __repr__(self) -> str:
        return f"<InvestorProfile(id={self.id}, user_id={self.user_id}, risk={self.risk_profile})>"


class AssessmentSession(Base):
    __tablename__ = "assessment_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(String(20), default="in_progress", nullable=False)
    answers: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    result: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
