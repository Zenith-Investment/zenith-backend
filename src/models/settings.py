"""User settings model for LLM and other preferences."""
from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Enum as SQLEnum, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.database import Base


class LLMProvider(str, Enum):
    """Available LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    TOGETHER = "together"
    AUTO = "auto"  # System auto-selects based on availability


class UserSettings(Base):
    """User settings and preferences."""
    __tablename__ = "user_settings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )

    # LLM Preferences
    llm_provider: Mapped[LLMProvider] = mapped_column(
        SQLEnum(LLMProvider, values_callable=lambda x: [e.value for e in x]),
        default=LLMProvider.AUTO,
        nullable=False,
    )
    llm_model: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # User's own API keys (encrypted in production)
    openai_api_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    anthropic_api_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    deepseek_api_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    groq_api_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    together_api_key: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # UI Preferences
    theme: Mapped[str] = mapped_column(String(20), default="system", nullable=False)
    language: Mapped[str] = mapped_column(String(10), default="pt-BR", nullable=False)

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

    # Relationship
    user: Mapped["User"] = relationship("User", back_populates="settings")

    def __repr__(self) -> str:
        return f"<UserSettings(user_id={self.user_id}, llm_provider={self.llm_provider})>"
