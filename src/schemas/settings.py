"""Schemas for user settings."""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class LLMProviderEnum(str, Enum):
    """Available LLM providers for user selection."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    TOGETHER = "together"
    AUTO = "auto"


class LLMModelInfo(BaseModel):
    """Information about an LLM model."""
    id: str
    name: str
    description: str


class LLMProviderInfo(BaseModel):
    """Information about an LLM provider."""
    enabled: bool
    model: Optional[str]
    type: str  # "local" or "api"
    cost: str  # "free", "low", "paid", "free_tier"
    user_key_configured: bool = False


class AvailableProvidersResponse(BaseModel):
    """Response with all available LLM providers."""
    active_provider: Optional[str]
    available_providers: dict[str, LLMProviderInfo]
    available_models: dict[str, list[LLMModelInfo]]


class UserSettingsResponse(BaseModel):
    """User settings response."""
    id: int
    user_id: int
    llm_provider: LLMProviderEnum
    llm_model: Optional[str]
    has_openai_key: bool
    has_anthropic_key: bool
    has_deepseek_key: bool
    has_groq_key: bool
    has_together_key: bool
    theme: str
    language: str
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class UpdateLLMSettingsRequest(BaseModel):
    """Request to update LLM settings."""
    llm_provider: Optional[LLMProviderEnum] = Field(
        None,
        description="Preferred LLM provider. Use 'auto' for system to choose automatically."
    )
    llm_model: Optional[str] = Field(
        None,
        description="Specific model to use (e.g., 'gpt-4o', 'claude-sonnet-4-5-20250929', 'qwen2.5:7b')"
    )


class UpdateAPIKeyRequest(BaseModel):
    """Request to update user's API key for a provider."""
    provider: LLMProviderEnum = Field(
        ...,
        description="The provider for this API key"
    )
    api_key: str = Field(
        ...,
        min_length=10,
        description="The API key (will be stored securely)"
    )


class DeleteAPIKeyRequest(BaseModel):
    """Request to delete a stored API key."""
    provider: LLMProviderEnum = Field(
        ...,
        description="The provider whose key should be deleted"
    )


class UpdateUISettingsRequest(BaseModel):
    """Request to update UI settings."""
    theme: Optional[str] = Field(
        None,
        pattern="^(light|dark|system)$",
        description="UI theme: light, dark, or system"
    )
    language: Optional[str] = Field(
        None,
        pattern="^[a-z]{2}(-[A-Z]{2})?$",
        description="Language code (e.g., 'pt-BR', 'en-US')"
    )


class TestLLMConnectionRequest(BaseModel):
    """Request to test LLM connection."""
    provider: LLMProviderEnum = Field(
        ...,
        description="Provider to test"
    )
    api_key: Optional[str] = Field(
        None,
        description="Optional API key to test (uses stored key if not provided)"
    )


class TestLLMConnectionResponse(BaseModel):
    """Response from LLM connection test."""
    success: bool
    provider: str
    model: str
    message: str
    response_time_ms: Optional[int] = None
