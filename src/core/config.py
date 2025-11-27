from functools import lru_cache
from typing import Literal

from pydantic import PostgresDsn, RedisDsn, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "InvestAI Platform"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Database
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "investai"
    POSTGRES_PASSWORD: str = "investai"
    POSTGRES_DB: str = "investai"

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @computed_field
    @property
    def DATABASE_URL_SYNC(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    REDIS_DB: int = 0

    @computed_field
    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ===========================================
    # AI/LLM Configuration - Multiple Providers
    # ===========================================
    # User can configure their preferred provider in settings
    # Priority: 1. User preference -> 2. Ollama -> 3. DeepSeek -> 4. Groq -> 5. Together -> 6. OpenAI -> 7. Anthropic

    # Default provider - Ollama is always the default (local, free, private)
    # User can override in settings if they have their own API keys
    DEFAULT_LLM_PROVIDER: str = "ollama"  # ollama, openai, anthropic, deepseek, groq, together

    # --- Open Source / Local ---

    # Ollama (local inference - FREE, private)
    # Best models for function calling in 2025:
    # - qwen2.5:14b or qwen2.5:32b (excellent multilingual + tool calling)
    # - llama3.3:70b (best quality, needs 48GB+ RAM)
    # - mistral:7b (fast, good for lower-end machines)
    OLLAMA_ENABLED: bool = True
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:14b"  # Best balance of quality/performance for tool calling

    # DeepSeek API (affordable - ~$0.14/1M tokens)
    DEEPSEEK_API_KEY: str | None = None
    DEEPSEEK_MODEL: str = "deepseek-chat"  # or deepseek-reasoner

    # Groq (fast inference - FREE tier available)
    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"  # or mixtral-8x7b-32768

    # Together AI (wide model selection)
    TOGETHER_API_KEY: str | None = None
    TOGETHER_MODEL: str = "Qwen/Qwen2.5-72B-Instruct-Turbo"

    # --- Proprietary APIs (user's own key) ---

    # OpenAI (GPT-4o, GPT-4-turbo)
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o"  # or gpt-4o-mini, gpt-4-turbo

    # Anthropic (Claude Sonnet 4.5)
    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_MODEL: str = "claude-sonnet-4-5-20250929"  # or claude-3-5-sonnet-20241022

    # Embedding model (local, no API required)
    DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Qdrant Vector DB
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str | None = None

    # Market Data APIs
    ALPHA_VANTAGE_API_KEY: str | None = None
    BRAPI_TOKEN: str | None = None

    # Email
    SMTP_HOST: str = "localhost"
    SMTP_PORT: int = 587
    SMTP_USER: str | None = None
    SMTP_PASSWORD: str | None = None
    SMTP_FROM_EMAIL: str = "noreply@investai.com.br"
    SMTP_FROM_NAME: str = "InvestAI"

    # API Base URL (for OAuth callbacks)
    API_BASE_URL: str = "http://localhost:8000"

    # Frontend URL (for password reset links)
    FRONTEND_URL: str = "http://localhost:3000"

    # Broker integrations
    BROKER_SANDBOX_MODE: bool = True

    # XP Investimentos (includes Rico and Clear)
    XP_CLIENT_ID: str | None = None
    XP_CLIENT_SECRET: str | None = None
    RICO_CLIENT_ID: str | None = None
    RICO_CLIENT_SECRET: str | None = None
    CLEAR_CLIENT_ID: str | None = None
    CLEAR_CLIENT_SECRET: str | None = None

    # BTG Pactual
    BTG_CLIENT_ID: str | None = None
    BTG_CLIENT_SECRET: str | None = None

    # NuInvest (Nubank)
    NUINVEST_CLIENT_ID: str | None = None
    NUINVEST_CLIENT_SECRET: str | None = None

    # Banco Inter
    INTER_CLIENT_ID: str | None = None
    INTER_CLIENT_SECRET: str | None = None

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Stripe Payment
    STRIPE_SECRET_KEY: str | None = None
    STRIPE_PUBLISHABLE_KEY: str | None = None
    STRIPE_WEBHOOK_SECRET: str | None = None

    # Subscription settings
    TRIAL_DAYS: int = 7


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
