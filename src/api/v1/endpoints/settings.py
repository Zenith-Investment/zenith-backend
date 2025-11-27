"""User settings endpoints for LLM and UI preferences."""
import time
from fastapi import APIRouter, HTTPException, status
import structlog

from src.core.deps import CurrentUser, DbSession
from src.models.settings import UserSettings, LLMProvider
from src.schemas.settings import (
    LLMProviderEnum,
    LLMModelInfo,
    LLMProviderInfo,
    AvailableProvidersResponse,
    UserSettingsResponse,
    UpdateLLMSettingsRequest,
    UpdateAPIKeyRequest,
    DeleteAPIKeyRequest,
    UpdateUISettingsRequest,
    TestLLMConnectionRequest,
    TestLLMConnectionResponse,
)
from src.ai.llm.client import llm_client, LLMClient

router = APIRouter()
logger = structlog.get_logger()


async def get_or_create_settings(db, user_id: int) -> UserSettings:
    """Get or create user settings."""
    from sqlalchemy import select

    result = await db.execute(
        select(UserSettings).where(UserSettings.user_id == user_id)
    )
    settings = result.scalar_one_or_none()

    if not settings:
        settings = UserSettings(user_id=user_id)
        db.add(settings)
        await db.commit()
        await db.refresh(settings)

    return settings


@router.get("/", response_model=UserSettingsResponse)
async def get_user_settings(
    current_user: CurrentUser,
    db: DbSession,
) -> UserSettingsResponse:
    """Get current user's settings."""
    settings = await get_or_create_settings(db, current_user.id)

    return UserSettingsResponse(
        id=settings.id,
        user_id=settings.user_id,
        llm_provider=LLMProviderEnum(settings.llm_provider.value),
        llm_model=settings.llm_model,
        has_openai_key=bool(settings.openai_api_key),
        has_anthropic_key=bool(settings.anthropic_api_key),
        has_deepseek_key=bool(settings.deepseek_api_key),
        has_groq_key=bool(settings.groq_api_key),
        has_together_key=bool(settings.together_api_key),
        theme=settings.theme,
        language=settings.language,
        created_at=settings.created_at,
        updated_at=settings.updated_at,
    )


@router.get("/llm/providers", response_model=AvailableProvidersResponse)
async def get_available_providers(
    current_user: CurrentUser,
    db: DbSession,
) -> AvailableProvidersResponse:
    """Get all available LLM providers and models."""
    settings = await get_or_create_settings(db, current_user.id)

    # Get system-level provider info
    provider_info = llm_client.get_provider_info()

    # Add user's key configuration status
    providers = {}
    for name, info in provider_info["available_providers"].items():
        providers[name] = LLMProviderInfo(
            enabled=info["enabled"],
            model=info["model"],
            type=info["type"],
            cost=info["cost"],
            user_key_configured=_has_user_key(settings, name),
        )

    # Get available models
    models_raw = LLMClient.get_available_models()
    models = {
        provider: [LLMModelInfo(**m) for m in model_list]
        for provider, model_list in models_raw.items()
    }

    return AvailableProvidersResponse(
        active_provider=provider_info["active_provider"],
        available_providers=providers,
        available_models=models,
    )


def _has_user_key(settings: UserSettings, provider: str) -> bool:
    """Check if user has configured a key for the provider."""
    key_map = {
        "openai": settings.openai_api_key,
        "anthropic": settings.anthropic_api_key,
        "deepseek": settings.deepseek_api_key,
        "groq": settings.groq_api_key,
        "together": settings.together_api_key,
    }
    return bool(key_map.get(provider))


@router.put("/llm", response_model=UserSettingsResponse)
async def update_llm_settings(
    request: UpdateLLMSettingsRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> UserSettingsResponse:
    """Update user's LLM preferences."""
    settings = await get_or_create_settings(db, current_user.id)

    if request.llm_provider is not None:
        settings.llm_provider = LLMProvider(request.llm_provider.value)

    if request.llm_model is not None:
        settings.llm_model = request.llm_model

    await db.commit()
    await db.refresh(settings)

    logger.info(
        "LLM settings updated",
        user_id=current_user.id,
        provider=settings.llm_provider.value,
        model=settings.llm_model,
    )

    return UserSettingsResponse(
        id=settings.id,
        user_id=settings.user_id,
        llm_provider=LLMProviderEnum(settings.llm_provider.value),
        llm_model=settings.llm_model,
        has_openai_key=bool(settings.openai_api_key),
        has_anthropic_key=bool(settings.anthropic_api_key),
        has_deepseek_key=bool(settings.deepseek_api_key),
        has_groq_key=bool(settings.groq_api_key),
        has_together_key=bool(settings.together_api_key),
        theme=settings.theme,
        language=settings.language,
        created_at=settings.created_at,
        updated_at=settings.updated_at,
    )


@router.post("/llm/api-key")
async def save_api_key(
    request: UpdateAPIKeyRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Save user's API key for a provider."""
    if request.provider in (LLMProviderEnum.OLLAMA, LLMProviderEnum.AUTO):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider {request.provider} does not require an API key",
        )

    settings = await get_or_create_settings(db, current_user.id)

    # Map provider to setting field
    key_field_map = {
        LLMProviderEnum.OPENAI: "openai_api_key",
        LLMProviderEnum.ANTHROPIC: "anthropic_api_key",
        LLMProviderEnum.DEEPSEEK: "deepseek_api_key",
        LLMProviderEnum.GROQ: "groq_api_key",
        LLMProviderEnum.TOGETHER: "together_api_key",
    }

    field = key_field_map.get(request.provider)
    if field:
        setattr(settings, field, request.api_key)
        await db.commit()

        logger.info(
            "API key saved",
            user_id=current_user.id,
            provider=request.provider.value,
        )

    return {"message": f"API key for {request.provider.value} saved successfully"}


@router.delete("/llm/api-key")
async def delete_api_key(
    request: DeleteAPIKeyRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Delete user's stored API key for a provider."""
    settings = await get_or_create_settings(db, current_user.id)

    key_field_map = {
        LLMProviderEnum.OPENAI: "openai_api_key",
        LLMProviderEnum.ANTHROPIC: "anthropic_api_key",
        LLMProviderEnum.DEEPSEEK: "deepseek_api_key",
        LLMProviderEnum.GROQ: "groq_api_key",
        LLMProviderEnum.TOGETHER: "together_api_key",
    }

    field = key_field_map.get(request.provider)
    if field:
        setattr(settings, field, None)
        await db.commit()

        logger.info(
            "API key deleted",
            user_id=current_user.id,
            provider=request.provider.value,
        )

    return {"message": f"API key for {request.provider.value} deleted"}


@router.post("/llm/test", response_model=TestLLMConnectionResponse)
async def test_llm_connection(
    request: TestLLMConnectionRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> TestLLMConnectionResponse:
    """Test connection to an LLM provider."""
    from src.ai.llm.client import (
        OllamaClient,
        OpenAIClient,
        AnthropicClient,
        DeepSeekClient,
        GroqClient,
        TogetherClient,
    )
    from src.core.config import settings as app_settings

    settings = await get_or_create_settings(db, current_user.id)

    # Get API key (use provided or stored)
    api_key = request.api_key

    if not api_key and request.provider not in (LLMProviderEnum.OLLAMA, LLMProviderEnum.AUTO):
        key_map = {
            LLMProviderEnum.OPENAI: settings.openai_api_key,
            LLMProviderEnum.ANTHROPIC: settings.anthropic_api_key,
            LLMProviderEnum.DEEPSEEK: settings.deepseek_api_key,
            LLMProviderEnum.GROQ: settings.groq_api_key,
            LLMProviderEnum.TOGETHER: settings.together_api_key,
        }
        api_key = key_map.get(request.provider)

    start_time = time.time()
    model = ""

    try:
        if request.provider == LLMProviderEnum.OLLAMA:
            client = OllamaClient(
                base_url=app_settings.OLLAMA_BASE_URL,
                model=app_settings.OLLAMA_MODEL,
            )
            model = app_settings.OLLAMA_MODEL
            is_available = await client.is_available()
            if not is_available:
                return TestLLMConnectionResponse(
                    success=False,
                    provider=request.provider.value,
                    model=model,
                    message="Ollama is not running or model is not available",
                )
            # Quick test
            await client.chat([{"role": "user", "content": "Hi"}], max_tokens=10)

        elif request.provider == LLMProviderEnum.OPENAI:
            if not api_key:
                return TestLLMConnectionResponse(
                    success=False,
                    provider=request.provider.value,
                    model="",
                    message="OpenAI API key not configured",
                )
            client = OpenAIClient(api_key=api_key, model=app_settings.OPENAI_MODEL)
            model = app_settings.OPENAI_MODEL
            await client.chat([{"role": "user", "content": "Hi"}], max_tokens=10)

        elif request.provider == LLMProviderEnum.ANTHROPIC:
            if not api_key:
                return TestLLMConnectionResponse(
                    success=False,
                    provider=request.provider.value,
                    model="",
                    message="Anthropic API key not configured",
                )
            client = AnthropicClient(api_key=api_key, model=app_settings.ANTHROPIC_MODEL)
            model = app_settings.ANTHROPIC_MODEL
            await client.chat([{"role": "user", "content": "Hi"}], max_tokens=10)

        elif request.provider == LLMProviderEnum.DEEPSEEK:
            if not api_key:
                return TestLLMConnectionResponse(
                    success=False,
                    provider=request.provider.value,
                    model="",
                    message="DeepSeek API key not configured",
                )
            client = DeepSeekClient(api_key=api_key, model=app_settings.DEEPSEEK_MODEL)
            model = app_settings.DEEPSEEK_MODEL
            await client.chat([{"role": "user", "content": "Hi"}], max_tokens=10)

        elif request.provider == LLMProviderEnum.GROQ:
            if not api_key:
                return TestLLMConnectionResponse(
                    success=False,
                    provider=request.provider.value,
                    model="",
                    message="Groq API key not configured",
                )
            client = GroqClient(api_key=api_key, model=app_settings.GROQ_MODEL)
            model = app_settings.GROQ_MODEL
            await client.chat([{"role": "user", "content": "Hi"}], max_tokens=10)

        elif request.provider == LLMProviderEnum.TOGETHER:
            if not api_key:
                return TestLLMConnectionResponse(
                    success=False,
                    provider=request.provider.value,
                    model="",
                    message="Together API key not configured",
                )
            client = TogetherClient(api_key=api_key, model=app_settings.TOGETHER_MODEL)
            model = app_settings.TOGETHER_MODEL
            await client.chat([{"role": "user", "content": "Hi"}], max_tokens=10)

        else:
            return TestLLMConnectionResponse(
                success=False,
                provider=request.provider.value,
                model="",
                message="Invalid provider",
            )

        elapsed_ms = int((time.time() - start_time) * 1000)

        return TestLLMConnectionResponse(
            success=True,
            provider=request.provider.value,
            model=model,
            message="Connection successful",
            response_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.error(
            "LLM connection test failed",
            provider=request.provider.value,
            error=str(e),
        )
        return TestLLMConnectionResponse(
            success=False,
            provider=request.provider.value,
            model=model,
            message=f"Connection failed: {str(e)}",
        )


@router.put("/ui", response_model=UserSettingsResponse)
async def update_ui_settings(
    request: UpdateUISettingsRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> UserSettingsResponse:
    """Update user's UI preferences."""
    settings = await get_or_create_settings(db, current_user.id)

    if request.theme is not None:
        settings.theme = request.theme

    if request.language is not None:
        settings.language = request.language

    await db.commit()
    await db.refresh(settings)

    logger.info(
        "UI settings updated",
        user_id=current_user.id,
        theme=settings.theme,
        language=settings.language,
    )

    return UserSettingsResponse(
        id=settings.id,
        user_id=settings.user_id,
        llm_provider=LLMProviderEnum(settings.llm_provider.value),
        llm_model=settings.llm_model,
        has_openai_key=bool(settings.openai_api_key),
        has_anthropic_key=bool(settings.anthropic_api_key),
        has_deepseek_key=bool(settings.deepseek_api_key),
        has_groq_key=bool(settings.groq_api_key),
        has_together_key=bool(settings.together_api_key),
        theme=settings.theme,
        language=settings.language,
        created_at=settings.created_at,
        updated_at=settings.updated_at,
    )
