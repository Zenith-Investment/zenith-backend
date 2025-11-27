"""
LLM client with support for multiple providers.

Supported providers (configurable by user):
1. Ollama (local, free, private) - DeepSeek, Qwen, Llama, Mistral
2. OpenAI (GPT-4o, GPT-4-turbo) - User's API key
3. Anthropic (Claude Sonnet 4.5) - User's API key
4. DeepSeek API (affordable, powerful reasoning)
5. Groq (fast inference for open-source models)
6. Together AI (wide model selection)
"""
import json
import httpx
import structlog
from typing import Optional, AsyncGenerator
from enum import Enum

from src.core.config import settings

logger = structlog.get_logger()


class LLMProvider(str, Enum):
    """Available LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    TOGETHER = "together"


# System prompt for the investment assistant
INVESTMENT_ASSISTANT_PROMPT = """Voce e o InvestAI, um assistente de investimentos especializado no mercado brasileiro.

Suas capacidades incluem:
- Analise de acoes, FIIs, ETFs e outros ativos da B3
- Explicacao de conceitos financeiros de forma clara
- Sugestoes personalizadas baseadas no perfil do investidor
- Analise fundamentalista e tecnica basica
- Orientacao sobre diversificacao de carteira
- Informacoes sobre tributacao de investimentos no Brasil

Diretrizes importantes:
1. Sempre mencione que voce nao e um consultor financeiro certificado
2. Recomende que o usuario consulte um profissional para decisoes importantes
3. Use dados e fatos quando disponiveis
4. Seja claro sobre riscos envolvidos em investimentos
5. Adapte a linguagem ao nivel de conhecimento do usuario
6. Responda em portugues brasileiro
7. Seja conciso mas completo nas respostas

Informacoes do usuario serao fornecidas no contexto quando disponiveis."""


class OllamaClient:
    """Ollama client for local LLM inference with tool calling support."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:14b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.supports_tools = True  # Qwen2.5, Llama3.3, Mistral support tools

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: list[dict] | None = None,
    ) -> str | dict:
        """Send chat request to Ollama with optional tool calling."""
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        # Add tools if provided and model supports them
        if tools and self.supports_tools:
            payload["tools"] = tools

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=120.0)
            response.raise_for_status()
            data = response.json()

            message = data.get("message", {})

            # Check if model wants to use a tool
            if message.get("tool_calls"):
                return {
                    "type": "tool_calls",
                    "tool_calls": message["tool_calls"],
                    "content": message.get("content", ""),
                }

            return message.get("content", "")

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Stream chat response from Ollama."""
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        # Use longer timeout for CPU inference - first token can take a while
        # but once streaming starts, it flows continuously
        timeout = httpx.Timeout(300.0, connect=30.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

    async def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    return any(self.model.split(":")[0] in m for m in models)
        except Exception:
            pass
        return False


class OpenAIClient:
    """OpenAI API client - GPT-4o, GPT-4-turbo, etc."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send chat request to OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Stream chat response from OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60.0,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            content = data["choices"][0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue


class AnthropicClient:
    """Anthropic API client - Claude Sonnet 4.5, etc."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send chat request to Anthropic API."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        # Anthropic uses a different message format - extract system message
        system_content = ""
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                api_messages.append(msg)

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }

        if system_content:
            payload["system"] = system_content

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            # Anthropic returns content as a list
            content = data.get("content", [])
            if content and isinstance(content, list):
                return content[0].get("text", "")
            return ""

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Stream chat response from Anthropic API."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        system_content = ""
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                api_messages.append(msg)

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": api_messages,
            "stream": True,
        }

        if system_content:
            payload["system"] = system_content

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60.0,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        try:
                            data = json.loads(data_str)
                            if data.get("type") == "content_block_delta":
                                content = data.get("delta", {}).get("text", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue


class DeepSeekClient:
    """DeepSeek API client - powerful and affordable."""

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.deepseek.com/v1/chat/completions"

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send chat request to DeepSeek API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Stream chat response from DeepSeek API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60.0,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            content = data["choices"][0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue


class GroqClient:
    """Groq API client - fast inference for open-source models."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send chat request to Groq API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Stream chat response from Groq API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60.0,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            content = data["choices"][0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue


class TogetherClient:
    """Together AI client - wide selection of open-source models."""

    def __init__(self, api_key: str, model: str = "Qwen/Qwen2.5-72B-Instruct-Turbo"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.together.xyz/v1/chat/completions"

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send chat request to Together API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Stream chat response from Together API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60.0,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            content = data["choices"][0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue


class LLMClient:
    """
    LLM client with support for multiple providers.

    Priority order (configurable):
    1. User's preferred provider (if configured)
    2. Ollama (local) - Free, private, no API costs
    3. DeepSeek API - Affordable, excellent reasoning
    4. Groq - Fast inference, free tier available
    5. Together AI - Wide model selection
    6. OpenAI - GPT-4o, GPT-4-turbo
    7. Anthropic - Claude Sonnet 4.5
    """

    def __init__(self):
        self._clients: dict = {}
        self._active_provider: Optional[str] = None

    @property
    def model(self) -> str:
        """Get the model name for the active provider."""
        if self._active_provider and self._active_provider in self._clients:
            client = self._clients[self._active_provider]
            return getattr(client, "model", "unknown")
        # Return default based on provider settings
        model_map = {
            "ollama": settings.OLLAMA_MODEL,
            "openai": settings.OPENAI_MODEL,
            "anthropic": settings.ANTHROPIC_MODEL,
            "deepseek": settings.DEEPSEEK_MODEL,
            "groq": settings.GROQ_MODEL,
            "together": settings.TOGETHER_MODEL,
        }
        if self._active_provider:
            return model_map.get(self._active_provider, "unknown")
        return "unknown"

    def _get_ollama_client(self) -> Optional[OllamaClient]:
        """Get Ollama client if enabled."""
        if settings.OLLAMA_ENABLED:
            if "ollama" not in self._clients:
                self._clients["ollama"] = OllamaClient(
                    base_url=settings.OLLAMA_BASE_URL,
                    model=settings.OLLAMA_MODEL,
                )
            return self._clients["ollama"]
        return None

    def _get_openai_client(self) -> Optional[OpenAIClient]:
        """Get OpenAI client if API key is configured."""
        if settings.OPENAI_API_KEY:
            if "openai" not in self._clients:
                self._clients["openai"] = OpenAIClient(
                    api_key=settings.OPENAI_API_KEY,
                    model=settings.OPENAI_MODEL,
                )
            return self._clients["openai"]
        return None

    def _get_anthropic_client(self) -> Optional[AnthropicClient]:
        """Get Anthropic client if API key is configured."""
        if settings.ANTHROPIC_API_KEY:
            if "anthropic" not in self._clients:
                self._clients["anthropic"] = AnthropicClient(
                    api_key=settings.ANTHROPIC_API_KEY,
                    model=settings.ANTHROPIC_MODEL,
                )
            return self._clients["anthropic"]
        return None

    def _get_deepseek_client(self) -> Optional[DeepSeekClient]:
        """Get DeepSeek client if API key is configured."""
        if settings.DEEPSEEK_API_KEY:
            if "deepseek" not in self._clients:
                self._clients["deepseek"] = DeepSeekClient(
                    api_key=settings.DEEPSEEK_API_KEY,
                    model=settings.DEEPSEEK_MODEL,
                )
            return self._clients["deepseek"]
        return None

    def _get_groq_client(self) -> Optional[GroqClient]:
        """Get Groq client if API key is configured."""
        if settings.GROQ_API_KEY:
            if "groq" not in self._clients:
                self._clients["groq"] = GroqClient(
                    api_key=settings.GROQ_API_KEY,
                    model=settings.GROQ_MODEL,
                )
            return self._clients["groq"]
        return None

    def _get_together_client(self) -> Optional[TogetherClient]:
        """Get Together AI client if API key is configured."""
        if settings.TOGETHER_API_KEY:
            if "together" not in self._clients:
                self._clients["together"] = TogetherClient(
                    api_key=settings.TOGETHER_API_KEY,
                    model=settings.TOGETHER_MODEL,
                )
            return self._clients["together"]
        return None

    async def _get_client(self, preferred_provider: Optional[str] = None):
        """
        Get the best available LLM client.

        Priority:
        1. Ollama (default - local, free, private) - ALWAYS tried first
        2. User's preferred provider (if they configured API keys)
        3. Fallback to any available provider with API key

        Args:
            preferred_provider: User's preferred provider (from settings)
        """
        # ===========================================
        # PRIORITY 1: Always try Ollama first (default)
        # ===========================================
        ollama = self._get_ollama_client()
        if ollama and await ollama.is_available():
            self._active_provider = "ollama"
            logger.info("Using Ollama (local, default)", model=settings.OLLAMA_MODEL)
            return ollama

        # ===========================================
        # PRIORITY 2: User's preferred provider (if configured)
        # ===========================================
        # Only use other providers if user explicitly configured API key
        if preferred_provider is None:
            preferred_provider = getattr(settings, "DEFAULT_LLM_PROVIDER", "ollama")

        # Skip ollama in preferred (already tried above)
        if preferred_provider and preferred_provider != "ollama":
            provider_map = {
                "openai": self._get_openai_client,
                "anthropic": self._get_anthropic_client,
                "deepseek": self._get_deepseek_client,
                "groq": self._get_groq_client,
                "together": self._get_together_client,
            }

            if preferred_provider in provider_map:
                client = provider_map[preferred_provider]()
                if client:
                    self._active_provider = preferred_provider
                    logger.info(
                        f"Using {preferred_provider} (user preference)",
                        model=getattr(settings, f"{preferred_provider.upper()}_MODEL", "unknown")
                    )
                    return client

        # ===========================================
        # PRIORITY 3: Fallback to any available provider
        # ===========================================
        # Only if Ollama is not available AND user has API keys configured

        # DeepSeek (affordable)
        deepseek = self._get_deepseek_client()
        if deepseek:
            self._active_provider = "deepseek"
            logger.info("Using DeepSeek API (fallback)", model=settings.DEEPSEEK_MODEL)
            return deepseek

        # Groq (fast, free tier)
        groq = self._get_groq_client()
        if groq:
            self._active_provider = "groq"
            logger.info("Using Groq API (fallback)", model=settings.GROQ_MODEL)
            return groq

        # Together AI
        together = self._get_together_client()
        if together:
            self._active_provider = "together"
            logger.info("Using Together AI (fallback)", model=settings.TOGETHER_MODEL)
            return together

        # OpenAI (user's key)
        openai = self._get_openai_client()
        if openai:
            self._active_provider = "openai"
            logger.info("Using OpenAI (fallback)", model=settings.OPENAI_MODEL)
            return openai

        # Anthropic (user's key)
        anthropic = self._get_anthropic_client()
        if anthropic:
            self._active_provider = "anthropic"
            logger.info("Using Anthropic (fallback)", model=settings.ANTHROPIC_MODEL)
            return anthropic

        # ===========================================
        # No provider available - show helpful error
        # ===========================================
        raise ValueError(
            "Nenhum provedor de IA disponivel.\n\n"
            "SOLUCAO RECOMENDADA (gratis e local):\n"
            "  docker-compose up -d ollama ollama-init\n"
            "  # Aguarde o modelo ser baixado (~10min)\n\n"
            "Ou configure uma API key no seu .env:\n"
            "  - GROQ_API_KEY (gratis, rapido)\n"
            "  - DEEPSEEK_API_KEY (barato, $0.14/1M tokens)\n"
            "  - OPENAI_API_KEY (GPT-4o)\n"
            "  - ANTHROPIC_API_KEY (Claude)"
        )

    async def chat(
        self,
        message: str,
        history: list[dict] | None = None,
        user_context: dict | None = None,
        preferred_provider: str | None = None,
        use_tools: bool = True,
    ) -> tuple[str, int]:
        """
        Send a chat message and get a response with optional MCP tool calling.

        Args:
            message: The user's message
            history: Previous messages in the conversation
            user_context: Additional context about the user
            preferred_provider: User's preferred LLM provider
            use_tools: Whether to enable MCP tool calling

        Returns:
            Tuple of (response_text, tokens_used)
        """
        client = await self._get_client(preferred_provider)

        # Build messages list
        messages = []

        # Add system prompt with user context
        system_content = INVESTMENT_ASSISTANT_PROMPT
        if user_context:
            context_str = self._format_user_context(user_context)
            system_content += f"\n\nContexto do usuario:\n{context_str}"

        # Add tool usage instructions if enabled
        if use_tools:
            system_content += """

Voce tem acesso a ferramentas para obter dados financeiros em tempo real.
Quando o usuario perguntar sobre cotacoes, precos, indicadores ou analises de ativos,
USE as ferramentas disponiveis para obter dados atualizados antes de responder.
Sempre cite a fonte dos dados e a data/hora da consulta."""

        messages.append({"role": "system", "content": system_content})

        # Add conversation history
        if history:
            for msg in history[-10:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current message
        messages.append({"role": "user", "content": message})

        # Get tools if enabled
        tools = None
        if use_tools:
            try:
                from src.mcp import mcp_client
                tools = mcp_client.get_available_tools()
            except ImportError:
                logger.warning("MCP client not available")

        try:
            # First call - may return tool calls
            response = await client.chat(messages, tools=tools)

            # Handle tool calls (agentic loop)
            if isinstance(response, dict) and response.get("type") == "tool_calls":
                response = await self._handle_tool_calls(client, messages, response, tools)

            content = response if isinstance(response, str) else str(response)
            tokens_used = self._estimate_tokens(messages, content)

            logger.info(
                "LLM response generated",
                provider=self._active_provider,
                tokens_used=tokens_used,
                response_length=len(content),
                used_tools=use_tools,
            )

            return content, tokens_used

        except Exception as e:
            logger.error("LLM chat error", error=str(e), provider=self._active_provider)
            raise

    async def _handle_tool_calls(
        self,
        client,
        messages: list[dict],
        response: dict,
        tools: list[dict] | None,
        max_iterations: int = 5,
    ) -> str:
        """Handle tool calls in an agentic loop."""
        from src.mcp import mcp_client

        iteration = 0
        current_response = response

        while iteration < max_iterations:
            iteration += 1
            tool_calls = current_response.get("tool_calls", [])

            if not tool_calls:
                break

            # Execute each tool call
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name")
                tool_args = tool_call.get("function", {}).get("arguments", {})

                if isinstance(tool_args, str):
                    import json
                    tool_args = json.loads(tool_args)

                logger.info("Executing tool", tool_name=tool_name, args=tool_args)

                try:
                    result = await mcp_client.execute_tool(tool_name, **tool_args)
                    result_str = json.dumps(result, ensure_ascii=False, default=str)
                except Exception as e:
                    result_str = json.dumps({"error": str(e)})
                    logger.error("Tool execution failed", tool_name=tool_name, error=str(e))

                # Add assistant message with tool call
                messages.append({
                    "role": "assistant",
                    "content": current_response.get("content", ""),
                    "tool_calls": tool_calls,
                })

                # Add tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", tool_name),
                    "name": tool_name,
                    "content": result_str,
                })

            # Get next response
            next_response = await client.chat(messages, tools=tools)

            if isinstance(next_response, str):
                return next_response

            current_response = next_response

        # Return final content
        return current_response.get("content", "")

    async def chat_stream(
        self,
        message: str,
        history: list[dict] | None = None,
        user_context: dict | None = None,
        preferred_provider: str | None = None,
        use_tools: bool = False,  # Disabled by default for streaming - makes response immediate
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat response.

        IMPORTANT: For truly interactive streaming, tool calling is disabled by default.
        This ensures the user sees tokens immediately instead of waiting for a blocking
        tool-check call to complete (which can take 1-2+ minutes on slow Ollama).

        To use tools, use the non-streaming `chat()` method instead.

        Args:
            message: The user's message
            history: Previous messages in the conversation
            user_context: Additional context about the user
            preferred_provider: User's preferred LLM provider
            use_tools: Whether to enable MCP tool calling (disabled for fast streaming)

        Yields:
            Chunks of the response text
        """
        client = await self._get_client(preferred_provider)

        # Build messages list
        messages = []

        system_content = INVESTMENT_ASSISTANT_PROMPT
        if user_context:
            context_str = self._format_user_context(user_context)
            system_content += f"\n\nContexto do usuario:\n{context_str}"

        messages.append({"role": "system", "content": system_content})

        if history:
            for msg in history[-10:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": message})

        logger.info(
            "Starting LLM stream",
            provider=self._active_provider,
            model=getattr(client, 'model', 'unknown'),
            use_tools=use_tools,
        )

        try:
            # Stream directly for immediate response
            # Tool calling is handled by the non-streaming chat() method
            async for chunk in client.chat_stream(messages):
                yield chunk

        except Exception as e:
            logger.error("LLM stream error", error=str(e), provider=self._active_provider)
            raise

    def _format_user_context(self, context: dict) -> str:
        """Format user context for the system prompt."""
        parts = []

        if context.get("risk_profile"):
            parts.append(f"- Perfil de risco: {context['risk_profile']}")

        if context.get("investment_horizon"):
            parts.append(f"- Horizonte de investimento: {context['investment_horizon']}")

        if context.get("portfolio_summary"):
            ps = context["portfolio_summary"]
            parts.append(f"- Patrimonio atual: R$ {ps.get('current_value', 0):,.2f}")
            parts.append(f"- Total investido: R$ {ps.get('total_invested', 0):,.2f}")

        if context.get("goals"):
            parts.append(f"- Objetivos: {', '.join(context['goals'])}")

        return "\n".join(parts) if parts else "Nenhum contexto adicional disponivel."

    def _estimate_tokens(self, messages: list, response: str) -> int:
        """Rough estimation of tokens used."""
        total_chars = sum(len(str(m.get("content", ""))) for m in messages) + len(response)
        return total_chars // 4

    def get_provider_info(self) -> dict:
        """Get information about available providers."""
        return {
            "active_provider": self._active_provider,
            "available_providers": {
                "ollama": {
                    "enabled": settings.OLLAMA_ENABLED,
                    "model": settings.OLLAMA_MODEL if settings.OLLAMA_ENABLED else None,
                    "type": "local",
                    "cost": "free",
                },
                "openai": {
                    "enabled": bool(settings.OPENAI_API_KEY),
                    "model": settings.OPENAI_MODEL if settings.OPENAI_API_KEY else None,
                    "type": "api",
                    "cost": "paid",
                },
                "anthropic": {
                    "enabled": bool(settings.ANTHROPIC_API_KEY),
                    "model": settings.ANTHROPIC_MODEL if settings.ANTHROPIC_API_KEY else None,
                    "type": "api",
                    "cost": "paid",
                },
                "deepseek": {
                    "enabled": bool(settings.DEEPSEEK_API_KEY),
                    "model": settings.DEEPSEEK_MODEL if settings.DEEPSEEK_API_KEY else None,
                    "type": "api",
                    "cost": "low",
                },
                "groq": {
                    "enabled": bool(settings.GROQ_API_KEY),
                    "model": settings.GROQ_MODEL if settings.GROQ_API_KEY else None,
                    "type": "api",
                    "cost": "free_tier",
                },
                "together": {
                    "enabled": bool(settings.TOGETHER_API_KEY),
                    "model": settings.TOGETHER_MODEL if settings.TOGETHER_API_KEY else None,
                    "type": "api",
                    "cost": "paid",
                },
            },
        }

    @staticmethod
    def get_available_models() -> dict:
        """Get list of available models for each provider."""
        return {
            "ollama": [
                {"id": "qwen2.5:7b", "name": "Qwen 2.5 7B", "description": "Excelente multilingue"},
                {"id": "qwen2.5:14b", "name": "Qwen 2.5 14B", "description": "Mais poderoso"},
                {"id": "deepseek-r1:7b", "name": "DeepSeek R1 7B", "description": "Melhor raciocinio"},
                {"id": "llama3.3:8b", "name": "Llama 3.3 8B", "description": "Meta, versatil"},
                {"id": "mistral:7b", "name": "Mistral 7B", "description": "Rapido e eficiente"},
            ],
            "openai": [
                {"id": "gpt-4o", "name": "GPT-4o", "description": "Mais recente e rapido"},
                {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "description": "Mais barato"},
                {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "description": "Alta capacidade"},
            ],
            "anthropic": [
                {"id": "claude-sonnet-4-5-20250929", "name": "Claude Sonnet 4.5", "description": "Mais recente"},
                {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet", "description": "Equilibrado"},
                {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus", "description": "Mais poderoso"},
            ],
            "deepseek": [
                {"id": "deepseek-chat", "name": "DeepSeek Chat", "description": "Chat geral"},
                {"id": "deepseek-reasoner", "name": "DeepSeek Reasoner", "description": "Raciocinio avancado"},
            ],
            "groq": [
                {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "description": "Muito poderoso"},
                {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B", "description": "Rapido"},
                {"id": "gemma2-9b-it", "name": "Gemma 2 9B", "description": "Google, eficiente"},
            ],
            "together": [
                {"id": "Qwen/Qwen2.5-72B-Instruct-Turbo", "name": "Qwen 2.5 72B", "description": "Muito poderoso"},
                {"id": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "name": "Llama 3.3 70B", "description": "Meta"},
                {"id": "mistralai/Mixtral-8x22B-Instruct-v0.1", "name": "Mixtral 8x22B", "description": "Mistral"},
            ],
        }


# Singleton instance
llm_client = LLMClient()
