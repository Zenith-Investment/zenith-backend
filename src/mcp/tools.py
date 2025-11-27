"""
Tool definitions for MCP integration.

Defines all tools available to LLMs for function calling.
Compatible with Ollama, OpenAI, and Anthropic function calling formats.
"""
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional
from enum import Enum
import structlog

logger = structlog.get_logger()


class ToolCategory(str, Enum):
    """Tool categories for organization."""
    MARKET_DATA = "market_data"
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    PORTFOLIO = "portfolio"
    NEWS = "news"
    ECONOMIC = "economic"


@dataclass
class ToolParameter:
    """Tool parameter definition."""
    name: str
    type: str  # string, number, integer, boolean, array, object
    description: str
    required: bool = True
    enum: list[str] | None = None
    default: Any = None


@dataclass
class Tool:
    """Tool definition for function calling."""
    name: str
    description: str
    category: ToolCategory
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable[..., Coroutine[Any, Any, Any]] | None = None

    def to_openai_format(self) -> dict:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_ollama_format(self) -> dict:
        """Convert to Ollama tool format (same as OpenAI)."""
        return self.to_openai_format()

    def to_anthropic_format(self) -> dict:
        """Convert to Anthropic tool format."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class ToolRegistry:
    """Registry for all available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._register_builtin_tools()

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug("Tool registered", tool_name=tool.name, category=tool.category.value)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_by_category(self, category: ToolCategory) -> list[Tool]:
        """Get tools by category."""
        return [t for t in self._tools.values() if t.category == category]

    def to_openai_format(self) -> list[dict]:
        """Get all tools in OpenAI format."""
        return [t.to_openai_format() for t in self._tools.values()]

    def to_ollama_format(self) -> list[dict]:
        """Get all tools in Ollama format."""
        return [t.to_ollama_format() for t in self._tools.values()]

    def to_anthropic_format(self) -> list[dict]:
        """Get all tools in Anthropic format."""
        return [t.to_anthropic_format() for t in self._tools.values()]

    async def execute(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        if not tool.handler:
            raise ValueError(f"Tool has no handler: {tool_name}")

        logger.info("Executing tool", tool_name=tool_name, args=kwargs)
        result = await tool.handler(**kwargs)
        logger.info("Tool executed", tool_name=tool_name, result_type=type(result).__name__)
        return result

    def _register_builtin_tools(self) -> None:
        """Register built-in financial tools."""

        # ==========================================
        # Market Data Tools
        # ==========================================

        self.register(Tool(
            name="get_stock_quote",
            description="Obter cotacao atual de uma acao ou FII da B3. Retorna preco atual, variacao do dia, volume e outros dados de mercado.",
            category=ToolCategory.MARKET_DATA,
            parameters=[
                ToolParameter(
                    name="ticker",
                    type="string",
                    description="Codigo do ativo na B3 (ex: PETR4, VALE3, ITUB4, HGLG11)",
                ),
            ],
        ))

        self.register(Tool(
            name="get_stock_history",
            description="Obter historico de precos de uma acao ou FII. Retorna precos de abertura, fechamento, maxima, minima e volume.",
            category=ToolCategory.MARKET_DATA,
            parameters=[
                ToolParameter(
                    name="ticker",
                    type="string",
                    description="Codigo do ativo na B3",
                ),
                ToolParameter(
                    name="period",
                    type="string",
                    description="Periodo do historico",
                    enum=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                    default="1mo",
                    required=False,
                ),
                ToolParameter(
                    name="interval",
                    type="string",
                    description="Intervalo entre pontos",
                    enum=["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"],
                    default="1d",
                    required=False,
                ),
            ],
        ))

        self.register(Tool(
            name="get_market_indices",
            description="Obter principais indices de mercado: IBOVESPA, IFIX (FIIs), S&P 500, Dolar, etc.",
            category=ToolCategory.MARKET_DATA,
            parameters=[],
        ))

        self.register(Tool(
            name="search_stocks",
            description="Buscar acoes, FIIs ou ETFs por nome ou ticker.",
            category=ToolCategory.MARKET_DATA,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Termo de busca (nome ou ticker)",
                ),
                ToolParameter(
                    name="asset_type",
                    type="string",
                    description="Tipo de ativo para filtrar",
                    enum=["stock", "fii", "etf", "bdr", "all"],
                    default="all",
                    required=False,
                ),
            ],
        ))

        # ==========================================
        # Technical Analysis Tools
        # ==========================================

        self.register(Tool(
            name="calculate_technical_indicators",
            description="Calcular indicadores tecnicos para um ativo: RSI, MACD, Medias Moveis, Bandas de Bollinger, etc.",
            category=ToolCategory.TECHNICAL_ANALYSIS,
            parameters=[
                ToolParameter(
                    name="ticker",
                    type="string",
                    description="Codigo do ativo na B3",
                ),
                ToolParameter(
                    name="indicators",
                    type="array",
                    description="Lista de indicadores a calcular",
                    default=["rsi", "macd", "sma_20", "sma_50", "bollinger"],
                    required=False,
                ),
                ToolParameter(
                    name="period",
                    type="string",
                    description="Periodo para calculo",
                    enum=["1mo", "3mo", "6mo", "1y"],
                    default="3mo",
                    required=False,
                ),
            ],
        ))

        self.register(Tool(
            name="identify_chart_patterns",
            description="Identificar padroes graficos em um ativo: suporte, resistencia, tendencias, padroes de candles.",
            category=ToolCategory.TECHNICAL_ANALYSIS,
            parameters=[
                ToolParameter(
                    name="ticker",
                    type="string",
                    description="Codigo do ativo na B3",
                ),
            ],
        ))

        # ==========================================
        # Fundamental Analysis Tools
        # ==========================================

        self.register(Tool(
            name="get_company_fundamentals",
            description="Obter dados fundamentalistas de uma empresa: P/L, P/VP, ROE, Dividend Yield, Margem Liquida, etc.",
            category=ToolCategory.FUNDAMENTAL_ANALYSIS,
            parameters=[
                ToolParameter(
                    name="ticker",
                    type="string",
                    description="Codigo do ativo na B3",
                ),
            ],
        ))

        self.register(Tool(
            name="get_income_statement",
            description="Obter DRE (Demonstracao do Resultado do Exercicio) de uma empresa.",
            category=ToolCategory.FUNDAMENTAL_ANALYSIS,
            parameters=[
                ToolParameter(
                    name="ticker",
                    type="string",
                    description="Codigo do ativo na B3",
                ),
                ToolParameter(
                    name="period",
                    type="string",
                    description="Periodo: anual ou trimestral",
                    enum=["annual", "quarterly"],
                    default="annual",
                    required=False,
                ),
            ],
        ))

        self.register(Tool(
            name="get_balance_sheet",
            description="Obter Balanco Patrimonial de uma empresa.",
            category=ToolCategory.FUNDAMENTAL_ANALYSIS,
            parameters=[
                ToolParameter(
                    name="ticker",
                    type="string",
                    description="Codigo do ativo na B3",
                ),
            ],
        ))

        self.register(Tool(
            name="get_dividends",
            description="Obter historico de dividendos e proventos de um ativo.",
            category=ToolCategory.FUNDAMENTAL_ANALYSIS,
            parameters=[
                ToolParameter(
                    name="ticker",
                    type="string",
                    description="Codigo do ativo na B3",
                ),
                ToolParameter(
                    name="years",
                    type="integer",
                    description="Numero de anos de historico",
                    default=5,
                    required=False,
                ),
            ],
        ))

        self.register(Tool(
            name="compare_stocks",
            description="Comparar multiplos ativos lado a lado com indicadores fundamentalistas.",
            category=ToolCategory.FUNDAMENTAL_ANALYSIS,
            parameters=[
                ToolParameter(
                    name="tickers",
                    type="array",
                    description="Lista de tickers para comparar (max 5)",
                ),
            ],
        ))

        # ==========================================
        # Portfolio Tools
        # ==========================================

        self.register(Tool(
            name="get_user_portfolio",
            description="Obter carteira atual do usuario com posicoes, valores e rentabilidade.",
            category=ToolCategory.PORTFOLIO,
            parameters=[],
        ))

        self.register(Tool(
            name="analyze_portfolio_allocation",
            description="Analisar alocacao da carteira do usuario e sugerir rebalanceamento.",
            category=ToolCategory.PORTFOLIO,
            parameters=[],
        ))

        self.register(Tool(
            name="calculate_portfolio_metrics",
            description="Calcular metricas da carteira: Sharpe Ratio, volatilidade, correlacao, drawdown.",
            category=ToolCategory.PORTFOLIO,
            parameters=[
                ToolParameter(
                    name="period",
                    type="string",
                    description="Periodo para calculo",
                    enum=["1mo", "3mo", "6mo", "1y", "ytd"],
                    default="1y",
                    required=False,
                ),
            ],
        ))

        # ==========================================
        # News & Economic Data
        # ==========================================

        self.register(Tool(
            name="get_market_news",
            description="Obter noticias recentes do mercado financeiro brasileiro.",
            category=ToolCategory.NEWS,
            parameters=[
                ToolParameter(
                    name="ticker",
                    type="string",
                    description="Filtrar noticias por ticker (opcional)",
                    required=False,
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Numero maximo de noticias",
                    default=10,
                    required=False,
                ),
            ],
        ))

        self.register(Tool(
            name="get_economic_calendar",
            description="Obter calendario economico com eventos importantes: COPOM, IPCA, PIB, etc.",
            category=ToolCategory.ECONOMIC,
            parameters=[
                ToolParameter(
                    name="days",
                    type="integer",
                    description="Numero de dias a frente",
                    default=30,
                    required=False,
                ),
            ],
        ))

        self.register(Tool(
            name="get_selic_rate",
            description="Obter taxa SELIC atual e historico.",
            category=ToolCategory.ECONOMIC,
            parameters=[],
        ))

        self.register(Tool(
            name="get_inflation_data",
            description="Obter dados de inflacao (IPCA, IGP-M) atual e historico.",
            category=ToolCategory.ECONOMIC,
            parameters=[
                ToolParameter(
                    name="index",
                    type="string",
                    description="Indice de inflacao",
                    enum=["ipca", "igpm", "inpc"],
                    default="ipca",
                    required=False,
                ),
            ],
        ))


# Singleton instance
tool_registry = ToolRegistry()
