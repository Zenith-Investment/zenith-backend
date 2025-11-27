"""Sentiment analysis module using open-source LLMs."""
import json
from dataclasses import dataclass
from typing import Optional
import structlog

from src.ai.llm.client import llm_client

logger = structlog.get_logger()


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    ticker: str
    overall_sentiment: str  # "positive", "negative", "neutral"
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    key_factors: list[str]
    news_summary: str
    disclaimer: str = (
        "Esta analise de sentimento e baseada em processamento de linguagem natural "
        "e NAO constitui recomendacao de investimento. Sentimento de mercado pode "
        "mudar rapidamente e nao preve movimentos de preco."
    )

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "overall_sentiment": self.overall_sentiment,
            "sentiment_score": round(self.sentiment_score, 2),
            "confidence": round(self.confidence, 2),
            "key_factors": self.key_factors,
            "news_summary": self.news_summary,
            "disclaimer": self.disclaimer,
        }


class SentimentAnalyzer:
    """Analyzes market sentiment using open-source LLMs."""

    SENTIMENT_PROMPT = """Analise o sentimento do mercado para o ativo {ticker} baseado nas seguintes informacoes:

{context}

Retorne uma analise estruturada em formato JSON com os seguintes campos:
- "sentimento": string (positivo/negativo/neutro)
- "score": number de -1 (muito negativo) a 1 (muito positivo)
- "confianca": number de 0 a 1 indicando nivel de confianca
- "fatores": array de strings com principais fatores que influenciam o sentimento
- "resumo": string com resumo das noticias/informacoes relevantes

Importante: Esta analise e apenas informativa e nao constitui recomendacao de investimento.

Responda APENAS com o JSON, sem texto adicional."""

    async def analyze(
        self,
        ticker: str,
        news_texts: Optional[list[str]] = None,
        market_context: Optional[str] = None,
    ) -> SentimentResult:
        """
        Analyze sentiment for a ticker.

        Args:
            ticker: Asset ticker
            news_texts: List of news/text to analyze
            market_context: Additional market context
        """
        # Build context
        context_parts = []

        if news_texts:
            context_parts.append("Noticias recentes:\n" + "\n---\n".join(news_texts[:5]))

        if market_context:
            context_parts.append(f"Contexto de mercado:\n{market_context}")

        if not context_parts:
            # Return neutral if no context
            return SentimentResult(
                ticker=ticker,
                overall_sentiment="neutral",
                sentiment_score=0.0,
                confidence=0.3,
                key_factors=["Sem informacoes suficientes para analise"],
                news_summary="Nao ha noticias ou contexto disponivel para analise.",
            )

        context = "\n\n".join(context_parts)

        try:
            # Use the unified LLM client
            prompt = self.SENTIMENT_PROMPT.format(ticker=ticker, context=context)
            response, _ = await llm_client.chat(
                message=prompt,
                user_context={
                    "task": "sentiment_analysis",
                    "ticker": ticker,
                },
            )

            # Try to extract JSON from response
            content = response.strip()

            # Handle cases where model wraps JSON in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # Parse JSON response
            result = json.loads(content)

            sentiment = result.get("sentimento", "neutro").lower()
            # Normalize sentiment to English
            sentiment_map = {
                "positivo": "positive",
                "negativo": "negative",
                "neutro": "neutral",
            }
            sentiment = sentiment_map.get(sentiment, "neutral")

            return SentimentResult(
                ticker=ticker,
                overall_sentiment=sentiment,
                sentiment_score=float(result.get("score", 0)),
                confidence=float(result.get("confianca", 0.5)),
                key_factors=result.get("fatores", []),
                news_summary=result.get("resumo", ""),
            )

        except json.JSONDecodeError as e:
            logger.error("Failed to parse sentiment JSON", ticker=ticker, error=str(e))
            return SentimentResult(
                ticker=ticker,
                overall_sentiment="neutral",
                sentiment_score=0.0,
                confidence=0.0,
                key_factors=["Erro no processamento da resposta"],
                news_summary="Nao foi possivel processar a analise de sentimento.",
            )

        except Exception as e:
            logger.error("Sentiment analysis failed", ticker=ticker, error=str(e))

            return SentimentResult(
                ticker=ticker,
                overall_sentiment="neutral",
                sentiment_score=0.0,
                confidence=0.0,
                key_factors=["Erro na analise"],
                news_summary="Nao foi possivel realizar a analise de sentimento.",
            )

    async def analyze_market_mood(
        self,
        indices: Optional[dict[str, float]] = None,
    ) -> dict:
        """Analyze overall market mood."""
        mood = {
            "status": "neutral",
            "fear_greed_index": 50,  # 0-100, 0=extreme fear, 100=extreme greed
            "factors": [],
        }

        if indices:
            # Simple analysis based on index changes
            ibov_change = indices.get("^BVSP", 0)
            sp500_change = indices.get("^GSPC", 0)

            if ibov_change > 1 and sp500_change > 0.5:
                mood["status"] = "bullish"
                mood["fear_greed_index"] = 70
                mood["factors"].append("Mercados em alta")
            elif ibov_change < -1 and sp500_change < -0.5:
                mood["status"] = "bearish"
                mood["fear_greed_index"] = 30
                mood["factors"].append("Mercados em queda")
            else:
                mood["status"] = "neutral"
                mood["fear_greed_index"] = 50
                mood["factors"].append("Mercados lateralizados")

        mood["disclaimer"] = (
            "O indice de medo/ganancia e uma estimativa baseada em indicadores de mercado "
            "e nao preve movimentos futuros."
        )

        return mood
