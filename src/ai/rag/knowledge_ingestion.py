"""Knowledge ingestion service for RAG system."""
import structlog
from typing import Optional

from src.ai.rag.financial_rag import rag_service

logger = structlog.get_logger()


class KnowledgeIngestionService:
    """Service for ingesting financial knowledge into RAG."""

    # Pre-defined financial education content
    FINANCIAL_EDUCATION_CONTENT = [
        {
            "topic": "O que é P/L (Preço/Lucro)?",
            "content": """O P/L é um indicador fundamentalista que relaciona o preço da ação
            com o lucro por ação (LPA) da empresa. É calculado dividindo o preço atual da ação
            pelo LPA dos últimos 12 meses. Um P/L de 10 significa que, mantendo o lucro atual,
            seriam necessários 10 anos para o investidor recuperar o investimento apenas com lucros.

            Pontos importantes:
            - P/L baixo não significa necessariamente "barato" - pode indicar problemas
            - Deve ser comparado com empresas do mesmo setor
            - Empresas em crescimento costumam ter P/L mais alto
            - P/L negativo indica prejuízo

            Este indicador deve ser usado em conjunto com outros para uma análise completa.""",
            "difficulty": "beginner",
            "tags": ["fundamentalista", "indicadores", "valuation"],
        },
        {
            "topic": "O que é Dividend Yield?",
            "content": """O Dividend Yield (DY) indica o retorno em dividendos de uma ação
            em relação ao seu preço. É calculado dividindo os dividendos pagos nos últimos
            12 meses pelo preço atual da ação, multiplicado por 100.

            Exemplo: Uma ação de R$100 que pagou R$6 em dividendos tem DY de 6%.

            Considerações importantes:
            - DY alto pode indicar preço deprimido (nem sempre é bom sinal)
            - Empresas maduras tendem a pagar mais dividendos
            - Empresas em crescimento reinvestem lucros em vez de distribuir
            - Compare DY com a taxa Selic e outras alternativas de renda

            Atenção: Dividendos passados não garantem dividendos futuros.""",
            "difficulty": "beginner",
            "tags": ["fundamentalista", "dividendos", "renda passiva"],
        },
        {
            "topic": "O que são FIIs (Fundos Imobiliários)?",
            "content": """FIIs são fundos de investimento que aplicam recursos em
            empreendimentos imobiliários. Ao comprar cotas de um FII, você se torna
            sócio de imóveis como shoppings, galpões logísticos, hospitais ou escritórios.

            Tipos principais:
            - Fundos de Tijolo: Investem em imóveis físicos
            - Fundos de Papel: Investem em títulos do mercado imobiliário (CRIs, LCIs)
            - Fundos de Fundos (FOFs): Investem em cotas de outros FIIs
            - Fundos Híbridos: Combinam diferentes estratégias

            Vantagens:
            - Rendimentos mensais (geralmente isentos de IR para pessoa física)
            - Diversificação com pouco capital
            - Liquidez maior que imóveis físicos

            Riscos:
            - Vacância dos imóveis
            - Inadimplência dos inquilinos
            - Variação do valor das cotas""",
            "difficulty": "beginner",
            "tags": ["FIIs", "fundos", "imóveis", "renda passiva"],
        },
        {
            "topic": "O que é Risco x Retorno?",
            "content": """A relação risco-retorno é um dos princípios fundamentais
            dos investimentos: para obter retornos maiores, geralmente é necessário
            aceitar riscos maiores.

            Tipos de risco:
            - Risco de Mercado: Variações gerais do mercado
            - Risco de Crédito: Possibilidade de calote
            - Risco de Liquidez: Dificuldade de vender o ativo
            - Risco Cambial: Variações em moedas estrangeiras

            Pirâmide de risco (do menor para maior):
            1. Tesouro Selic, CDBs de bancos grandes
            2. Tesouro IPCA+, LCIs/LCAs
            3. Fundos Multimercado, FIIs
            4. Ações de empresas consolidadas (Blue Chips)
            5. Ações de empresas menores (Small Caps)
            6. Criptomoedas, Opções, Day Trade

            Importante: O perfil de risco deve estar alinhado com seus objetivos
            e horizonte de investimento.""",
            "difficulty": "beginner",
            "tags": ["risco", "retorno", "fundamentos"],
        },
        {
            "topic": "O que é Análise Técnica?",
            "content": """A Análise Técnica (AT) estuda o comportamento dos preços
            através de gráficos para identificar padrões e tendências. Diferente da
            análise fundamentalista, não considera os fundamentos da empresa.

            Conceitos básicos:
            - Suporte: Nível de preço onde há demanda suficiente para impedir quedas
            - Resistência: Nível onde há oferta suficiente para impedir altas
            - Tendência: Direção geral dos preços (alta, baixa ou lateral)
            - Volume: Quantidade negociada, confirma ou não movimentos

            Indicadores comuns:
            - Médias Móveis (SMA, EMA)
            - RSI (Índice de Força Relativa)
            - MACD
            - Bandas de Bollinger

            Limitações importantes:
            - Baseada em dados passados, não prevê o futuro
            - Pode gerar sinais falsos
            - Funciona melhor em mercados líquidos
            - Deve ser usada como ferramenta auxiliar, não única""",
            "difficulty": "intermediate",
            "tags": ["análise técnica", "gráficos", "trading"],
        },
        {
            "topic": "O que é Diversificação?",
            "content": """Diversificação é a estratégia de distribuir investimentos
            entre diferentes ativos para reduzir riscos. É o famoso "não colocar
            todos os ovos na mesma cesta".

            Níveis de diversificação:
            1. Por classe de ativo: Renda fixa, ações, FIIs, internacional
            2. Por setor: Bancos, varejo, energia, tecnologia
            3. Por geografia: Brasil, EUA, Europa, emergentes
            4. Por prazo: Curto, médio e longo prazo

            Alocação sugerida por perfil:
            - Conservador: 70-80% renda fixa, 20-30% variável
            - Moderado: 50-60% renda fixa, 40-50% variável
            - Arrojado: 20-30% renda fixa, 70-80% variável

            Importante: Diversificação reduz risco, mas não elimina.
            Em crises severas, correlações aumentam e todos os ativos podem cair juntos.""",
            "difficulty": "beginner",
            "tags": ["diversificação", "alocação", "risco"],
        },
        {
            "topic": "O que é Valuation?",
            "content": """Valuation é o processo de estimar o valor justo de uma
            empresa ou ativo. É usado para identificar se um investimento está
            "caro" ou "barato" em relação ao seu valor intrínseco.

            Principais métodos:

            1. Fluxo de Caixa Descontado (DCF):
               - Projeta fluxos de caixa futuros
               - Desconta a valor presente
               - Mais usado para empresas maduras

            2. Múltiplos de Mercado:
               - P/L, P/VP, EV/EBITDA
               - Compara com empresas similares
               - Mais simples e rápido

            3. Valor Patrimonial:
               - Soma dos ativos menos passivos
               - Usado para empresas com muitos ativos tangíveis

            Limitações:
            - Depende de premissas e projeções
            - Diferentes analistas chegam a valores diferentes
            - Mercado pode demorar para reconhecer valor

            Valuation é uma estimativa, não uma certeza.""",
            "difficulty": "intermediate",
            "tags": ["valuation", "análise fundamentalista", "valor justo"],
        },
        {
            "topic": "O que é Rebalanceamento de Carteira?",
            "content": """Rebalanceamento é o processo de ajustar a alocação da
            carteira de volta aos percentuais-alvo originais. Com o tempo, alguns
            ativos valorizam mais que outros, alterando a composição da carteira.

            Exemplo:
            - Alocação inicial: 60% ações, 40% renda fixa
            - Após 1 ano: 70% ações, 30% renda fixa (ações subiram mais)
            - Rebalanceamento: Vender ações e comprar renda fixa

            Benefícios:
            - Mantém o nível de risco desejado
            - Força a vender na alta e comprar na baixa
            - Disciplina emocional

            Frequência sugerida:
            - Temporal: A cada 6-12 meses
            - Por desvio: Quando alocação desviar 5-10% do alvo

            Custos a considerar:
            - Taxas de corretagem
            - Impostos sobre ganhos de capital
            - Spread na negociação""",
            "difficulty": "intermediate",
            "tags": ["rebalanceamento", "alocação", "gestão de carteira"],
        },
    ]

    MARKET_CATEGORIES = [
        "indices",
        "acoes",
        "fiis",
        "renda_fixa",
        "cripto",
        "internacional",
        "economia",
    ]

    async def ingest_education_content(self) -> int:
        """Ingest all pre-defined education content."""
        count = 0
        for item in self.FINANCIAL_EDUCATION_CONTENT:
            await rag_service.add_financial_education(
                topic=item["topic"],
                content=item["content"],
                difficulty=item["difficulty"],
                tags=item.get("tags", []),
            )
            count += 1

        logger.info(f"Ingested {count} education documents")
        return count

    async def ingest_market_report(
        self,
        content: str,
        source: str,
        category: str,
        tickers: Optional[list[str]] = None,
    ) -> str:
        """Ingest a market report or analysis."""
        if category not in self.MARKET_CATEGORIES:
            raise ValueError(f"Invalid category. Must be one of: {self.MARKET_CATEGORIES}")

        metadata = {}
        if tickers:
            metadata["tickers"] = tickers

        point_id = await rag_service.add_market_knowledge(
            content=content,
            source=source,
            category=category,
            metadata=metadata,
        )

        return point_id

    async def ingest_asset_analysis(
        self,
        ticker: str,
        analysis_type: str,
        content: str,
        analyst: Optional[str] = None,
    ) -> str:
        """Ingest analysis for a specific asset."""
        full_content = f"""
Análise de {ticker}
Tipo: {analysis_type}
{f'Analista: {analyst}' if analyst else ''}

{content}
"""
        return await rag_service.add_market_knowledge(
            content=full_content,
            source=analyst or "InvestAI",
            category="acoes" if not ticker.endswith("11") else "fiis",
            metadata={
                "ticker": ticker,
                "analysis_type": analysis_type,
            },
        )

    async def ingest_economic_indicator(
        self,
        indicator_name: str,
        value: str,
        analysis: str,
        source: str = "BCB",
    ) -> str:
        """Ingest economic indicator data and analysis."""
        content = f"""
Indicador: {indicator_name}
Valor atual: {value}

Análise:
{analysis}
"""
        return await rag_service.add_market_knowledge(
            content=content,
            source=source,
            category="economia",
            metadata={
                "indicator": indicator_name,
                "value": value,
            },
        )


# Global instance
knowledge_service = KnowledgeIngestionService()
