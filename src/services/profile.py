"""Investor profile service with suitability assessment."""
import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.profile import AssessmentSession, InvestorProfile
from src.models.user import User
from src.schemas.profile import (
    AllocationRecommendation,
    AssessmentResultResponse,
    InvestmentGoal,
    InvestmentHorizon,
    RiskProfile,
)

# Suitability Assessment Questions
ASSESSMENT_QUESTIONS = [
    {
        "id": 1,
        "question": "Qual é a sua idade?",
        "category": "demographics",
        "options": [
            {"id": 1, "text": "Menos de 25 anos", "score": 10},
            {"id": 2, "text": "Entre 25 e 35 anos", "score": 8},
            {"id": 3, "text": "Entre 36 e 50 anos", "score": 6},
            {"id": 4, "text": "Entre 51 e 65 anos", "score": 4},
            {"id": 5, "text": "Mais de 65 anos", "score": 2},
        ],
    },
    {
        "id": 2,
        "question": "Qual é a sua principal fonte de renda?",
        "category": "financial_situation",
        "options": [
            {"id": 1, "text": "Salário fixo (CLT)", "score": 6},
            {"id": 2, "text": "Autônomo/Empresário", "score": 8},
            {"id": 3, "text": "Aposentadoria/Pensão", "score": 4},
            {"id": 4, "text": "Rendimentos de investimentos", "score": 10},
            {"id": 5, "text": "Não possuo renda fixa", "score": 2},
        ],
    },
    {
        "id": 3,
        "question": "Qual percentual da sua renda mensal você consegue poupar/investir?",
        "category": "financial_situation",
        "options": [
            {"id": 1, "text": "Menos de 10%", "score": 2},
            {"id": 2, "text": "Entre 10% e 20%", "score": 4},
            {"id": 3, "text": "Entre 20% e 30%", "score": 6},
            {"id": 4, "text": "Entre 30% e 50%", "score": 8},
            {"id": 5, "text": "Mais de 50%", "score": 10},
        ],
    },
    {
        "id": 4,
        "question": "Você possui reserva de emergência (3-6 meses de despesas)?",
        "category": "financial_situation",
        "options": [
            {"id": 1, "text": "Não possuo reserva", "score": 0},
            {"id": 2, "text": "Tenho menos de 3 meses", "score": 3},
            {"id": 3, "text": "Tenho entre 3 e 6 meses", "score": 6},
            {"id": 4, "text": "Tenho entre 6 e 12 meses", "score": 8},
            {"id": 5, "text": "Tenho mais de 12 meses", "score": 10},
        ],
    },
    {
        "id": 5,
        "question": "Qual é o seu principal objetivo com investimentos?",
        "category": "goals",
        "options": [
            {"id": 1, "text": "Preservar meu patrimônio", "score": 2, "goal": "capital_preservation"},
            {"id": 2, "text": "Gerar renda passiva", "score": 4, "goal": "income"},
            {"id": 3, "text": "Aposentadoria", "score": 6, "goal": "retirement"},
            {"id": 4, "text": "Construir patrimônio a longo prazo", "score": 8, "goal": "wealth_building"},
            {"id": 5, "text": "Multiplicar capital rapidamente", "score": 10, "goal": "wealth_building"},
        ],
    },
    {
        "id": 6,
        "question": "Em quanto tempo você pretende utilizar o dinheiro investido?",
        "category": "horizon",
        "options": [
            {"id": 1, "text": "Menos de 1 ano", "score": 2, "horizon": "short_term"},
            {"id": 2, "text": "Entre 1 e 3 anos", "score": 4, "horizon": "short_term"},
            {"id": 3, "text": "Entre 3 e 5 anos", "score": 6, "horizon": "medium_term"},
            {"id": 4, "text": "Entre 5 e 10 anos", "score": 8, "horizon": "long_term"},
            {"id": 5, "text": "Mais de 10 anos", "score": 10, "horizon": "very_long_term"},
        ],
    },
    {
        "id": 7,
        "question": "Qual é o seu nível de conhecimento sobre investimentos?",
        "category": "experience",
        "options": [
            {"id": 1, "text": "Nenhum - nunca investi", "score": 2},
            {"id": 2, "text": "Básico - apenas poupança/CDB", "score": 4},
            {"id": 3, "text": "Intermediário - já invisto em fundos/ações", "score": 6},
            {"id": 4, "text": "Avançado - diversifico em várias classes", "score": 8},
            {"id": 5, "text": "Expert - opero derivativos/day trade", "score": 10},
        ],
    },
    {
        "id": 8,
        "question": "Se seus investimentos caíssem 20% em um mês, o que você faria?",
        "category": "risk_tolerance",
        "options": [
            {"id": 1, "text": "Venderia tudo imediatamente", "score": 0},
            {"id": 2, "text": "Venderia parte para reduzir perdas", "score": 3},
            {"id": 3, "text": "Manteria e aguardaria recuperação", "score": 6},
            {"id": 4, "text": "Compraria mais aproveitando a queda", "score": 10},
        ],
    },
    {
        "id": 9,
        "question": "Qual cenário te deixaria mais confortável?",
        "category": "risk_tolerance",
        "options": [
            {"id": 1, "text": "Ganhar 5% ao ano com certeza", "score": 2},
            {"id": 2, "text": "50% chance de ganhar 10% ou 0%", "score": 4},
            {"id": 3, "text": "50% chance de ganhar 20% ou perder 5%", "score": 6},
            {"id": 4, "text": "50% chance de ganhar 40% ou perder 15%", "score": 8},
            {"id": 5, "text": "50% chance de ganhar 100% ou perder 30%", "score": 10},
        ],
    },
    {
        "id": 10,
        "question": "Quanto do seu patrimônio você está disposto a colocar em risco?",
        "category": "risk_tolerance",
        "options": [
            {"id": 1, "text": "Nada - não aceito perdas", "score": 0},
            {"id": 2, "text": "Até 5% do patrimônio", "score": 3},
            {"id": 3, "text": "Até 15% do patrimônio", "score": 5},
            {"id": 4, "text": "Até 30% do patrimônio", "score": 7},
            {"id": 5, "text": "Mais de 30% se necessário", "score": 10},
        ],
    },
    {
        "id": 11,
        "question": "Como você se sentiria se não pudesse acessar seu dinheiro por 5 anos?",
        "category": "liquidity",
        "options": [
            {"id": 1, "text": "Muito desconfortável", "score": 2},
            {"id": 2, "text": "Desconfortável", "score": 4},
            {"id": 3, "text": "Indiferente", "score": 6},
            {"id": 4, "text": "Confortável", "score": 8},
            {"id": 5, "text": "Totalmente confortável", "score": 10},
        ],
    },
    {
        "id": 12,
        "question": "Você possui dívidas (exceto financiamento imobiliário)?",
        "category": "financial_situation",
        "options": [
            {"id": 1, "text": "Sim, com juros altos (cartão/cheque especial)", "score": 0},
            {"id": 2, "text": "Sim, mas juros baixos (consignado)", "score": 4},
            {"id": 3, "text": "Apenas financiamento de veículo", "score": 6},
            {"id": 4, "text": "Não possuo dívidas", "score": 10},
        ],
    },
]

# Risk profile thresholds and allocations
RISK_PROFILES = {
    RiskProfile.CONSERVATIVE: {
        "min_score": 0,
        "max_score": 30,
        "allocation": [
            {"asset_class": "Renda Fixa Pós-Fixada", "percentage": 50, "description": "CDI, Tesouro Selic, CDBs"},
            {"asset_class": "Renda Fixa Inflação", "percentage": 30, "description": "Tesouro IPCA+, Debêntures"},
            {"asset_class": "Fundos Multimercado", "percentage": 15, "description": "Fundos conservadores"},
            {"asset_class": "Ações/FIIs", "percentage": 5, "description": "Exposição mínima a renda variável"},
        ],
    },
    RiskProfile.MODERATE: {
        "min_score": 31,
        "max_score": 50,
        "allocation": [
            {"asset_class": "Renda Fixa Pós-Fixada", "percentage": 35, "description": "CDI, Tesouro Selic"},
            {"asset_class": "Renda Fixa Inflação", "percentage": 25, "description": "Tesouro IPCA+, CRIs/CRAs"},
            {"asset_class": "Fundos Multimercado", "percentage": 20, "description": "Fundos moderados"},
            {"asset_class": "Ações", "percentage": 10, "description": "Blue chips e dividendos"},
            {"asset_class": "FIIs", "percentage": 10, "description": "Fundos imobiliários"},
        ],
    },
    RiskProfile.BALANCED: {
        "min_score": 51,
        "max_score": 65,
        "allocation": [
            {"asset_class": "Renda Fixa", "percentage": 40, "description": "Mix de pós e IPCA+"},
            {"asset_class": "Ações", "percentage": 25, "description": "Carteira diversificada"},
            {"asset_class": "FIIs", "percentage": 15, "description": "Fundos imobiliários"},
            {"asset_class": "Fundos Multimercado", "percentage": 15, "description": "Fundos balanceados"},
            {"asset_class": "Internacional", "percentage": 5, "description": "ETFs/BDRs"},
        ],
    },
    RiskProfile.GROWTH: {
        "min_score": 66,
        "max_score": 80,
        "allocation": [
            {"asset_class": "Ações", "percentage": 40, "description": "Carteira diversificada + small caps"},
            {"asset_class": "Renda Fixa", "percentage": 25, "description": "Principalmente IPCA+"},
            {"asset_class": "FIIs", "percentage": 15, "description": "Fundos imobiliários"},
            {"asset_class": "Internacional", "percentage": 15, "description": "ETFs globais"},
            {"asset_class": "Alternativos", "percentage": 5, "description": "Criptomoedas, commodities"},
        ],
    },
    RiskProfile.AGGRESSIVE: {
        "min_score": 81,
        "max_score": 100,
        "allocation": [
            {"asset_class": "Ações", "percentage": 50, "description": "Small caps, growth, IPOs"},
            {"asset_class": "Internacional", "percentage": 20, "description": "ETFs e ações globais"},
            {"asset_class": "Alternativos", "percentage": 15, "description": "Cripto, venture, commodities"},
            {"asset_class": "FIIs", "percentage": 10, "description": "FIIs de tijolo e papel"},
            {"asset_class": "Renda Fixa", "percentage": 5, "description": "Apenas reserva estratégica"},
        ],
    },
}


class ProfileService:
    """Service for investor profile and suitability assessment."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def start_assessment(self, user: User) -> dict:
        """Start a new assessment session."""
        session_id = str(uuid.uuid4())

        session = AssessmentSession(
            session_id=session_id,
            user_id=user.id,
            status="in_progress",
        )
        self.db.add(session)
        await self.db.commit()

        return {
            "session_id": session_id,
            "total_questions": len(ASSESSMENT_QUESTIONS),
            "questions": ASSESSMENT_QUESTIONS,
        }

    async def get_session(self, session_id: str) -> AssessmentSession | None:
        """Get assessment session by ID."""
        result = await self.db.execute(
            select(AssessmentSession).where(AssessmentSession.session_id == session_id)
        )
        return result.scalar_one_or_none()

    async def submit_answers(
        self, user: User, session_id: str, answers: dict[int, int]
    ) -> AssessmentResultResponse:
        """Submit all answers and calculate profile."""
        session = await self.get_session(session_id)
        if not session or session.user_id != user.id:
            raise ValueError("Invalid session")

        # Calculate score
        total_score = 0
        max_possible_score = 0
        goals = []
        horizon = InvestmentHorizon.MEDIUM_TERM

        for question in ASSESSMENT_QUESTIONS:
            question_id = question["id"]
            if question_id in answers:
                answer_id = answers[question_id]
                for option in question["options"]:
                    if option["id"] == answer_id:
                        total_score += option["score"]
                        if "goal" in option:
                            goals.append(option["goal"])
                        if "horizon" in option:
                            horizon = InvestmentHorizon(option["horizon"])
                        break
            max_possible_score += max(opt["score"] for opt in question["options"])

        # Normalize score to 0-100
        normalized_score = int((total_score / max_possible_score) * 100)

        # Determine risk profile
        risk_profile = self._get_risk_profile(normalized_score)

        # Get allocation recommendation
        allocation = self._get_allocation(risk_profile)

        # Save result
        session.status = "completed"
        session.answers = json.dumps(answers)
        session.result = json.dumps({
            "risk_score": normalized_score,
            "risk_profile": risk_profile.value,
            "horizon": horizon.value,
            "goals": goals,
        })
        session.completed_at = datetime.now(timezone.utc)

        # Create or update investor profile
        await self._save_investor_profile(
            user, normalized_score, risk_profile, horizon, goals
        )

        await self.db.commit()

        # Generate explanation
        explanation = self._generate_explanation(risk_profile, normalized_score)

        return AssessmentResultResponse(
            risk_profile=risk_profile,
            risk_score=normalized_score,
            investment_horizon=horizon,
            primary_goals=[InvestmentGoal(g) for g in goals[:3]] if goals else [InvestmentGoal.WEALTH_BUILDING],
            recommended_allocation=allocation,
            explanation=explanation,
        )

    def _get_risk_profile(self, score: int) -> RiskProfile:
        """Determine risk profile based on score."""
        for profile, config in RISK_PROFILES.items():
            if config["min_score"] <= score <= config["max_score"]:
                return profile
        return RiskProfile.MODERATE

    def _get_allocation(self, profile: RiskProfile) -> list[AllocationRecommendation]:
        """Get recommended allocation for profile."""
        config = RISK_PROFILES[profile]
        return [
            AllocationRecommendation(
                asset_class=item["asset_class"],
                percentage=item["percentage"],
                description=item["description"],
            )
            for item in config["allocation"]
        ]

    def _generate_explanation(self, profile: RiskProfile, score: int) -> str:
        """Generate personalized explanation."""
        explanations = {
            RiskProfile.CONSERVATIVE: f"""
Com base nas suas respostas, identificamos que você tem um perfil **Conservador** (score: {score}/100).

Isso significa que você prioriza a segurança do seu capital e prefere investimentos com menor volatilidade, mesmo que isso signifique retornos mais modestos.

Nossa recomendação é focar em renda fixa de qualidade, com pequena exposição a renda variável para potencializar seus ganhos no longo prazo.
""",
            RiskProfile.MODERATE: f"""
Com base nas suas respostas, identificamos que você tem um perfil **Moderado** (score: {score}/100).

Você busca um equilíbrio entre segurança e rentabilidade. Aceita alguma volatilidade em troca de melhores retornos no médio/longo prazo.

Recomendamos uma carteira diversificada com predominância de renda fixa, mas com exposição relevante a FIIs e ações de empresas sólidas.
""",
            RiskProfile.BALANCED: f"""
Com base nas suas respostas, identificamos que você tem um perfil **Balanceado** (score: {score}/100).

Você está confortável com oscilações de mercado e entende que volatilidade faz parte do processo de construção de patrimônio.

Sua carteira ideal combina renda fixa e variável em proporções similares, permitindo capturar oportunidades de crescimento.
""",
            RiskProfile.GROWTH: f"""
Com base nas suas respostas, identificamos que você tem um perfil **Arrojado** (score: {score}/100).

Você prioriza o crescimento do patrimônio e tem tolerância a riscos. Entende que perdas temporárias são aceitáveis em busca de maiores retornos.

Recomendamos maior exposição a renda variável, incluindo ações e investimentos internacionais.
""",
            RiskProfile.AGGRESSIVE: f"""
Com base nas suas respostas, identificamos que você tem um perfil **Agressivo** (score: {score}/100).

Você busca maximizar retornos e aceita alta volatilidade. Tem conhecimento de mercado e horizonte de longo prazo.

Sua carteira pode incluir ativos de maior risco como small caps, criptomoedas e investimentos alternativos.
""",
        }
        return explanations.get(profile, "").strip()

    async def _save_investor_profile(
        self,
        user: User,
        score: int,
        profile: RiskProfile,
        horizon: InvestmentHorizon,
        goals: list[str],
    ) -> InvestorProfile:
        """Create or update investor profile."""
        # Check if profile exists
        result = await self.db.execute(
            select(InvestorProfile).where(InvestorProfile.user_id == user.id)
        )
        investor_profile = result.scalar_one_or_none()

        if investor_profile:
            investor_profile.risk_profile = profile
            investor_profile.risk_score = score
            investor_profile.investment_horizon = horizon
            investor_profile.primary_goals = goals
        else:
            investor_profile = InvestorProfile(
                user_id=user.id,
                risk_profile=profile,
                risk_score=score,
                investment_horizon=horizon,
                primary_goals=goals,
            )
            self.db.add(investor_profile)

        return investor_profile

    async def get_investor_profile(self, user: User) -> InvestorProfile | None:
        """Get user's investor profile."""
        result = await self.db.execute(
            select(InvestorProfile).where(InvestorProfile.user_id == user.id)
        )
        return result.scalar_one_or_none()
