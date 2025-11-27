"""LGPD (Lei Geral de Proteção de Dados) compliance service."""
import json
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.user import User
from src.models.portfolio import Portfolio, PortfolioAsset, Transaction
from src.models.profile import InvestorProfile, AssessmentSession
from src.models.chat import ChatSession, ChatMessage
from src.models.alert import PriceAlert
from src.models.broker import BrokerConnection, BrokerSyncHistory
from src.models.subscription import Subscription, Payment
from src.core.security import decrypt_cpf, mask_cpf

logger = structlog.get_logger()


class LGPDService:
    """Service for LGPD compliance operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def export_user_data(self, user: User) -> dict[str, Any]:
        """
        Export all user data (LGPD Art. 18, III - Portabilidade).

        Returns all personal data associated with the user in a
        structured, machine-readable format.
        """
        export_data = {
            "export_date": datetime.now(timezone.utc).isoformat(),
            "user_id": user.id,
            "data_categories": {},
        }

        # 1. Personal Information
        personal_data = {
            "email": user.email,
            "full_name": user.full_name,
            "phone": user.phone,
            "cpf_masked": None,
            "avatar_url": user.avatar_url,
            "is_active": user.is_active,
            "is_verified": user.is_verified,
            "subscription_plan": user.subscription_plan.value,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
        }

        # Include masked CPF if available
        if user.cpf_encrypted:
            try:
                cpf = decrypt_cpf(user.cpf_encrypted)
                personal_data["cpf_masked"] = mask_cpf(cpf)
            except Exception:
                personal_data["cpf_masked"] = "***.***.***-**"

        export_data["data_categories"]["personal_information"] = personal_data

        # 2. Investor Profile
        result = await self.db.execute(
            select(InvestorProfile).where(InvestorProfile.user_id == user.id)
        )
        profile = result.scalar_one_or_none()

        if profile:
            export_data["data_categories"]["investor_profile"] = {
                "risk_score": profile.risk_score,
                "risk_category": profile.risk_category,
                "investment_horizon": profile.investment_horizon,
                "monthly_income": str(profile.monthly_income) if profile.monthly_income else None,
                "total_assets": str(profile.total_assets) if profile.total_assets else None,
                "investment_experience": profile.investment_experience,
                "investment_goals": profile.investment_goals,
                "created_at": profile.created_at.isoformat() if profile.created_at else None,
            }

        # 3. Portfolio Data
        result = await self.db.execute(
            select(Portfolio)
            .where(Portfolio.user_id == user.id)
            .options(selectinload(Portfolio.assets))
        )
        portfolios = result.scalars().all()

        portfolio_data = []
        for portfolio in portfolios:
            assets_data = [
                {
                    "ticker": asset.ticker,
                    "name": asset.name,
                    "quantity": str(asset.quantity),
                    "average_price": str(asset.average_price),
                    "asset_class": asset.asset_class,
                    "broker": asset.broker,
                    "created_at": asset.created_at.isoformat() if asset.created_at else None,
                }
                for asset in portfolio.assets
            ]

            portfolio_data.append({
                "name": portfolio.name,
                "description": portfolio.description,
                "assets": assets_data,
                "created_at": portfolio.created_at.isoformat() if portfolio.created_at else None,
            })

        export_data["data_categories"]["portfolios"] = portfolio_data

        # 4. Transaction History
        result = await self.db.execute(
            select(Transaction)
            .where(Transaction.user_id == user.id)
            .order_by(Transaction.transaction_date.desc())
        )
        transactions = result.scalars().all()

        export_data["data_categories"]["transactions"] = [
            {
                "ticker": t.ticker,
                "transaction_type": t.transaction_type,
                "quantity": str(t.quantity),
                "price": str(t.price),
                "total_value": str(t.total_value) if t.total_value else None,
                "fees": str(t.fees) if t.fees else None,
                "broker": t.broker,
                "transaction_date": t.transaction_date.isoformat() if t.transaction_date else None,
            }
            for t in transactions
        ]

        # 5. Chat History
        result = await self.db.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user.id)
            .options(selectinload(ChatSession.messages))
        )
        sessions = result.scalars().all()

        chat_data = []
        for session in sessions:
            messages = [
                {
                    "role": msg.role,
                    "content": msg.content[:500] + "..." if len(msg.content) > 500 else msg.content,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None,
                }
                for msg in session.messages
            ]
            chat_data.append({
                "session_id": session.id,
                "title": session.title,
                "messages_count": len(messages),
                "messages": messages,
                "created_at": session.created_at.isoformat() if session.created_at else None,
            })

        export_data["data_categories"]["chat_history"] = chat_data

        # 6. Price Alerts
        result = await self.db.execute(
            select(PriceAlert).where(PriceAlert.user_id == user.id)
        )
        alerts = result.scalars().all()

        export_data["data_categories"]["price_alerts"] = [
            {
                "ticker": alert.ticker,
                "target_price": str(alert.target_price),
                "condition": alert.condition,
                "is_active": alert.is_active,
                "created_at": alert.created_at.isoformat() if alert.created_at else None,
            }
            for alert in alerts
        ]

        # 7. Broker Connections (without tokens)
        result = await self.db.execute(
            select(BrokerConnection).where(BrokerConnection.user_id == user.id)
        )
        connections = result.scalars().all()

        export_data["data_categories"]["broker_connections"] = [
            {
                "broker_type": conn.broker_type.value,
                "status": conn.status.value,
                "broker_account_id": conn.broker_account_id,
                "broker_account_name": conn.broker_account_name,
                "last_sync_at": conn.last_sync_at.isoformat() if conn.last_sync_at else None,
                "created_at": conn.created_at.isoformat() if conn.created_at else None,
            }
            for conn in connections
        ]

        # 8. Subscription & Payments
        result = await self.db.execute(
            select(Subscription).where(Subscription.user_id == user.id)
        )
        subscription = result.scalar_one_or_none()

        if subscription:
            export_data["data_categories"]["subscription"] = {
                "plan": subscription.plan.value,
                "status": subscription.status.value,
                "billing_cycle": subscription.billing_cycle,
                "current_period_start": subscription.current_period_start.isoformat() if subscription.current_period_start else None,
                "current_period_end": subscription.current_period_end.isoformat() if subscription.current_period_end else None,
                "created_at": subscription.created_at.isoformat() if subscription.created_at else None,
            }

        result = await self.db.execute(
            select(Payment).where(Payment.user_id == user.id)
        )
        payments = result.scalars().all()

        export_data["data_categories"]["payments"] = [
            {
                "amount": str(p.amount),
                "currency": p.currency,
                "status": p.status.value,
                "payment_method": p.payment_method.value if p.payment_method else None,
                "card_last_four": p.card_last_four,
                "paid_at": p.paid_at.isoformat() if p.paid_at else None,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in payments
        ]

        logger.info("User data exported", user_id=user.id)
        return export_data

    async def delete_user_data(
        self,
        user: User,
        keep_financial_records: bool = True,
    ) -> dict:
        """
        Delete user data (LGPD Art. 18, VI - Eliminação).

        By default, keeps financial records for legal compliance.

        Args:
            user: User object
            keep_financial_records: If True, keeps payment/transaction data
                                   for legal compliance (5 years in Brazil)

        Returns:
            Summary of deleted data
        """
        deleted_summary = {
            "user_id": user.id,
            "deleted_at": datetime.now(timezone.utc).isoformat(),
            "categories_deleted": [],
            "categories_retained": [],
        }

        # 1. Delete chat history
        await self.db.execute(
            delete(ChatMessage).where(
                ChatMessage.session_id.in_(
                    select(ChatSession.id).where(ChatSession.user_id == user.id)
                )
            )
        )
        await self.db.execute(
            delete(ChatSession).where(ChatSession.user_id == user.id)
        )
        deleted_summary["categories_deleted"].append("chat_history")

        # 2. Delete price alerts
        await self.db.execute(
            delete(PriceAlert).where(PriceAlert.user_id == user.id)
        )
        deleted_summary["categories_deleted"].append("price_alerts")

        # 3. Delete assessment sessions
        await self.db.execute(
            delete(AssessmentSession).where(
                AssessmentSession.profile_id.in_(
                    select(InvestorProfile.id).where(InvestorProfile.user_id == user.id)
                )
            )
        )

        # 4. Delete investor profile
        await self.db.execute(
            delete(InvestorProfile).where(InvestorProfile.user_id == user.id)
        )
        deleted_summary["categories_deleted"].append("investor_profile")

        # 5. Delete broker sync history and connections
        await self.db.execute(
            delete(BrokerSyncHistory).where(
                BrokerSyncHistory.connection_id.in_(
                    select(BrokerConnection.id).where(BrokerConnection.user_id == user.id)
                )
            )
        )
        await self.db.execute(
            delete(BrokerConnection).where(BrokerConnection.user_id == user.id)
        )
        deleted_summary["categories_deleted"].append("broker_connections")

        if keep_financial_records:
            # Keep transactions and payments for legal compliance
            deleted_summary["categories_retained"].extend([
                "transactions (retained for legal compliance - 5 years)",
                "payments (retained for legal compliance - 5 years)",
            ])

            # Anonymize portfolio data but keep structure
            result = await self.db.execute(
                select(Portfolio).where(Portfolio.user_id == user.id)
            )
            portfolios = result.scalars().all()
            for portfolio in portfolios:
                portfolio.name = "Deleted User Portfolio"
                portfolio.description = None
        else:
            # Delete everything
            await self.db.execute(
                delete(Transaction).where(Transaction.user_id == user.id)
            )
            deleted_summary["categories_deleted"].append("transactions")

            await self.db.execute(
                delete(Payment).where(Payment.user_id == user.id)
            )
            await self.db.execute(
                delete(Subscription).where(Subscription.user_id == user.id)
            )
            deleted_summary["categories_deleted"].append("payments")

            await self.db.execute(
                delete(PortfolioAsset).where(
                    PortfolioAsset.portfolio_id.in_(
                        select(Portfolio.id).where(Portfolio.user_id == user.id)
                    )
                )
            )
            await self.db.execute(
                delete(Portfolio).where(Portfolio.user_id == user.id)
            )
            deleted_summary["categories_deleted"].append("portfolios")

        # 6. Anonymize user (soft delete)
        user.email = f"deleted_{user.id}@deleted.investai.com.br"
        user.full_name = "Usuário Excluído"
        user.phone = None
        user.cpf_encrypted = None
        user.avatar_url = None
        user.is_active = False
        user.hashed_password = "DELETED"
        user.updated_at = datetime.now(timezone.utc)

        await self.db.commit()

        logger.info(
            "User data deleted",
            user_id=user.id,
            categories_deleted=deleted_summary["categories_deleted"],
        )

        return deleted_summary

    async def get_data_processing_info(self) -> dict:
        """
        Get information about data processing (LGPD Art. 18, I).

        Returns information about what data is collected and why.
        """
        return {
            "controller": {
                "name": "InvestAI Platform",
                "email": "privacidade@investai.com.br",
                "address": "Brasil",
            },
            "data_categories": {
                "personal_identification": {
                    "data": ["Nome completo", "Email", "Telefone", "CPF"],
                    "purpose": "Identificacao do usuario e cumprimento de obrigacoes legais",
                    "legal_basis": "Execucao de contrato e cumprimento de obrigacao legal",
                    "retention": "Durante a vigencia da conta + 5 anos apos encerramento",
                },
                "financial_data": {
                    "data": ["Portfolio de investimentos", "Transacoes", "Dados de corretoras"],
                    "purpose": "Prestacao do servico de gestao de portfolio",
                    "legal_basis": "Execucao de contrato",
                    "retention": "Durante a vigencia da conta + 5 anos apos encerramento",
                },
                "investment_profile": {
                    "data": ["Perfil de risco", "Objetivos de investimento", "Experiencia"],
                    "purpose": "Personalizacao de recomendacoes e analises",
                    "legal_basis": "Consentimento e execucao de contrato",
                    "retention": "Durante a vigencia da conta",
                },
                "usage_data": {
                    "data": ["Historico de chat com IA", "Alertas configurados"],
                    "purpose": "Melhoria do servico e personalizacao",
                    "legal_basis": "Interesse legitimo e consentimento",
                    "retention": "Durante a vigencia da conta",
                },
                "payment_data": {
                    "data": ["Historico de pagamentos", "Ultimos 4 digitos do cartao"],
                    "purpose": "Processamento de pagamentos e emissao de notas fiscais",
                    "legal_basis": "Execucao de contrato e cumprimento de obrigacao legal",
                    "retention": "5 anos apos o pagamento (obrigacao fiscal)",
                },
            },
            "data_sharing": {
                "payment_processors": "Stripe (processamento de pagamentos)",
                "broker_integrations": "Corretoras autorizadas pelo usuario",
                "ai_providers": "Modelos open-source locais (Qwen, DeepSeek, Llama) - dados processados localmente",
            },
            "user_rights": [
                "Confirmacao da existencia de tratamento (Art. 18, I)",
                "Acesso aos dados (Art. 18, II)",
                "Correcao de dados incompletos (Art. 18, III)",
                "Anonimizacao, bloqueio ou eliminacao (Art. 18, IV)",
                "Portabilidade dos dados (Art. 18, V)",
                "Eliminacao dos dados pessoais (Art. 18, VI)",
                "Informacao sobre compartilhamento (Art. 18, VII)",
                "Revogacao do consentimento (Art. 18, IX)",
            ],
            "contact": {
                "dpo_email": "dpo@investai.com.br",
                "support_email": "suporte@investai.com.br",
            },
        }

    async def revoke_consent(self, user: User, consent_type: str) -> bool:
        """
        Revoke specific consent (LGPD Art. 18, IX).

        Args:
            user: User object
            consent_type: Type of consent to revoke

        Returns:
            True if consent was revoked successfully
        """
        # For MVP, we track consent revocation by disabling features
        consent_actions = {
            "marketing": self._revoke_marketing_consent,
            "ai_analysis": self._revoke_ai_consent,
            "broker_sync": self._revoke_broker_consent,
        }

        action = consent_actions.get(consent_type)
        if not action:
            return False

        await action(user)
        await self.db.commit()

        logger.info(
            "Consent revoked",
            user_id=user.id,
            consent_type=consent_type,
        )

        return True

    async def _revoke_marketing_consent(self, user: User) -> None:
        """Revoke marketing consent - stop sending promotional emails."""
        # In a real implementation, this would update a user preference
        pass

    async def _revoke_ai_consent(self, user: User) -> None:
        """Revoke AI analysis consent - delete chat history."""
        await self.db.execute(
            delete(ChatMessage).where(
                ChatMessage.session_id.in_(
                    select(ChatSession.id).where(ChatSession.user_id == user.id)
                )
            )
        )
        await self.db.execute(
            delete(ChatSession).where(ChatSession.user_id == user.id)
        )

    async def _revoke_broker_consent(self, user: User) -> None:
        """Revoke broker sync consent - disconnect all brokers."""
        await self.db.execute(
            delete(BrokerSyncHistory).where(
                BrokerSyncHistory.connection_id.in_(
                    select(BrokerConnection.id).where(BrokerConnection.user_id == user.id)
                )
            )
        )
        await self.db.execute(
            delete(BrokerConnection).where(BrokerConnection.user_id == user.id)
        )
