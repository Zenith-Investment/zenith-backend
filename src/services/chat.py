"""Chat service for managing conversations with the AI assistant."""
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.ai.llm.client import llm_client, LLMClient
from src.models.chat import ChatFeedback, ChatMessage, ChatSession
from src.models.settings import UserSettings, LLMProvider
from src.models.user import User
from src.schemas.chat import (
    ChatHistoryResponse,
    ChatMessage as ChatMessageSchema,
    ChatMessageResponse,
    ChatSession as ChatSessionSchema,
    FeedbackType,
    MessageRole,
    SuggestedQuestion,
)

logger = structlog.get_logger()


class ChatService:
    """Service for chat operations with AI assistant."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def _get_user_llm_settings(self, user_id: int) -> tuple[Optional[str], Optional[LLMClient]]:
        """
        Get user's preferred LLM provider and create a custom client if needed.

        Returns:
            Tuple of (preferred_provider, custom_client)
            - preferred_provider: The provider name to use
            - custom_client: A custom LLMClient if user has custom API keys, None otherwise
        """
        result = await self.db.execute(
            select(UserSettings).where(UserSettings.user_id == user_id)
        )
        settings = result.scalar_one_or_none()

        if not settings or settings.llm_provider == LLMProvider.AUTO:
            # Use system defaults
            return None, None

        preferred_provider = settings.llm_provider.value

        # Check if user has custom API keys for the selected provider
        user_api_key = None
        if preferred_provider == "openai" and settings.openai_api_key:
            user_api_key = settings.openai_api_key
        elif preferred_provider == "anthropic" and settings.anthropic_api_key:
            user_api_key = settings.anthropic_api_key
        elif preferred_provider == "deepseek" and settings.deepseek_api_key:
            user_api_key = settings.deepseek_api_key
        elif preferred_provider == "groq" and settings.groq_api_key:
            user_api_key = settings.groq_api_key
        elif preferred_provider == "together" and settings.together_api_key:
            user_api_key = settings.together_api_key

        # If user has a custom API key, we might need to create a custom client
        # For now, we just return the preferred provider and let the LLM client handle it
        # In the future, we could create user-specific clients here

        return preferred_provider, None

    async def get_or_create_session(
        self, user_id: int, session_id: str | None = None
    ) -> ChatSession:
        """Get existing session or create a new one."""
        if session_id:
            result = await self.db.execute(
                select(ChatSession)
                .where(
                    ChatSession.session_id == session_id,
                    ChatSession.user_id == user_id,
                )
                .options(selectinload(ChatSession.messages))
            )
            session = result.scalar_one_or_none()
            if session:
                return session

        # Create new session
        new_session = ChatSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
        )
        self.db.add(new_session)
        await self.db.commit()
        await self.db.refresh(new_session)

        logger.info("New chat session created", session_id=new_session.session_id, user_id=user_id)

        return new_session

    async def send_message(
        self, user: User, message: str, session_id: str | None = None
    ) -> ChatMessageResponse:
        """Send a message and get AI response."""
        # Extract user_id to avoid accessing detached object in async context
        user_id = user.id

        # Get or create session
        session = await self.get_or_create_session(user_id, session_id)

        # Extract session attributes BEFORE any commits to avoid greenlet errors
        # after commit expires the object
        session_db_id = session.id
        session_uuid = session.session_id
        session_has_title = session.title is not None

        # Get user's LLM preferences
        preferred_provider, _ = await self._get_user_llm_settings(user_id)

        # Get user context for personalized responses
        user_context = await self._get_user_context(user_id)

        # Get conversation history
        history = await self._get_conversation_history(session_db_id)

        # Save user message
        user_message = ChatMessage(
            message_id=str(uuid.uuid4()),
            session_id=session_db_id,
            role=MessageRole.USER,
            content=message,
        )
        self.db.add(user_message)
        await self.db.commit()

        try:
            # Get AI response with user's preferred provider
            response_text, tokens_used = await llm_client.chat(
                message=message,
                history=history,
                user_context=user_context,
                preferred_provider=preferred_provider,
            )

            # Save assistant message
            assistant_message = ChatMessage(
                message_id=str(uuid.uuid4()),
                session_id=session_db_id,
                role=MessageRole.ASSISTANT,
                content=response_text,
                tokens_used=tokens_used,
                model_used=llm_client.model,
            )
            self.db.add(assistant_message)

            # Update session title if first message - need to re-fetch session
            if not session_has_title:
                result = await self.db.execute(
                    select(ChatSession).where(ChatSession.id == session_db_id)
                )
                session_to_update = result.scalar_one_or_none()
                if session_to_update:
                    session_to_update.title = self._generate_session_title(message)

            await self.db.commit()
            await self.db.refresh(assistant_message)

            logger.info(
                "Chat message processed",
                session_id=session_uuid,
                tokens_used=tokens_used,
            )

            # Generate suggested follow-up questions
            suggested_questions = self._generate_suggested_questions(message, response_text)

            return ChatMessageResponse(
                id=assistant_message.message_id,
                session_id=session_uuid,
                message=ChatMessageSchema(
                    id=assistant_message.message_id,
                    role=assistant_message.role,
                    content=assistant_message.content,
                    created_at=assistant_message.created_at,
                ),
                suggested_questions=suggested_questions,
            )

        except Exception as e:
            logger.error("Failed to process chat message", error=str(e))
            # Save error message
            error_message = ChatMessage(
                message_id=str(uuid.uuid4()),
                session_id=session_db_id,
                role=MessageRole.ASSISTANT,
                content="Desculpe, ocorreu um erro ao processar sua mensagem. Por favor, tente novamente.",
            )
            self.db.add(error_message)
            await self.db.commit()
            raise

    async def prepare_stream(
        self, user: User, message: str, session_id: str | None = None
    ) -> tuple[int, str | None, dict, list[dict]]:
        """
        Prepare data for streaming - all DB operations happen here.

        Returns:
            Tuple of (session_db_id, preferred_provider, user_context, history)
        """
        # Extract user_id to avoid accessing detached object in async context
        user_id = user.id

        session = await self.get_or_create_session(user_id, session_id)

        # Extract session.id BEFORE any await operations to avoid
        # greenlet errors after commit expires the object
        session_db_id = session.id

        # Get user's LLM preferences
        preferred_provider, _ = await self._get_user_llm_settings(user_id)

        user_context = await self._get_user_context(user_id)
        history = await self._get_conversation_history(session_db_id)

        # Save user message
        user_message = ChatMessage(
            message_id=str(uuid.uuid4()),
            session_id=session_db_id,
            role=MessageRole.USER,
            content=message,
        )
        self.db.add(user_message)
        await self.db.commit()

        return session_db_id, preferred_provider, user_context, history

    async def save_stream_response(
        self, session_id: int, response: str, first_message: str, message_id: str | None = None
    ) -> str:
        """Save the streamed response to database.

        Args:
            session_id: The database ID of the chat session
            response: The full assistant response content
            first_message: The first user message (for title generation)
            message_id: Pre-generated message ID (or will generate a new one)

        Returns:
            The message_id that was saved (for feedback)
        """
        from src.core.database import async_session_factory

        # Use provided message_id or generate a new one
        final_message_id = message_id or str(uuid.uuid4())

        async with async_session_factory() as db:
            # Get session
            result = await db.execute(
                select(ChatSession).where(ChatSession.id == session_id)
            )
            session = result.scalar_one_or_none()

            if session:
                # Save assistant message
                assistant_message = ChatMessage(
                    message_id=final_message_id,
                    session_id=session_id,
                    role=MessageRole.ASSISTANT,
                    content=response,
                    model_used=llm_client.model,
                )
                db.add(assistant_message)

                if not session.title:
                    session.title = self._generate_session_title(first_message)

                await db.commit()

        return final_message_id

    async def stream_message(
        self, user: User, message: str, session_id: str | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream a message response."""
        # Prepare everything with DB operations first
        session_db_id, preferred_provider, user_context, history = await self.prepare_stream(
            user, message, session_id
        )

        # Stream response - no DB operations here
        full_response = ""
        try:
            async for chunk in llm_client.chat_stream(
                message=message,
                history=history,
                user_context=user_context,
                preferred_provider=preferred_provider,
            ):
                full_response += chunk
                yield chunk

        except Exception as e:
            logger.error("Stream error", error=str(e))
            raise
        finally:
            # Save response with a new DB session
            if full_response:
                try:
                    await self.save_stream_response(session_db_id, full_response, message)
                except Exception as e:
                    logger.error("Failed to save stream response", error=str(e))

    async def get_history(
        self, user: User, session_id: str | None = None
    ) -> ChatHistoryResponse:
        """Get chat history for user."""
        # Extract user_id to avoid accessing detached object in async context
        user_id = user.id

        # Get all sessions for user
        result = await self.db.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc().nullslast(), ChatSession.created_at.desc())
            .options(selectinload(ChatSession.messages))
        )
        sessions = result.scalars().all()

        sessions_list = []
        messages_list = []

        for session in sessions:
            sessions_list.append(
                ChatSessionSchema(
                    id=session.session_id,
                    title=session.title,
                    created_at=session.created_at,
                    updated_at=session.updated_at or session.created_at,
                    message_count=len(session.messages),
                )
            )

            # If specific session requested, include messages
            if session_id and session.session_id == session_id:
                for msg in session.messages:
                    messages_list.append(
                        ChatMessageSchema(
                            id=msg.message_id,
                            role=msg.role,
                            content=msg.content,
                            created_at=msg.created_at,
                        )
                    )

        return ChatHistoryResponse(
            sessions=sessions_list,
            messages=messages_list,
        )

    async def submit_feedback(
        self, user: User, message_id: str, feedback_type: FeedbackType, comment: str | None = None
    ) -> bool:
        """Submit feedback for a message."""
        # Extract user_id to avoid accessing detached object in async context
        user_id = user.id

        # Find the message
        result = await self.db.execute(
            select(ChatMessage)
            .join(ChatSession)
            .where(
                ChatMessage.message_id == message_id,
                ChatSession.user_id == user_id,
            )
        )
        message = result.scalar_one_or_none()

        if not message:
            return False

        # Check for existing feedback
        existing = await self.db.execute(
            select(ChatFeedback).where(ChatFeedback.message_id == message.id)
        )
        if existing.scalar_one_or_none():
            # Update existing feedback
            await self.db.execute(
                ChatFeedback.__table__.update()
                .where(ChatFeedback.message_id == message.id)
                .values(feedback_type=feedback_type, comment=comment)
            )
        else:
            # Create new feedback
            feedback = ChatFeedback(
                message_id=message.id,
                feedback_type=feedback_type,
                comment=comment,
            )
            self.db.add(feedback)

        await self.db.commit()

        logger.info(
            "Feedback submitted",
            message_id=message_id,
            feedback_type=feedback_type.value,
        )

        return True

    async def clear_history(self, user: User) -> bool:
        """Clear all chat history for user."""
        # Extract user_id to avoid accessing detached object in async context
        user_id = user.id

        result = await self.db.execute(
            select(ChatSession).where(ChatSession.user_id == user_id)
        )
        sessions = result.scalars().all()

        for session in sessions:
            await self.db.delete(session)

        await self.db.commit()

        logger.info("Chat history cleared", user_id=user.id)

        return True

    async def _get_conversation_history(self, session_db_id: int) -> list[dict]:
        """Get conversation history for context."""
        # Always query fresh to avoid lazy loading issues
        result = await self.db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_db_id)
            .order_by(ChatMessage.created_at)
        )
        messages = result.scalars().all()

        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]

    async def _get_user_context(self, user_id: int) -> dict:
        """Get user context for personalized responses."""
        context = {}

        # Fetch user with relationships eagerly loaded
        try:
            result = await self.db.execute(
                select(User)
                .where(User.id == user_id)
                .options(
                    selectinload(User.investor_profile),
                    selectinload(User.portfolios)
                )
            )
            user_with_relations = result.scalar_one_or_none()

            if not user_with_relations:
                return context

            if user_with_relations.investor_profile:
                profile = user_with_relations.investor_profile
                context["risk_profile"] = profile.risk_profile
                context["investment_horizon"] = profile.investment_horizon
                context["goals"] = profile.primary_goals or []

            if user_with_relations.portfolios:
                # Use the first portfolio (primary portfolio)
                portfolio = user_with_relations.portfolios[0]
                # Load portfolio assets
                await self.db.refresh(portfolio, ["assets"])
                assets = portfolio.assets or []

                total_invested = sum(
                    float(a.quantity * a.average_price) for a in assets
                )

                context["portfolio_summary"] = {
                    "total_invested": total_invested,
                    "current_value": total_invested,  # Would need market data for real value
                    "assets_count": len(assets),
                }

        except Exception as e:
            logger.warning("Could not load user context", error=str(e))

        return context

    def _generate_session_title(self, first_message: str) -> str:
        """Generate a title from the first message."""
        # Take first 50 chars or first sentence
        title = first_message[:50]
        if len(first_message) > 50:
            title += "..."
        return title

    def _generate_suggested_questions(
        self, user_message: str, response: str
    ) -> list[SuggestedQuestion]:
        """Generate suggested follow-up questions based on context."""
        suggestions = []

        # Common follow-up questions based on topic detection
        user_lower = user_message.lower()

        if any(word in user_lower for word in ["ação", "ações", "stock", "petr", "vale", "itub"]):
            suggestions.append(
                SuggestedQuestion(
                    text="Quais são os principais indicadores fundamentalistas?",
                    category="analise"
                )
            )
            suggestions.append(
                SuggestedQuestion(
                    text="Como funciona o pagamento de dividendos?",
                    category="rendimentos"
                )
            )

        elif any(word in user_lower for word in ["fii", "fundos imobiliários", "imobiliário"]):
            suggestions.append(
                SuggestedQuestion(
                    text="Qual a diferença entre FIIs de tijolo e papel?",
                    category="fiis"
                )
            )
            suggestions.append(
                SuggestedQuestion(
                    text="Como analisar o P/VP de um FII?",
                    category="analise"
                )
            )

        elif any(word in user_lower for word in ["carteira", "portfolio", "diversific"]):
            suggestions.append(
                SuggestedQuestion(
                    text="Como calcular a alocação ideal para meu perfil?",
                    category="portfolio"
                )
            )
            suggestions.append(
                SuggestedQuestion(
                    text="Quando devo rebalancear minha carteira?",
                    category="estrategia"
                )
            )

        else:
            # Generic suggestions
            suggestions.append(
                SuggestedQuestion(
                    text="Como posso começar a investir?",
                    category="iniciante"
                )
            )
            suggestions.append(
                SuggestedQuestion(
                    text="Quais são as melhores opções para meu perfil?",
                    category="recomendacoes"
                )
            )

        return suggestions[:3]  # Return max 3 suggestions
