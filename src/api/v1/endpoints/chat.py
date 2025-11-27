"""Chat endpoints for AI assistant."""
import json
import uuid

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
import structlog

from src.core.deps import CurrentUser, DbSession
from src.schemas.chat import (
    ChatFeedbackRequest,
    ChatHistoryResponse,
    ChatMessageRequest,
    ChatMessageResponse,
)
from src.services.chat import ChatService

router = APIRouter()
logger = structlog.get_logger()


@router.post("/message", response_model=ChatMessageResponse)
async def send_message(
    request: ChatMessageRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> ChatMessageResponse:
    """Send a message to the AI assistant and receive a response."""
    chat_service = ChatService(db)

    try:
        response = await chat_service.send_message(
            user=current_user,
            message=request.message,
            session_id=request.session_id,
        )

        logger.info(
            "Chat message sent",
            user_id=current_user.id,
            session_id=response.session_id,
        )

        return response

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            "Failed to process chat message",
            user_id=current_user.id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Falha ao processar mensagem. Verifique se a API de IA está configurada.",
        )


@router.post("/message/stream")
async def send_message_stream(
    request: ChatMessageRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> StreamingResponse:
    """Send a message and receive streaming response via Server-Sent Events."""
    from src.ai.llm.client import llm_client

    chat_service = ChatService(db)

    # ALL database operations MUST happen here, BEFORE returning StreamingResponse
    try:
        session_db_id, preferred_provider, user_context, history = await chat_service.prepare_stream(
            user=current_user,
            message=request.message,
            session_id=request.session_id,
        )
    except Exception as e:
        logger.error("Failed to prepare stream", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha ao preparar chat: {str(e)}",
        )

    # Check if LLM is available
    try:
        provider_info = llm_client.get_provider_info()
        has_provider = any(
            p["enabled"] for p in provider_info["available_providers"].values()
        )
        if not has_provider:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Nenhum provedor de IA configurado. Configure DEEPSEEK_API_KEY, GROQ_API_KEY, ou inicie o Ollama localmente.",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Could not check LLM availability", error=str(e))

    # Store values for the generator (no DB operations in generator!)
    message = request.message
    user_id = current_user.id

    # Pre-generate message_id so we can send it to frontend in [DONE] event
    assistant_message_id = str(uuid.uuid4())

    async def generate():
        full_response = ""
        try:
            async for chunk in llm_client.chat_stream(
                message=message,
                history=history,
                user_context=user_context,
                preferred_provider=preferred_provider,
            ):
                full_response += chunk
                # Format as SSE
                data = json.dumps({"content": chunk})
                yield f"data: {data}\n\n"

            # Send done event with message_id for feedback feature
            done_data = json.dumps({"message_id": assistant_message_id})
            yield f"data: {done_data}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(
                "Stream error",
                user_id=user_id,
                error=str(e),
            )
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"
        finally:
            # Save response with a NEW database session (not the request session)
            if full_response:
                try:
                    await chat_service.save_stream_response(
                        session_db_id, full_response, message, assistant_message_id
                    )
                except Exception as e:
                    logger.error("Failed to save stream response", error=str(e))

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    current_user: CurrentUser,
    db: DbSession,
    session_id: str | None = None,
) -> ChatHistoryResponse:
    """Get chat history for current user.

    Args:
        session_id: Optional session ID to get messages for a specific session.
            If not provided, returns all sessions without messages.
    """
    chat_service = ChatService(db)

    try:
        history = await chat_service.get_history(current_user, session_id)
        return history

    except Exception as e:
        logger.error(
            "Failed to get chat history",
            user_id=current_user.id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Falha ao recuperar histórico de chat.",
        )


@router.post("/feedback")
async def submit_feedback(
    request: ChatFeedbackRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Submit feedback for a chat message (like/dislike)."""
    chat_service = ChatService(db)

    try:
        success = await chat_service.submit_feedback(
            user=current_user,
            message_id=request.message_id,
            feedback_type=request.feedback_type,
            comment=request.comment,
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Mensagem não encontrada.",
            )

        return {"status": "success", "message": "Feedback registrado com sucesso."}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to submit feedback",
            user_id=current_user.id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Falha ao registrar feedback.",
        )


@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
async def clear_chat_history(
    current_user: CurrentUser,
    db: DbSession,
) -> None:
    """Clear all chat history for current user."""
    chat_service = ChatService(db)

    try:
        await chat_service.clear_history(current_user)

        logger.info("Chat history cleared", user_id=current_user.id)

    except Exception as e:
        logger.error(
            "Failed to clear chat history",
            user_id=current_user.id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Falha ao limpar histórico de chat.",
        )
