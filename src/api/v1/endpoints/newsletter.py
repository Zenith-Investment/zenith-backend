"""Newsletter subscription endpoints."""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.core.deps import DbSession
from src.models.newsletter import NewsletterSubscriber

router = APIRouter()


class NewsletterSubscribeRequest(BaseModel):
    email: EmailStr


class NewsletterSubscribeResponse(BaseModel):
    success: bool
    message: str


@router.post(
    "/subscribe",
    response_model=NewsletterSubscribeResponse,
    status_code=status.HTTP_201_CREATED,
)
async def subscribe_newsletter(
    request: NewsletterSubscribeRequest,
    db: DbSession,
) -> NewsletterSubscribeResponse:
    """Subscribe an email to the newsletter."""
    # Check if already subscribed
    result = await db.execute(
        select(NewsletterSubscriber).where(NewsletterSubscriber.email == request.email)
    )
    existing = result.scalar_one_or_none()

    if existing:
        if existing.is_active:
            return NewsletterSubscribeResponse(
                success=True,
                message="Email ja esta inscrito na newsletter.",
            )
        else:
            # Reactivate subscription
            existing.is_active = True
            existing.unsubscribed_at = None
            await db.commit()
            return NewsletterSubscribeResponse(
                success=True,
                message="Inscricao reativada com sucesso!",
            )

    # Create new subscription
    try:
        subscriber = NewsletterSubscriber(email=request.email)
        db.add(subscriber)
        await db.commit()
        return NewsletterSubscribeResponse(
            success=True,
            message="Inscricao realizada com sucesso!",
        )
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Erro ao processar inscricao. Tente novamente.",
        )


@router.post("/unsubscribe", response_model=NewsletterSubscribeResponse)
async def unsubscribe_newsletter(
    request: NewsletterSubscribeRequest,
    db: DbSession,
) -> NewsletterSubscribeResponse:
    """Unsubscribe an email from the newsletter."""
    from datetime import datetime, timezone

    result = await db.execute(
        select(NewsletterSubscriber).where(NewsletterSubscriber.email == request.email)
    )
    subscriber = result.scalar_one_or_none()

    if not subscriber:
        return NewsletterSubscribeResponse(
            success=True,
            message="Email nao encontrado na lista.",
        )

    if not subscriber.is_active:
        return NewsletterSubscribeResponse(
            success=True,
            message="Email ja foi removido da lista.",
        )

    subscriber.is_active = False
    subscriber.unsubscribed_at = datetime.now(timezone.utc)
    await db.commit()

    return NewsletterSubscribeResponse(
        success=True,
        message="Inscricao cancelada com sucesso.",
    )
