from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class FeedbackType(str, Enum):
    LIKE = "like"
    DISLIKE = "dislike"


class ChatMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None


class ChatMessage(BaseModel):
    id: str
    role: MessageRole
    content: str
    created_at: datetime


class SuggestedQuestion(BaseModel):
    text: str
    category: str | None = None


class ChatMessageResponse(BaseModel):
    id: str
    session_id: str
    message: ChatMessage
    suggested_questions: list[SuggestedQuestion] = []


class ChatSession(BaseModel):
    id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    message_count: int


class ChatHistoryResponse(BaseModel):
    sessions: list[ChatSession]
    messages: list[ChatMessage] = []


class ChatFeedbackRequest(BaseModel):
    message_id: str
    feedback_type: FeedbackType
    comment: str | None = Field(None, max_length=500)
