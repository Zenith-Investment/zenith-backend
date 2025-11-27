"""Price alert schemas."""
from datetime import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field


class AlertCondition(str, Enum):
    ABOVE = "above"
    BELOW = "below"


class PriceAlertCreate(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20)
    target_price: Decimal = Field(..., gt=0)
    condition: AlertCondition
    notes: str | None = None


class PriceAlertUpdate(BaseModel):
    target_price: Decimal | None = Field(None, gt=0)
    condition: AlertCondition | None = None
    is_active: bool | None = None
    notes: str | None = None


class PriceAlertResponse(BaseModel):
    id: int
    ticker: str
    target_price: Decimal
    condition: AlertCondition
    is_active: bool
    is_triggered: bool
    triggered_at: datetime | None = None
    triggered_price: Decimal | None = None
    notes: str | None = None
    created_at: datetime
    updated_at: datetime | None = None

    class Config:
        from_attributes = True


class PriceAlertListResponse(BaseModel):
    alerts: list[PriceAlertResponse]
    total: int
    page: int
    page_size: int
