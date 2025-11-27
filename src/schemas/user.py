from datetime import datetime
from enum import Enum

from pydantic import BaseModel, EmailStr, Field


class SubscriptionPlan(str, Enum):
    STARTER = "starter"
    SMART = "smart"
    PRO = "pro"
    PREMIUM = "premium"


class UserBase(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=2, max_length=100)
    phone: str | None = None


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)
    cpf: str | None = Field(None, pattern=r"^\d{11}$")


class UserUpdate(BaseModel):
    full_name: str | None = Field(None, min_length=2, max_length=100)
    phone: str | None = None
    avatar_url: str | None = None


class UserResponse(UserBase):
    id: int
    is_active: bool
    is_verified: bool
    subscription_plan: SubscriptionPlan
    created_at: datetime
    updated_at: datetime | None = None

    class Config:
        from_attributes = True


class UserInDB(UserResponse):
    hashed_password: str
    cpf_encrypted: str | None = None
