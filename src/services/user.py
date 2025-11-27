"""User service."""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.user import User
from src.models.portfolio import Portfolio
from src.schemas.user import UserUpdate


class UserService:
    """Service for user operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_by_id(self, user_id: int) -> User | None:
        """Get user by ID."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def update_user(self, user: User, update_data: UserUpdate) -> User:
        """Update user profile."""
        update_dict = update_data.model_dump(exclude_unset=True)

        for field, value in update_dict.items():
            setattr(user, field, value)

        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def delete_user(self, user: User) -> None:
        """Delete user account (LGPD compliance)."""
        await self.db.delete(user)
        await self.db.commit()

    async def create_default_portfolio(self, user: User) -> Portfolio:
        """Create default portfolio for new user."""
        portfolio = Portfolio(
            user_id=user.id,
            name="Minha Carteira",
        )
        self.db.add(portfolio)
        await self.db.commit()
        await self.db.refresh(portfolio)
        return portfolio

    async def get_user_with_profile(self, user_id: int) -> User | None:
        """Get user with investor profile loaded."""
        result = await self.db.execute(
            select(User)
            .where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        if user:
            # Eager load investor profile
            await self.db.refresh(user, ["investor_profile"])
        return user
