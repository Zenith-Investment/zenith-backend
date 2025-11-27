"""Pytest configuration and fixtures."""
import asyncio
from collections.abc import AsyncGenerator, Generator
from datetime import datetime
from decimal import Decimal
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from src.core.database import Base, get_db
from src.core.security import get_password_hash, create_access_token
from src.main import app
from src.models.user import User
from src.models.portfolio import Portfolio, PortfolioAsset
from src.models.alert import PriceAlert, AlertCondition
from src.models.broker import BrokerConnection, BrokerType, ConnectionStatus
from src.models.market import PriceHistory, PortfolioSnapshot
from src.schemas.user import SubscriptionPlan
from src.schemas.portfolio import AssetClass


# Test database URL - using SQLite for speed
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def async_engine():
    """Create async test engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session_maker = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_maker() as session:
        yield session


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with overridden database dependency."""

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create a test user."""
    user = User(
        email="test@example.com",
        hashed_password=get_password_hash("testpassword123"),
        full_name="Test User",
        is_active=True,
        is_verified=True,
        subscription_plan=SubscriptionPlan.STARTER,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def test_user_token(test_user: User) -> str:
    """Create access token for test user."""
    return create_access_token(subject=str(test_user.id))


@pytest_asyncio.fixture
async def auth_headers(test_user_token: str) -> dict[str, str]:
    """Create authorization headers."""
    return {"Authorization": f"Bearer {test_user_token}"}


@pytest_asyncio.fixture
async def test_portfolio(db_session: AsyncSession, test_user: User) -> Portfolio:
    """Create a test portfolio."""
    portfolio = Portfolio(
        user_id=test_user.id,
        name="Test Portfolio",
    )
    db_session.add(portfolio)
    await db_session.commit()
    await db_session.refresh(portfolio)
    return portfolio


@pytest_asyncio.fixture
async def test_asset(
    db_session: AsyncSession, test_portfolio: Portfolio
) -> PortfolioAsset:
    """Create a test portfolio asset."""
    asset = PortfolioAsset(
        portfolio_id=test_portfolio.id,
        ticker="PETR4",
        name="Petrobras PN",
        asset_class=AssetClass.STOCKS,
        quantity=Decimal("100"),
        average_price=Decimal("35.50"),
        broker="XP",
    )
    db_session.add(asset)
    await db_session.commit()
    await db_session.refresh(asset)
    return asset


@pytest_asyncio.fixture
async def test_alert(db_session: AsyncSession, test_user: User) -> PriceAlert:
    """Create a test price alert."""
    alert = PriceAlert(
        user_id=test_user.id,
        ticker="PETR4",
        target_price=Decimal("40.00"),
        condition=AlertCondition.ABOVE,
        is_active=True,
        notes="Test alert",
    )
    db_session.add(alert)
    await db_session.commit()
    await db_session.refresh(alert)
    return alert
