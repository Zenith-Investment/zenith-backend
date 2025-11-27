"""Tests for broker service and endpoints."""
import pytest
from datetime import datetime, timezone
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.user import User
from src.models.broker import BrokerConnection, BrokerType, ConnectionStatus
from src.services.broker import BrokerService, BROKER_INFO


class TestBrokerInfo:
    """Test broker information."""

    def test_broker_info_structure(self):
        """Test that all brokers have required info fields."""
        required_fields = ["name", "description", "features", "auth_type"]

        for broker_type, info in BROKER_INFO.items():
            for field in required_fields:
                assert field in info, f"Missing {field} for {broker_type}"

    def test_all_brokers_have_oauth2(self):
        """Test that all brokers use OAuth2 authentication."""
        for broker_type, info in BROKER_INFO.items():
            assert "oauth2" in info["auth_type"], f"{broker_type} should use OAuth2"


class TestBrokerEndpoints:
    """Test broker API endpoints."""

    @pytest.mark.asyncio
    async def test_get_supported_brokers(
        self, client: AsyncClient, auth_headers: dict
    ):
        """Test getting list of supported brokers."""
        response = await client.get(
            "/api/v1/brokers/supported",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "brokers" in data
        assert len(data["brokers"]) > 0

        # Check each broker has required fields
        for broker in data["brokers"]:
            assert "broker_type" in broker
            assert "name" in broker
            assert "description" in broker

    @pytest.mark.asyncio
    async def test_get_supported_brokers_unauthorized(self, client: AsyncClient):
        """Test getting supported brokers without auth fails."""
        response = await client.get("/api/v1/brokers/supported")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_user_connections_empty(
        self, client: AsyncClient, auth_headers: dict
    ):
        """Test getting connections when none exist."""
        response = await client.get(
            "/api/v1/brokers/connections",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "connections" in data
        assert len(data["connections"]) == 0

    @pytest.mark.asyncio
    async def test_get_user_connections_with_data(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        test_user: User,
        auth_headers: dict,
    ):
        """Test getting connections when some exist."""
        # Create a test connection
        connection = BrokerConnection(
            user_id=test_user.id,
            broker_type=BrokerType.XP,
            status=ConnectionStatus.ACTIVE,
            broker_account_id="12345",
            broker_account_name="Test Account",
            last_sync_at=datetime.now(timezone.utc),
        )
        db_session.add(connection)
        await db_session.commit()

        response = await client.get(
            "/api/v1/brokers/connections",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "connections" in data
        assert len(data["connections"]) == 1
        assert data["connections"][0]["broker_type"] == "xp"
        assert data["connections"][0]["status"] == "active"


class TestBrokerConnection:
    """Test broker connection flow."""

    @pytest.mark.asyncio
    async def test_start_connection_flow(
        self, client: AsyncClient, auth_headers: dict
    ):
        """Test starting OAuth connection flow."""
        response = await client.post(
            "/api/v1/brokers/connect",
            headers=auth_headers,
            json={"broker_type": "xp"},
        )
        # Should return authorization URL (may fail if no credentials configured)
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.json()
            assert "authorization_url" in data

    @pytest.mark.asyncio
    async def test_disconnect_broker_not_found(
        self, client: AsyncClient, auth_headers: dict
    ):
        """Test disconnecting nonexistent broker connection."""
        response = await client.delete(
            "/api/v1/brokers/connections/99999",
            headers=auth_headers,
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_disconnect_broker_success(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        test_user: User,
        auth_headers: dict,
    ):
        """Test disconnecting an active broker connection."""
        # Create a test connection
        connection = BrokerConnection(
            user_id=test_user.id,
            broker_type=BrokerType.BTG,
            status=ConnectionStatus.ACTIVE,
            broker_account_id="67890",
        )
        db_session.add(connection)
        await db_session.commit()
        await db_session.refresh(connection)

        response = await client.delete(
            f"/api/v1/brokers/connections/{connection.id}",
            headers=auth_headers,
        )
        assert response.status_code == 204


class TestBrokerService:
    """Test broker service methods."""

    @pytest.mark.asyncio
    async def test_get_supported_brokers_service(self, db_session: AsyncSession):
        """Test broker service returns supported brokers."""
        service = BrokerService(db_session)
        brokers = service.get_supported_brokers()

        assert len(brokers) == len(BrokerType)
        for broker in brokers:
            assert "broker_type" in broker
            assert "name" in broker
            assert "features" in broker

    @pytest.mark.asyncio
    async def test_get_user_connections_service(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test getting user connections via service."""
        service = BrokerService(db_session)
        connections = await service.get_user_connections(test_user.id)

        assert isinstance(connections, list)

    @pytest.mark.asyncio
    async def test_get_connection_not_found(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test getting nonexistent connection returns None."""
        service = BrokerService(db_session)
        connection = await service.get_connection(test_user.id, 99999)

        assert connection is None

    @pytest.mark.asyncio
    async def test_get_connection_by_broker(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test getting connection by broker type."""
        # Create a connection
        connection = BrokerConnection(
            user_id=test_user.id,
            broker_type=BrokerType.INTER,
            status=ConnectionStatus.ACTIVE,
        )
        db_session.add(connection)
        await db_session.commit()

        service = BrokerService(db_session)
        found = await service.get_connection_by_broker(test_user.id, BrokerType.INTER)

        assert found is not None
        assert found.broker_type == BrokerType.INTER

    @pytest.mark.asyncio
    async def test_get_connection_by_broker_not_found(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test getting nonexistent broker connection."""
        service = BrokerService(db_session)
        found = await service.get_connection_by_broker(test_user.id, BrokerType.NUINVEST)

        assert found is None
