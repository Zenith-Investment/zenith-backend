"""Tests for price alerts endpoints."""
import pytest
from httpx import AsyncClient

from src.models.user import User
from src.models.alert import PriceAlert


class TestAlertsEndpoints:
    """Test price alerts API endpoints."""

    @pytest.mark.asyncio
    async def test_list_alerts_empty(
        self, client: AsyncClient, test_user: User, auth_headers: dict
    ):
        """Test listing alerts when none exist."""
        response = await client.get("/api/v1/alerts/", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["alerts"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_list_alerts_with_data(
        self, client: AsyncClient, test_alert: PriceAlert, auth_headers: dict
    ):
        """Test listing alerts with existing data."""
        response = await client.get("/api/v1/alerts/", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data["alerts"]) == 1
        assert data["total"] == 1
        assert data["alerts"][0]["ticker"] == "PETR4"
        assert float(data["alerts"][0]["target_price"]) == 40.00
        assert data["alerts"][0]["condition"] == "above"

    @pytest.mark.asyncio
    async def test_create_alert_success(
        self, client: AsyncClient, test_user: User, auth_headers: dict
    ):
        """Test creating a new price alert."""
        response = await client.post(
            "/api/v1/alerts/",
            headers=auth_headers,
            json={
                "ticker": "VALE3",
                "target_price": 70.00,
                "condition": "below",
                "notes": "Entry point",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["ticker"] == "VALE3"
        assert float(data["target_price"]) == 70.00
        assert data["condition"] == "below"
        assert data["is_active"] is True
        assert data["is_triggered"] is False
        assert data["notes"] == "Entry point"

    @pytest.mark.asyncio
    async def test_create_alert_duplicate(
        self, client: AsyncClient, test_alert: PriceAlert, auth_headers: dict
    ):
        """Test creating duplicate alert fails."""
        response = await client.post(
            "/api/v1/alerts/",
            headers=auth_headers,
            json={
                "ticker": "PETR4",
                "target_price": 45.00,
                "condition": "above",
            },
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_create_alert_unauthorized(self, client: AsyncClient):
        """Test creating alert without auth fails."""
        response = await client.post(
            "/api/v1/alerts/",
            json={
                "ticker": "VALE3",
                "target_price": 70.00,
                "condition": "above",
            },
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_update_alert_success(
        self, client: AsyncClient, test_alert: PriceAlert, auth_headers: dict
    ):
        """Test updating a price alert."""
        response = await client.patch(
            f"/api/v1/alerts/{test_alert.id}",
            headers=auth_headers,
            json={
                "target_price": 42.00,
                "is_active": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert float(data["target_price"]) == 42.00
        assert data["is_active"] is False

    @pytest.mark.asyncio
    async def test_update_alert_not_found(
        self, client: AsyncClient, test_user: User, auth_headers: dict
    ):
        """Test updating nonexistent alert fails."""
        response = await client.patch(
            "/api/v1/alerts/99999",
            headers=auth_headers,
            json={"target_price": 50.00},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_alert_success(
        self, client: AsyncClient, test_alert: PriceAlert, auth_headers: dict
    ):
        """Test deleting a price alert."""
        response = await client.delete(
            f"/api/v1/alerts/{test_alert.id}",
            headers=auth_headers,
        )
        assert response.status_code == 204

        # Verify it's deleted
        response = await client.get("/api/v1/alerts/", headers=auth_headers)
        assert response.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_delete_alert_not_found(
        self, client: AsyncClient, test_user: User, auth_headers: dict
    ):
        """Test deleting nonexistent alert fails."""
        response = await client.delete(
            "/api/v1/alerts/99999",
            headers=auth_headers,
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_filter_alerts_by_active(
        self, client: AsyncClient, test_alert: PriceAlert, auth_headers: dict
    ):
        """Test filtering alerts by active status."""
        response = await client.get(
            "/api/v1/alerts/?is_active=true",
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert len(response.json()["alerts"]) == 1

        response = await client.get(
            "/api/v1/alerts/?is_active=false",
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert len(response.json()["alerts"]) == 0
