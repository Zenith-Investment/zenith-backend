"""Tests for portfolio endpoints."""
import pytest
from httpx import AsyncClient

from src.models.user import User
from src.models.portfolio import Portfolio, PortfolioAsset


class TestPortfolioEndpoints:
    """Test portfolio API endpoints."""

    @pytest.mark.asyncio
    async def test_get_portfolio_empty(
        self, client: AsyncClient, test_user: User, auth_headers: dict
    ):
        """Test getting portfolio when none exists."""
        response = await client.get("/api/v1/portfolio/", headers=auth_headers)
        # Should return empty portfolio or create one
        assert response.status_code in [200, 404]

    @pytest.mark.asyncio
    async def test_get_portfolio_with_assets(
        self, client: AsyncClient, test_asset: PortfolioAsset, auth_headers: dict
    ):
        """Test getting portfolio with assets."""
        response = await client.get("/api/v1/portfolio/", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "assets" in data
        assert len(data["assets"]) >= 1

    @pytest.mark.asyncio
    async def test_add_asset_success(
        self, client: AsyncClient, test_portfolio: Portfolio, auth_headers: dict
    ):
        """Test adding a new asset to portfolio."""
        response = await client.post(
            "/api/v1/portfolio/assets",
            headers=auth_headers,
            json={
                "ticker": "ITUB4",
                "asset_class": "stocks",
                "quantity": 50,
                "average_price": 28.50,
                "broker": "Clear",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["ticker"] == "ITUB4"
        assert data["asset_class"] == "stocks"
        assert float(data["quantity"]) == 50
        assert float(data["average_price"]) == 28.50

    @pytest.mark.asyncio
    async def test_add_asset_duplicate_ticker(
        self, client: AsyncClient, test_asset: PortfolioAsset, auth_headers: dict
    ):
        """Test adding duplicate ticker fails."""
        response = await client.post(
            "/api/v1/portfolio/assets",
            headers=auth_headers,
            json={
                "ticker": "PETR4",  # Same as test_asset
                "asset_class": "stocks",
                "quantity": 10,
                "average_price": 30.00,
            },
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_add_asset_unauthorized(self, client: AsyncClient):
        """Test adding asset without auth fails."""
        response = await client.post(
            "/api/v1/portfolio/assets",
            json={
                "ticker": "VALE3",
                "asset_class": "stocks",
                "quantity": 100,
                "average_price": 65.00,
            },
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_add_asset_invalid_quantity(
        self, client: AsyncClient, test_portfolio: Portfolio, auth_headers: dict
    ):
        """Test adding asset with invalid quantity fails."""
        response = await client.post(
            "/api/v1/portfolio/assets",
            headers=auth_headers,
            json={
                "ticker": "BBDC4",
                "asset_class": "stocks",
                "quantity": -10,  # Invalid
                "average_price": 15.00,
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_remove_asset_success(
        self, client: AsyncClient, test_asset: PortfolioAsset, auth_headers: dict
    ):
        """Test removing an asset from portfolio."""
        response = await client.delete(
            f"/api/v1/portfolio/assets/{test_asset.id}",
            headers=auth_headers,
        )
        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_remove_asset_not_found(
        self, client: AsyncClient, test_portfolio: Portfolio, auth_headers: dict
    ):
        """Test removing nonexistent asset fails."""
        response = await client.delete(
            "/api/v1/portfolio/assets/99999",
            headers=auth_headers,
        )
        assert response.status_code == 404


class TestPortfolioAllocation:
    """Test portfolio allocation features."""

    @pytest.mark.asyncio
    async def test_allocation_by_class(
        self, client: AsyncClient, test_asset: PortfolioAsset, auth_headers: dict
    ):
        """Test getting allocation by asset class."""
        response = await client.get("/api/v1/portfolio/", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "allocation_by_class" in data
        # Should have at least stocks allocation
        allocations = data["allocation_by_class"]
        assert len(allocations) >= 1
        assert any(a["asset_class"] == "stocks" for a in allocations)
