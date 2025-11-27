"""WebSocket endpoints for real-time notifications."""
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, status
import structlog

from src.core.security import TokenStatus, check_token_status
from src.core.websocket import manager

router = APIRouter()
logger = structlog.get_logger()


async def get_user_from_token(token: str) -> tuple[int | None, TokenStatus]:
    """
    Validate JWT token and return user_id with status.

    Returns:
        Tuple of (user_id or None, TokenStatus)
    """
    token_status, payload = await check_token_status(token)

    if token_status != TokenStatus.VALID or payload is None:
        return None, token_status

    user_id = payload.get("sub")
    if user_id is None:
        return None, TokenStatus.INVALID

    return int(user_id), TokenStatus.VALID


@router.websocket("/notifications")
async def websocket_notifications(
    websocket: WebSocket,
    token: str = Query(...),
):
    """
    WebSocket endpoint for real-time notifications.

    Connect with: ws://host/api/v1/ws/notifications?token=<jwt_token>

    Message types received:
    - price_alert: When a price alert is triggered
    - portfolio_update: When portfolio values change
    - recommendation: New AI recommendation available
    - system: System announcements
    """
    # Validate token with detailed status
    user_id, token_status = await get_user_from_token(token)
    if user_id is None:
        # Log the specific reason for rejection
        logger.warning(
            "WebSocket connection rejected",
            token_status=token_status.value,
            endpoint="notifications",
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await manager.connect(websocket, user_id)

    try:
        # Send welcome message
        await websocket.send_json({
            "type": "system",
            "title": "Conectado",
            "message": "Voce esta conectado ao sistema de notificacoes em tempo real.",
            "data": {},
        })

        # Keep connection alive and handle incoming messages
        while True:
            # Wait for messages (ping/pong or client commands)
            data = await websocket.receive_text()

            # Handle ping (plain text)
            if data == "ping":
                await websocket.send_text("pong")
            else:
                # Handle JSON messages
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except json.JSONDecodeError:
                    pass  # Ignore invalid JSON

            # Could handle other client commands here

    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        logger.info("Client disconnected", user_id=user_id)
    except Exception as e:
        logger.error("WebSocket error", error=str(e), user_id=user_id)
        manager.disconnect(websocket, user_id)


@router.websocket("/realtime")
async def websocket_realtime(
    websocket: WebSocket,
    token: str = Query(...),
):
    """
    WebSocket endpoint for real-time market data.

    Connect with: ws://host/api/v1/ws/realtime?token=<jwt_token>

    Send messages:
    - { "action": "subscribe", "ticker": "PETR4" } - Subscribe to ticker quotes
    - { "action": "unsubscribe", "ticker": "PETR4" } - Unsubscribe from ticker

    Message types received:
    - quote: Real-time price updates for subscribed tickers
    - portfolio_update: Portfolio value changes
    - alert_triggered: Price alert notifications
    """
    # Validate token with detailed status
    user_id, token_status = await get_user_from_token(token)
    if user_id is None:
        # Log the specific reason for rejection
        logger.warning(
            "WebSocket connection rejected",
            token_status=token_status.value,
            endpoint="realtime",
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await manager.connect(websocket, user_id)
    subscribed_tickers: set[str] = set()

    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to realtime data stream",
        })

        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                action = message.get("action")
                ticker = message.get("ticker", "").upper()

                if action == "subscribe" and ticker:
                    subscribed_tickers.add(ticker)
                    await websocket.send_json({
                        "type": "subscribed",
                        "ticker": ticker,
                    })
                    logger.info("Client subscribed to ticker", user_id=user_id, ticker=ticker)

                elif action == "unsubscribe" and ticker:
                    subscribed_tickers.discard(ticker)
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "ticker": ticker,
                    })

                elif message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

            except json.JSONDecodeError:
                # Handle plain text ping
                if data == "ping":
                    await websocket.send_text("pong")

    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        logger.info("Realtime client disconnected", user_id=user_id)
    except Exception as e:
        logger.error("Realtime WebSocket error", error=str(e), user_id=user_id)
        manager.disconnect(websocket, user_id)


@router.get("/status")
async def websocket_status():
    """Get WebSocket connection status."""
    return {
        "connected_users": len(manager.get_connected_users()),
        "user_ids": manager.get_connected_users(),
    }
