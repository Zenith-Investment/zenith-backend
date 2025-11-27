"""WebSocket connection manager for real-time notifications."""
from typing import Dict, List, Any
import json
import structlog
from fastapi import WebSocket

logger = structlog.get_logger()


class ConnectionManager:
    """Manages WebSocket connections for users."""

    def __init__(self):
        # Map user_id to list of active connections
        self.active_connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
        logger.info("WebSocket connected", user_id=user_id)

    def disconnect(self, websocket: WebSocket, user_id: int):
        """Remove a WebSocket connection."""
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        logger.info("WebSocket disconnected", user_id=user_id)

    async def send_personal_message(self, message: dict, user_id: int):
        """Send a message to a specific user's connections."""
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error("Error sending WebSocket message", error=str(e))

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected users."""
        for user_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error("Error broadcasting message", error=str(e))

    def get_connected_users(self) -> List[int]:
        """Get list of connected user IDs."""
        return list(self.active_connections.keys())

    def is_user_connected(self, user_id: int) -> bool:
        """Check if a user has active connections."""
        return user_id in self.active_connections and len(self.active_connections[user_id]) > 0


# Global connection manager instance
manager = ConnectionManager()


# Notification types
class NotificationType:
    PRICE_ALERT = "price_alert"
    PORTFOLIO_UPDATE = "portfolio_update"
    RECOMMENDATION = "recommendation"
    SYSTEM = "system"


async def send_notification(
    user_id: int,
    notification_type: str,
    title: str,
    message: str,
    data: dict | None = None,
):
    """Helper function to send a notification to a user."""
    notification = {
        "type": notification_type,
        "title": title,
        "message": message,
        "data": data or {},
    }
    await manager.send_personal_message(notification, user_id)
    logger.info(
        "Notification sent",
        user_id=user_id,
        type=notification_type,
        title=title,
    )
