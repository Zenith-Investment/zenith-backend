"""
MCP (Model Context Protocol) integration for InvestAI.

Provides tool calling capabilities for LLMs to access:
- Real-time market data
- Financial analysis tools
- Portfolio management
- Technical indicators
"""
from src.mcp.client import MCPClient, mcp_client
from src.mcp.tools import ToolRegistry, tool_registry

__all__ = ["MCPClient", "mcp_client", "ToolRegistry", "tool_registry"]
