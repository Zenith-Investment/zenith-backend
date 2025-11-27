"""
Broker integrations module.

Supported brokers:
- XP Investimentos (includes Rico and Clear)
- BTG Pactual
- NuInvest (Nubank)
- Banco Inter

Integration URLs:
- XP/Rico/Clear: https://developer.xpinc.com/
- BTG Pactual: https://developer.btgpactual.com/
- NuInvest: https://openfinance.dev.br/provider/Nubank
- Inter: https://developers.inter.co/
"""

from src.integrations.brokers.base import (
    BaseBrokerClient,
    BrokerAccount,
    BrokerConfig,
    BrokerPosition,
    BrokerTransaction,
    BrokerType,
)
from src.integrations.brokers.xp import XPClient, RicoClient, ClearClient
from src.integrations.brokers.btg import BTGClient
from src.integrations.brokers.nuinvest import NuInvestClient
from src.integrations.brokers.inter import InterClient

__all__ = [
    "BaseBrokerClient",
    "BrokerAccount",
    "BrokerConfig",
    "BrokerPosition",
    "BrokerTransaction",
    "BrokerType",
    "XPClient",
    "RicoClient",
    "ClearClient",
    "BTGClient",
    "NuInvestClient",
    "InterClient",
    "get_broker_client",
]


def get_broker_client(broker_type: BrokerType, config: BrokerConfig) -> BaseBrokerClient:
    """Factory function to get the appropriate broker client."""
    clients = {
        BrokerType.XP: XPClient,
        BrokerType.RICO: RicoClient,
        BrokerType.CLEAR: ClearClient,
        BrokerType.BTG: BTGClient,
        BrokerType.NUINVEST: NuInvestClient,
        BrokerType.INTER: InterClient,
    }
    client_class = clients.get(broker_type)
    if not client_class:
        raise ValueError(f"Unsupported broker type: {broker_type}")
    return client_class(config)
