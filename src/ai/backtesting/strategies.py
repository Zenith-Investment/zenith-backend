"""Pre-built backtesting strategies."""
from abc import ABC, abstractmethod
from datetime import date
from decimal import Decimal
from typing import Optional


class BaseStrategy(ABC):
    """Base class for backtesting strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(
        self,
        date: date,
        prices: dict[str, Decimal],
        positions: dict,
        cash: Decimal,
        portfolio_value: Decimal,
    ) -> list[dict]:
        """
        Generate trading signals for the current date.

        Returns list of signals:
        [{"ticker": "PETR4", "action": "buy", "amount": Decimal("1000")}]
        """
        pass


class BuyAndHoldStrategy(BaseStrategy):
    """Simple buy and hold strategy."""

    def __init__(
        self,
        tickers: list[str],
        allocation: Optional[dict[str, float]] = None,
    ):
        super().__init__("Buy and Hold")
        self.tickers = tickers
        self.allocation = allocation or {t: 1.0 / len(tickers) for t in tickers}
        self.initialized = False

    def generate_signals(
        self,
        date: date,
        prices: dict[str, Decimal],
        positions: dict,
        cash: Decimal,
        portfolio_value: Decimal,
    ) -> list[dict]:
        if self.initialized:
            return []

        signals = []
        for ticker in self.tickers:
            if ticker in prices:
                alloc = self.allocation.get(ticker, 0)
                amount = portfolio_value * Decimal(str(alloc)) * Decimal("0.99")  # 1% buffer
                signals.append({
                    "ticker": ticker,
                    "action": "buy",
                    "amount": amount,
                })

        self.initialized = True
        return signals


class RebalancingStrategy(BaseStrategy):
    """Strategy that rebalances portfolio periodically."""

    def __init__(
        self,
        target_allocation: dict[str, float],
        rebalance_threshold: float = 0.05,  # 5% deviation triggers rebalance
        rebalance_frequency: int = 30,  # days between rebalances
    ):
        super().__init__("Rebalancing")
        self.target_allocation = target_allocation
        self.rebalance_threshold = rebalance_threshold
        self.rebalance_frequency = rebalance_frequency
        self.last_rebalance: Optional[date] = None
        self.initialized = False

    def generate_signals(
        self,
        date: date,
        prices: dict[str, Decimal],
        positions: dict,
        cash: Decimal,
        portfolio_value: Decimal,
    ) -> list[dict]:
        signals = []

        # Initial allocation
        if not self.initialized:
            for ticker, target_pct in self.target_allocation.items():
                if ticker in prices:
                    amount = portfolio_value * Decimal(str(target_pct)) * Decimal("0.99")
                    signals.append({
                        "ticker": ticker,
                        "action": "buy",
                        "amount": amount,
                    })
            self.initialized = True
            self.last_rebalance = date
            return signals

        # Check if it's time to rebalance
        if self.last_rebalance:
            days_since = (date - self.last_rebalance).days
            if days_since < self.rebalance_frequency:
                return []

        # Calculate current allocation
        current_allocation = {}
        for ticker, pos in positions.items():
            if portfolio_value > 0:
                current_allocation[ticker] = float(pos.current_value / portfolio_value)

        # Check for deviations and generate signals
        for ticker, target_pct in self.target_allocation.items():
            current_pct = current_allocation.get(ticker, 0)
            deviation = target_pct - current_pct

            if abs(deviation) >= self.rebalance_threshold:
                if deviation > 0:
                    # Need to buy more
                    amount = portfolio_value * Decimal(str(deviation))
                    if amount <= cash:
                        signals.append({
                            "ticker": ticker,
                            "action": "buy",
                            "amount": amount,
                        })
                else:
                    # Need to sell
                    amount = portfolio_value * Decimal(str(abs(deviation)))
                    signals.append({
                        "ticker": ticker,
                        "action": "sell",
                        "amount": amount,
                    })

        if signals:
            self.last_rebalance = date

        return signals


class MomentumStrategy(BaseStrategy):
    """Simple momentum strategy based on price trend."""

    def __init__(
        self,
        tickers: list[str],
        lookback_days: int = 20,
        momentum_threshold: float = 0.05,  # 5% gain to trigger buy
    ):
        super().__init__("Momentum")
        self.tickers = tickers
        self.lookback_days = lookback_days
        self.momentum_threshold = momentum_threshold
        self.price_history: dict[str, list[Decimal]] = {t: [] for t in tickers}

    def generate_signals(
        self,
        date: date,
        prices: dict[str, Decimal],
        positions: dict,
        cash: Decimal,
        portfolio_value: Decimal,
    ) -> list[dict]:
        signals = []

        for ticker in self.tickers:
            if ticker not in prices:
                continue

            current_price = prices[ticker]

            # Update price history
            self.price_history[ticker].append(current_price)

            # Keep only lookback period
            if len(self.price_history[ticker]) > self.lookback_days:
                self.price_history[ticker] = self.price_history[ticker][-self.lookback_days:]

            # Need enough history
            if len(self.price_history[ticker]) < self.lookback_days:
                continue

            # Calculate momentum
            start_price = self.price_history[ticker][0]
            momentum = float((current_price - start_price) / start_price)

            # Generate signals
            if momentum > self.momentum_threshold and ticker not in positions:
                # Positive momentum - buy
                amount = portfolio_value * Decimal("0.1")  # 10% per position
                if amount <= cash:
                    signals.append({
                        "ticker": ticker,
                        "action": "buy",
                        "amount": amount,
                    })
            elif momentum < -self.momentum_threshold and ticker in positions:
                # Negative momentum - sell
                signals.append({
                    "ticker": ticker,
                    "action": "sell",
                    "amount": positions[ticker].current_value,
                })

        return signals


class DCAStrategy(BaseStrategy):
    """Dollar Cost Averaging strategy - invest fixed amount periodically."""

    def __init__(
        self,
        tickers: list[str],
        investment_amount: Decimal = Decimal("1000"),
        investment_frequency: int = 30,  # days between investments
        allocation: Optional[dict[str, float]] = None,
    ):
        super().__init__("DCA (Custo Médio)")
        self.tickers = tickers
        self.investment_amount = investment_amount
        self.investment_frequency = investment_frequency
        self.allocation = allocation or {t: 1.0 / len(tickers) for t in tickers}
        self.last_investment: Optional[date] = None

    def generate_signals(
        self,
        date: date,
        prices: dict[str, Decimal],
        positions: dict,
        cash: Decimal,
        portfolio_value: Decimal,
    ) -> list[dict]:
        # Check if it's time to invest
        if self.last_investment:
            days_since = (date - self.last_investment).days
            if days_since < self.investment_frequency:
                return []

        signals = []
        total_to_invest = min(self.investment_amount, cash * Decimal("0.9"))

        for ticker in self.tickers:
            if ticker in prices:
                alloc = self.allocation.get(ticker, 0)
                amount = total_to_invest * Decimal(str(alloc))
                if amount > 0:
                    signals.append({
                        "ticker": ticker,
                        "action": "buy",
                        "amount": amount,
                    })

        if signals:
            self.last_investment = date

        return signals


class ValueStrategy(BaseStrategy):
    """Value investing strategy based on P/L ratio."""

    def __init__(
        self,
        universe: dict[str, dict],  # ticker -> {"pl": float, "dy": float}
        max_pl: float = 15.0,
        min_dy: float = 0.03,
    ):
        super().__init__("Value Investing")
        self.universe = universe
        self.max_pl = max_pl
        self.min_dy = min_dy

    def generate_signals(
        self,
        date: date,
        prices: dict[str, Decimal],
        positions: dict,
        cash: Decimal,
        portfolio_value: Decimal,
    ) -> list[dict]:
        signals = []

        # Find undervalued stocks
        for ticker, metrics in self.universe.items():
            if ticker not in prices:
                continue

            pl = metrics.get("pl", 999)
            dy = metrics.get("dy", 0)

            is_undervalued = pl < self.max_pl and dy > self.min_dy

            if is_undervalued and ticker not in positions:
                amount = portfolio_value * Decimal("0.1")
                if amount <= cash:
                    signals.append({
                        "ticker": ticker,
                        "action": "buy",
                        "amount": amount,
                    })
            elif not is_undervalued and ticker in positions:
                # No longer undervalued - sell
                signals.append({
                    "ticker": ticker,
                    "action": "sell",
                    "amount": positions[ticker].current_value,
                })

        return signals
