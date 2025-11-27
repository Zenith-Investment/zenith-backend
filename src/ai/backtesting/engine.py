"""Backtesting engine for strategy simulation."""
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from typing import Optional
import statistics

import structlog
import pandas as pd
import numpy as np

logger = structlog.get_logger()


@dataclass
class Trade:
    """Represents a single trade."""
    date: date
    ticker: str
    action: str  # "buy" or "sell"
    quantity: Decimal
    price: Decimal
    value: Decimal
    fees: Decimal = Decimal("0")


@dataclass
class Position:
    """Represents a portfolio position."""
    ticker: str
    quantity: Decimal
    average_price: Decimal
    current_price: Decimal = Decimal("0")

    @property
    def current_value(self) -> Decimal:
        return self.quantity * self.current_price

    @property
    def profit_loss(self) -> Decimal:
        return self.current_value - (self.quantity * self.average_price)

    @property
    def profit_loss_pct(self) -> float:
        if self.average_price == 0:
            return 0.0
        return float((self.current_price - self.average_price) / self.average_price * 100)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy_name: str
    start_date: date
    end_date: date
    initial_capital: Decimal
    final_value: Decimal
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    trades: list[Trade]
    daily_values: list[dict]
    positions: list[Position]
    benchmark_return: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": float(self.initial_capital),
            "final_value": float(self.final_value),
            "total_return": round(self.total_return, 2),
            "annualized_return": round(self.annualized_return, 2),
            "volatility": round(self.volatility, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "win_rate": round(self.win_rate, 2),
            "total_trades": self.total_trades,
            "benchmark_return": round(self.benchmark_return, 2) if self.benchmark_return else None,
            "alpha": round(self.alpha, 2) if self.alpha else None,
            "beta": round(self.beta, 2) if self.beta else None,
            "disclaimer": "Resultados passados não garantem resultados futuros. "
                         "Este backtest é apenas para fins educacionais.",
        }


class BacktestEngine:
    """Engine for running strategy backtests."""

    RISK_FREE_RATE = 0.1075  # ~Selic annual rate

    def __init__(
        self,
        initial_capital: Decimal = Decimal("100000"),
        fee_per_trade: Decimal = Decimal("0"),  # Many brokers have zero fee now
        slippage: float = 0.001,  # 0.1% slippage
    ):
        self.initial_capital = initial_capital
        self.fee_per_trade = fee_per_trade
        self.slippage = slippage

    async def run(
        self,
        strategy,
        price_data: dict[str, pd.DataFrame],
        start_date: date,
        end_date: date,
        benchmark_ticker: Optional[str] = "^BVSP",
    ) -> BacktestResult:
        """
        Run a backtest for a strategy.

        Args:
            strategy: Strategy instance to test
            price_data: Dict of ticker -> DataFrame with OHLCV data
            start_date: Start date for backtest
            end_date: End date for backtest
            benchmark_ticker: Ticker to compare against (default: Ibovespa)
        """
        # Initialize state
        cash = self.initial_capital
        positions: dict[str, Position] = {}
        trades: list[Trade] = []
        daily_values: list[dict] = []

        # Get all trading dates
        all_dates = set()
        for df in price_data.values():
            all_dates.update(df.index.date)

        trading_dates = sorted([d for d in all_dates if start_date <= d <= end_date])

        if not trading_dates:
            raise ValueError("No trading dates in the specified range")

        # Run simulation
        for current_date in trading_dates:
            # Get current prices
            current_prices = {}
            for ticker, df in price_data.items():
                if current_date in df.index.date:
                    idx = df.index.date == current_date
                    current_prices[ticker] = Decimal(str(df.loc[idx, "close"].iloc[0]))

            # Update position prices
            for ticker, pos in positions.items():
                if ticker in current_prices:
                    pos.current_price = current_prices[ticker]

            # Calculate current portfolio value
            portfolio_value = cash + sum(p.current_value for p in positions.values())

            # Get strategy signals
            signals = strategy.generate_signals(
                date=current_date,
                prices=current_prices,
                positions=positions,
                cash=cash,
                portfolio_value=portfolio_value,
            )

            # Execute trades
            for signal in signals:
                ticker = signal["ticker"]
                action = signal["action"]
                amount = signal.get("amount", portfolio_value * Decimal("0.1"))

                if ticker not in current_prices:
                    continue

                price = current_prices[ticker]

                # Apply slippage
                if action == "buy":
                    execution_price = price * Decimal(str(1 + self.slippage))
                    quantity = (amount - self.fee_per_trade) / execution_price

                    if quantity * execution_price + self.fee_per_trade <= cash:
                        # Execute buy
                        trade_value = quantity * execution_price

                        if ticker in positions:
                            # Average up
                            old_pos = positions[ticker]
                            total_qty = old_pos.quantity + quantity
                            avg_price = (
                                (old_pos.quantity * old_pos.average_price) + trade_value
                            ) / total_qty
                            positions[ticker] = Position(
                                ticker=ticker,
                                quantity=total_qty,
                                average_price=avg_price,
                                current_price=price,
                            )
                        else:
                            positions[ticker] = Position(
                                ticker=ticker,
                                quantity=quantity,
                                average_price=execution_price,
                                current_price=price,
                            )

                        cash -= trade_value + self.fee_per_trade

                        trades.append(Trade(
                            date=current_date,
                            ticker=ticker,
                            action="buy",
                            quantity=quantity,
                            price=execution_price,
                            value=trade_value,
                            fees=self.fee_per_trade,
                        ))

                elif action == "sell":
                    execution_price = price * Decimal(str(1 - self.slippage))

                    if ticker in positions:
                        pos = positions[ticker]
                        sell_qty = min(pos.quantity, amount / execution_price)
                        trade_value = sell_qty * execution_price

                        if sell_qty >= pos.quantity:
                            # Close position
                            del positions[ticker]
                        else:
                            # Partial sell
                            positions[ticker].quantity -= sell_qty

                        cash += trade_value - self.fee_per_trade

                        trades.append(Trade(
                            date=current_date,
                            ticker=ticker,
                            action="sell",
                            quantity=sell_qty,
                            price=execution_price,
                            value=trade_value,
                            fees=self.fee_per_trade,
                        ))

            # Record daily value
            final_portfolio_value = cash + sum(p.current_value for p in positions.values())
            daily_values.append({
                "date": current_date.isoformat(),
                "value": float(final_portfolio_value),
                "cash": float(cash),
                "invested": float(sum(p.current_value for p in positions.values())),
            })

        # Calculate metrics
        values = [d["value"] for d in daily_values]
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]

        final_value = Decimal(str(values[-1])) if values else self.initial_capital
        total_return = float((final_value - self.initial_capital) / self.initial_capital * 100)

        # Annualized return
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = ((float(final_value / self.initial_capital)) ** (1/years) - 1) * 100 if years > 0 else 0

        # Volatility (annualized)
        volatility = statistics.stdev(returns) * (252 ** 0.5) * 100 if len(returns) > 1 else 0

        # Sharpe Ratio
        excess_return = (annualized_return / 100) - self.RISK_FREE_RATE
        sharpe_ratio = excess_return / (volatility / 100) if volatility > 0 else 0

        # Max Drawdown
        peak = values[0]
        max_dd = 0
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Win Rate
        winning_trades = len([t for t in trades if t.action == "sell" and t.value > 0])
        total_sells = len([t for t in trades if t.action == "sell"])
        win_rate = (winning_trades / total_sells * 100) if total_sells > 0 else 0

        # Benchmark comparison
        benchmark_return = None
        alpha = None
        beta = None

        if benchmark_ticker and benchmark_ticker in price_data:
            bench_df = price_data[benchmark_ticker]
            bench_start = bench_df.loc[bench_df.index.date >= start_date, "close"].iloc[0]
            bench_end = bench_df.loc[bench_df.index.date <= end_date, "close"].iloc[-1]
            benchmark_return = float((bench_end - bench_start) / bench_start * 100)

            # Simple alpha (excess return over benchmark)
            alpha = total_return - benchmark_return

        return BacktestResult(
            strategy_name=strategy.name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_dd,
            win_rate=win_rate,
            total_trades=len(trades),
            trades=trades,
            daily_values=daily_values,
            positions=list(positions.values()),
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
        )

    async def compare_strategies(
        self,
        strategies: list,
        price_data: dict[str, pd.DataFrame],
        start_date: date,
        end_date: date,
    ) -> list[BacktestResult]:
        """Run multiple strategies and compare results."""
        results = []

        for strategy in strategies:
            result = await self.run(strategy, price_data, start_date, end_date)
            results.append(result)

        # Sort by total return
        results.sort(key=lambda x: x.total_return, reverse=True)

        return results
