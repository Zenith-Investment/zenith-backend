"""Financial calculations for portfolio analytics."""
import math
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Sequence

import numpy as np
import structlog

logger = structlog.get_logger()

# Risk-free rate (SELIC approximation for Brazil - adjust as needed)
RISK_FREE_RATE = 0.1075  # 10.75% annual


def calculate_returns(prices: Sequence[float | Decimal]) -> list[float]:
    """
    Calculate daily returns from a price series.

    Returns = (P_t - P_{t-1}) / P_{t-1}
    """
    if len(prices) < 2:
        return []

    prices_float = [float(p) for p in prices]
    returns = []

    for i in range(1, len(prices_float)):
        if prices_float[i - 1] != 0:
            daily_return = (prices_float[i] - prices_float[i - 1]) / prices_float[i - 1]
            returns.append(daily_return)

    return returns


def calculate_annualized_return(
    start_value: Decimal | float,
    end_value: Decimal | float,
    days: int
) -> float | None:
    """
    Calculate annualized return.

    Annualized Return = ((End Value / Start Value) ^ (365 / days)) - 1
    """
    if days <= 0 or float(start_value) <= 0:
        return None

    start = float(start_value)
    end = float(end_value)

    try:
        total_return = end / start
        annualized = (total_return ** (365 / days)) - 1
        return round(annualized * 100, 2)  # Return as percentage
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def calculate_volatility(returns: Sequence[float], annualize: bool = True) -> float | None:
    """
    Calculate volatility (standard deviation of returns).

    Annualized Volatility = Daily Std Dev * sqrt(252)
    252 = typical trading days per year
    """
    if len(returns) < 2:
        return None

    try:
        std_dev = np.std(returns, ddof=1)  # Sample standard deviation

        if annualize:
            # Annualize using square root of trading days
            annualized_vol = std_dev * math.sqrt(252)
            return round(annualized_vol * 100, 2)  # Return as percentage

        return round(std_dev * 100, 2)
    except Exception as e:
        logger.warning("Error calculating volatility", error=str(e))
        return None


def calculate_sharpe_ratio(
    returns: Sequence[float],
    risk_free_rate: float = RISK_FREE_RATE
) -> float | None:
    """
    Calculate Sharpe Ratio.

    Sharpe Ratio = (Portfolio Return - Risk Free Rate) / Portfolio Std Dev

    Uses annualized values for consistency.
    """
    if len(returns) < 2:
        return None

    try:
        # Annualized return from daily returns
        mean_daily_return = np.mean(returns)
        annualized_return = mean_daily_return * 252

        # Annualized volatility
        daily_std = np.std(returns, ddof=1)
        annualized_vol = daily_std * math.sqrt(252)

        if annualized_vol == 0:
            return None

        sharpe = (annualized_return - risk_free_rate) / annualized_vol
        return round(sharpe, 2)
    except Exception as e:
        logger.warning("Error calculating Sharpe ratio", error=str(e))
        return None


def calculate_max_drawdown(values: Sequence[float | Decimal]) -> float | None:
    """
    Calculate Maximum Drawdown.

    Max Drawdown = (Trough Value - Peak Value) / Peak Value

    Represents the largest peak-to-trough decline.
    """
    if len(values) < 2:
        return None

    try:
        values_float = [float(v) for v in values]

        peak = values_float[0]
        max_drawdown = 0.0

        for value in values_float[1:]:
            if value > peak:
                peak = value

            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        return round(max_drawdown * 100, 2)  # Return as percentage
    except Exception as e:
        logger.warning("Error calculating max drawdown", error=str(e))
        return None


def calculate_sortino_ratio(
    returns: Sequence[float],
    risk_free_rate: float = RISK_FREE_RATE
) -> float | None:
    """
    Calculate Sortino Ratio (focuses only on downside volatility).

    Sortino Ratio = (Portfolio Return - Risk Free Rate) / Downside Deviation
    """
    if len(returns) < 2:
        return None

    try:
        mean_daily_return = np.mean(returns)
        annualized_return = mean_daily_return * 252

        # Calculate downside deviation (only negative returns)
        negative_returns = [r for r in returns if r < 0]

        if len(negative_returns) < 2:
            return None

        downside_std = np.std(negative_returns, ddof=1)
        annualized_downside = downside_std * math.sqrt(252)

        if annualized_downside == 0:
            return None

        sortino = (annualized_return - risk_free_rate) / annualized_downside
        return round(sortino, 2)
    except Exception as e:
        logger.warning("Error calculating Sortino ratio", error=str(e))
        return None


def calculate_beta(
    portfolio_returns: Sequence[float],
    market_returns: Sequence[float]
) -> float | None:
    """
    Calculate Beta (portfolio sensitivity to market movements).

    Beta = Covariance(Portfolio, Market) / Variance(Market)
    """
    if len(portfolio_returns) < 2 or len(market_returns) < 2:
        return None

    if len(portfolio_returns) != len(market_returns):
        # Truncate to matching length
        min_len = min(len(portfolio_returns), len(market_returns))
        portfolio_returns = portfolio_returns[:min_len]
        market_returns = market_returns[:min_len]

    try:
        covariance = np.cov(portfolio_returns, market_returns)[0][1]
        market_variance = np.var(market_returns, ddof=1)

        if market_variance == 0:
            return None

        beta = covariance / market_variance
        return round(beta, 2)
    except Exception as e:
        logger.warning("Error calculating Beta", error=str(e))
        return None


def calculate_cagr(
    start_value: Decimal | float,
    end_value: Decimal | float,
    years: float
) -> float | None:
    """
    Calculate Compound Annual Growth Rate (CAGR).

    CAGR = (End Value / Start Value) ^ (1 / years) - 1
    """
    if years <= 0 or float(start_value) <= 0:
        return None

    try:
        start = float(start_value)
        end = float(end_value)

        cagr = (end / start) ** (1 / years) - 1
        return round(cagr * 100, 2)  # Return as percentage
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def get_period_days(period: str) -> int:
    """Convert period string to number of days."""
    period_map = {
        "1d": 1,
        "1w": 7,
        "1m": 30,
        "3m": 90,
        "6m": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "ytd": (datetime.now() - datetime(datetime.now().year, 1, 1)).days,
        "all": 3650,  # ~10 years
    }
    return period_map.get(period.lower(), 365)


def get_period_start_date(period: str) -> datetime:
    """Get the start date for a given period."""
    days = get_period_days(period)
    return datetime.now() - timedelta(days=days)
