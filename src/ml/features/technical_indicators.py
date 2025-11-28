"""
Technical indicators calculation using TA-Lib.

Implements common technical analysis indicators:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages (SMA, EMA)
- Stochastic Oscillator
- ATR (Average True Range)
- OBV (On-Balance Volume)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import talib


class TechnicalIndicators:
    """Calculate technical indicators for stock analysis."""

    @staticmethod
    def add_rsi(df: pd.DataFrame, periods: int = 14, column: str = "close") -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI) indicator.

        Args:
            df: DataFrame with OHLCV data
            periods: RSI period (default: 14)
            column: Column to calculate RSI on

        Returns:
            DataFrame with RSI column added
        """
        df = df.copy()
        df[f"rsi_{periods}"] = talib.RSI(df[column].values.astype(np.float64), timeperiod=periods)
        return df

    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = "close"
    ) -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence) indicator.

        Args:
            df: DataFrame with OHLCV data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            column: Column to calculate MACD on

        Returns:
            DataFrame with MACD columns added
        """
        df = df.copy()
        macd, macd_signal, macd_hist = talib.MACD(
            df[column].values.astype(np.float64),
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period
        )
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist
        return df

    @staticmethod
    def add_bollinger_bands(
        df: pd.DataFrame,
        periods: int = 20,
        std_dev: int = 2,
        column: str = "close"
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands indicator.

        Args:
            df: DataFrame with OHLCV data
            periods: MA period
            std_dev: Number of standard deviations
            column: Column to calculate Bollinger Bands on

        Returns:
            DataFrame with Bollinger Bands columns added
        """
        df = df.copy()
        upper, middle, lower = talib.BBANDS(
            df[column].values.astype(np.float64),
            timeperiod=periods,
            nbdevup=std_dev,
            nbdevdn=std_dev,
            matype=0
        )
        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower
        df["bb_width"] = (upper - lower) / middle
        df["bb_position"] = (df[column] - lower) / (upper - lower)
        return df

    @staticmethod
    def add_moving_averages(
        df: pd.DataFrame,
        periods: list = [5, 10, 20, 50, 200],
        column: str = "close"
    ) -> pd.DataFrame:
        """
        Add Simple Moving Averages (SMA) and Exponential Moving Averages (EMA).

        Args:
            df: DataFrame with OHLCV data
            periods: List of MA periods
            column: Column to calculate MAs on

        Returns:
            DataFrame with MA columns added
        """
        df = df.copy()
        values = df[column].values.astype(np.float64)
        for period in periods:
            df[f"sma_{period}"] = talib.SMA(values, timeperiod=period)
            df[f"ema_{period}"] = talib.EMA(values, timeperiod=period)
        return df

    @staticmethod
    def add_stochastic(
        df: pd.DataFrame,
        fastk_period: int = 14,
        slowk_period: int = 3,
        slowd_period: int = 3
    ) -> pd.DataFrame:
        """
        Add Stochastic Oscillator indicator.

        Args:
            df: DataFrame with OHLCV data
            fastk_period: Fast %K period
            slowk_period: Slow %K period
            slowd_period: Slow %D period

        Returns:
            DataFrame with Stochastic columns added
        """
        df = df.copy()
        slowk, slowd = talib.STOCH(
            df["high"].values.astype(np.float64),
            df["low"].values.astype(np.float64),
            df["close"].values.astype(np.float64),
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowk_matype=0,
            slowd_period=slowd_period,
            slowd_matype=0
        )
        df["stoch_k"] = slowk
        df["stoch_d"] = slowd
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, periods: int = 14) -> pd.DataFrame:
        """
        Add Average True Range (ATR) indicator.

        Args:
            df: DataFrame with OHLCV data
            periods: ATR period

        Returns:
            DataFrame with ATR column added
        """
        df = df.copy()
        df[f"atr_{periods}"] = talib.ATR(
            df["high"].values.astype(np.float64),
            df["low"].values.astype(np.float64),
            df["close"].values.astype(np.float64),
            timeperiod=periods
        )
        return df

    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add On-Balance Volume (OBV) indicator.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with OBV column added
        """
        df = df.copy()
        df["obv"] = talib.OBV(
            df["close"].values.astype(np.float64),
            df["volume"].values.astype(np.float64)
        )
        return df

    @staticmethod
    def add_adx(df: pd.DataFrame, periods: int = 14) -> pd.DataFrame:
        """
        Add Average Directional Index (ADX) indicator.

        Args:
            df: DataFrame with OHLCV data
            periods: ADX period

        Returns:
            DataFrame with ADX column added
        """
        df = df.copy()
        df[f"adx_{periods}"] = talib.ADX(
            df["high"].values.astype(np.float64),
            df["low"].values.astype(np.float64),
            df["close"].values.astype(np.float64),
            timeperiod=periods
        )
        return df

    @staticmethod
    def add_cci(df: pd.DataFrame, periods: int = 20) -> pd.DataFrame:
        """
        Add Commodity Channel Index (CCI) indicator.

        Args:
            df: DataFrame with OHLCV data
            periods: CCI period

        Returns:
            DataFrame with CCI column added
        """
        df = df.copy()
        df[f"cci_{periods}"] = talib.CCI(
            df["high"].values.astype(np.float64),
            df["low"].values.astype(np.float64),
            df["close"].values.astype(np.float64),
            timeperiod=periods
        )
        return df

    @classmethod
    def add_all_indicators(
        cls,
        df: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Add all technical indicators to the DataFrame.

        Args:
            df: DataFrame with OHLCV data
            config: Optional configuration dict for indicator parameters

        Returns:
            DataFrame with all indicators added
        """
        if config is None:
            config = {}

        df = cls.add_rsi(df, periods=config.get("rsi_period", 14))
        df = cls.add_macd(df)
        df = cls.add_bollinger_bands(df, periods=config.get("bb_period", 20))
        df = cls.add_moving_averages(df, periods=config.get("ma_periods", [5, 10, 20, 50, 200]))
        df = cls.add_stochastic(df)
        df = cls.add_atr(df, periods=config.get("atr_period", 14))
        df = cls.add_obv(df)
        df = cls.add_adx(df, periods=config.get("adx_period", 14))
        df = cls.add_cci(df, periods=config.get("cci_period", 20))

        return df

    @staticmethod
    def get_feature_names() -> list:
        """
        Get list of all feature names generated by indicators.

        Returns:
            List of feature column names
        """
        return [
            "rsi_14",
            "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position",
            "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
            "ema_5", "ema_10", "ema_20", "ema_50", "ema_200",
            "stoch_k", "stoch_d",
            "atr_14",
            "obv",
            "adx_14",
            "cci_20"
        ]
