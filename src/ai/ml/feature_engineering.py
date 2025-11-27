"""
Feature Engineering for financial ML models.

Creates sophisticated features from raw price data for optimal ML performance.
"""
from typing import Optional
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()


class FeatureEngineer:
    """
    Creates features for ML models from OHLCV data.

    Features include:
    - Technical indicators (MA, RSI, MACD, Bollinger Bands)
    - Price patterns
    - Volatility measures
    - Volume analysis
    - Momentum indicators
    - Lagged features
    """

    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set from OHLCV data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume

        Returns:
            DataFrame with all features
        """
        features = df.copy()

        # Price features
        features = self._add_price_features(features)

        # Moving averages
        features = self._add_moving_averages(features)

        # Momentum indicators
        features = self._add_momentum_indicators(features)

        # Volatility features
        features = self._add_volatility_features(features)

        # Volume features
        features = self._add_volume_features(features)

        # Lagged features
        features = self._add_lagged_features(features)

        # Pattern features
        features = self._add_pattern_features(features)

        # Drop rows with NaN
        features = features.dropna()

        return features

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price features."""
        # Returns
        df["return_1d"] = df["close"].pct_change()
        df["return_5d"] = df["close"].pct_change(5)
        df["return_10d"] = df["close"].pct_change(10)
        df["return_20d"] = df["close"].pct_change(20)

        # Log returns
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # Price position
        df["high_low_ratio"] = df["high"] / df["low"]
        df["close_open_ratio"] = df["close"] / df["open"]

        # Gap
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        # Range
        df["range_pct"] = (df["high"] - df["low"]) / df["close"]

        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average features."""
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 200]:
            df[f"sma_{period}"] = df["close"].rolling(window=period).mean()
            df[f"sma_{period}_ratio"] = df["close"] / df[f"sma_{period}"]

        # Exponential Moving Averages
        for period in [12, 26, 50]:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
            df[f"ema_{period}_ratio"] = df["close"] / df[f"ema_{period}"]

        # MA crossovers
        df["sma_5_20_cross"] = (df["sma_5"] > df["sma_20"]).astype(int)
        df["sma_20_50_cross"] = (df["sma_20"] > df["sma_50"]).astype(int)
        df["golden_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
        df["rsi_overbought"] = (df["rsi"] > 70).astype(int)

        # MACD
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["macd_cross"] = (df["macd"] > df["macd_signal"]).astype(int)

        # Stochastic Oscillator
        low_14 = df["low"].rolling(window=14).min()
        high_14 = df["high"].rolling(window=14).max()
        df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14)
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

        # Williams %R
        df["williams_r"] = -100 * (high_14 - df["close"]) / (high_14 - low_14)

        # Rate of Change (ROC)
        df["roc_10"] = ((df["close"] - df["close"].shift(10)) / df["close"].shift(10)) * 100
        df["roc_20"] = ((df["close"] - df["close"].shift(20)) / df["close"].shift(20)) * 100

        # Momentum
        df["momentum_10"] = df["close"] - df["close"].shift(10)
        df["momentum_20"] = df["close"] - df["close"].shift(20)

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        # Historical volatility
        df["volatility_10"] = df["return_1d"].rolling(window=10).std() * np.sqrt(252)
        df["volatility_20"] = df["return_1d"].rolling(window=20).std() * np.sqrt(252)
        df["volatility_60"] = df["return_1d"].rolling(window=60).std() * np.sqrt(252)

        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # Average True Range (ATR)
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(window=14).mean()
        df["atr_ratio"] = df["atr_14"] / df["close"]

        # Keltner Channels
        df["kc_middle"] = df["close"].ewm(span=20, adjust=False).mean()
        df["kc_upper"] = df["kc_middle"] + (df["atr_14"] * 2)
        df["kc_lower"] = df["kc_middle"] - (df["atr_14"] * 2)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume features."""
        # Volume moving averages
        df["volume_sma_10"] = df["volume"].rolling(window=10).mean()
        df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # On-Balance Volume (OBV)
        obv = [0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i-1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i-1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["obv"] = obv
        df["obv_sma"] = pd.Series(obv).rolling(window=20).mean().values

        # Volume Price Trend (VPT)
        df["vpt"] = (df["volume"] * df["close"].pct_change()).cumsum()

        # Money Flow Index (MFI)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        money_flow = typical_price * df["volume"]

        positive_flow = []
        negative_flow = []
        for i in range(1, len(df)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.append(money_flow.iloc[i])
                negative_flow.append(0)
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                negative_flow.append(money_flow.iloc[i])
                positive_flow.append(0)
            else:
                positive_flow.append(0)
                negative_flow.append(0)

        positive_flow = [0] + positive_flow
        negative_flow = [0] + negative_flow

        positive_mf = pd.Series(positive_flow).rolling(window=14).sum()
        negative_mf = pd.Series(negative_flow).rolling(window=14).sum()

        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
        df["mfi"] = mfi.values

        return df

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features for time series."""
        # Lagged returns
        for lag in [1, 2, 3, 5, 10]:
            df[f"return_lag_{lag}"] = df["return_1d"].shift(lag)
            df[f"close_lag_{lag}"] = df["close"].shift(lag)

        # Lagged volatility
        df["volatility_lag_5"] = df["volatility_20"].shift(5)

        # Lagged RSI
        df["rsi_lag_1"] = df["rsi"].shift(1)
        df["rsi_lag_5"] = df["rsi"].shift(5)

        return df

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features."""
        # Body size
        body = df["close"] - df["open"]
        df["body_size"] = abs(body) / df["open"]
        df["body_direction"] = np.sign(body)

        # Shadow sizes
        df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"]
        df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"]

        # Doji pattern (body < 10% of range)
        range_size = df["high"] - df["low"]
        df["doji"] = (abs(body) < (range_size * 0.1)).astype(int)

        # Hammer pattern
        df["hammer"] = (
            (df["lower_shadow"] > (abs(body) * 2)) &
            (df["upper_shadow"] < abs(body))
        ).astype(int)

        # Engulfing pattern
        df["bullish_engulfing"] = (
            (body.shift(1) < 0) &
            (body > 0) &
            (df["open"] < df["close"].shift(1)) &
            (df["close"] > df["open"].shift(1))
        ).astype(int)

        df["bearish_engulfing"] = (
            (body.shift(1) > 0) &
            (body < 0) &
            (df["open"] > df["close"].shift(1)) &
            (df["close"] < df["open"].shift(1))
        ).astype(int)

        # Trend strength (consecutive up/down days)
        up_days = (df["close"] > df["close"].shift(1)).astype(int)
        down_days = (df["close"] < df["close"].shift(1)).astype(int)

        df["consecutive_up"] = up_days.groupby((up_days != up_days.shift()).cumsum()).cumsum()
        df["consecutive_down"] = down_days.groupby((down_days != down_days.shift()).cumsum()).cumsum()

        return df

    def create_target(
        self,
        df: pd.DataFrame,
        target_days: int = 5,
        target_type: str = "return",
    ) -> pd.Series:
        """
        Create prediction target.

        Args:
            df: DataFrame with close prices
            target_days: Days ahead to predict
            target_type: "return" for % return, "direction" for up/down

        Returns:
            Series with target values
        """
        future_close = df["close"].shift(-target_days)

        if target_type == "return":
            target = (future_close - df["close"]) / df["close"]
        elif target_type == "direction":
            target = (future_close > df["close"]).astype(int)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        return target

    def get_feature_names(self) -> list[str]:
        """Get list of feature names (excluding target and price columns)."""
        exclude = ["open", "high", "low", "close", "volume", "target"]

        # Create dummy df to get feature names
        dummy = pd.DataFrame({
            "open": [100] * 300,
            "high": [101] * 300,
            "low": [99] * 300,
            "close": [100.5] * 300,
            "volume": [1000000] * 300,
        })

        features = self.create_features(dummy)
        return [col for col in features.columns if col not in exclude]


# Global instance
feature_engineer = FeatureEngineer()
